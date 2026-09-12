"""
Step 4. Which experience attributes move the outcomes, and by how much.

Two outcomes, different in kind:

  1. `review_scores_rating` - what guests say after a stay.
  2. **Price premium vs comparable set** - what the listing can charge. Comp
     set is same neighbourhood group x room type x capacity band. The premium
     is log price minus the *leave-one-out* median log price of that set.

The second exists because the first is nearly saturated (see
`analysis_corpus.py`: 87% of listings score 90+). Price has room to vary.

Every model here is a **conditional association**. A host who keeps a spotless
flat is also, on average, one who writes an accurate listing, replies within
the hour and prices sensibly. Nothing in an observational cross-section
separates "cleanliness raises the rating" from "conscientious hosts do
everything well, and cleanliness is the part guests write down".

Outputs
  data/processed/model_frame.parquet
  figures/06_rating_drivers.png
  figures/07_model_comparison.png
  figures/08_premium_drivers.png
  tables/driver_coefficients.csv
  tables/comp_sets.csv
  driver_metrics.json
"""
from __future__ import annotations

import json
import textwrap

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as C
import viz
from lexicon import ASPECTS, LABELS

viz.use_style()

ALPHAS = np.logspace(-2, 3, 40)

CONTROL_LABELS = {
    "log_accommodates": "Capacity (log)",
    "bedrooms": "Bedrooms",
    "log_reviews": "Review count (log)",
    "host_tenure_years": "Host tenure (years)",
    "n_amenities": "Amenities listed",
    "log_price": "Nightly price (log)",
    "min_nights": "Minimum nights",
    "room_Private room": "Room type: private room",
    "room_Shared room": "Room type: shared room",
}


def comp_set_premium(listings: pd.DataFrame) -> pd.DataFrame:
    """
    Price premium against a leave-one-out comp-set median.

    With a plain group median a comp set of one produces a premium of exactly
    0.0, because the listing *is* the market. Those rows look like average
    listings and drag every coefficient towards zero. A comp set with fewer
    than MIN_COMP_SET peers returns NaN and drops out of the premium model
    rather than getting a fabricated baseline.
    """
    df = listings.copy()
    keys = ["neighbourhood_group_cleansed", "room_type", "accommodates_band"]
    g = df.groupby(keys, observed=True)["log_price"]

    # Leave-one-out median: recomputed per row from the peers only. The
    # median is not a leave-one-out-friendly statistic the way a mean is.
    def loo_median(s: pd.Series) -> pd.Series:
        arr = s.to_numpy()
        out = np.empty(arr.size)
        for i in range(arr.size):
            peers = np.delete(arr, i)
            out[i] = np.median(peers) if peers.size else np.nan
        return pd.Series(out, index=s.index)

    df["comp_n"] = g.transform("size") - 1          # peers, excluding self
    df["comp_median_log_price"] = g.transform(loo_median)
    df["price_premium"] = df["log_price"] - df["comp_median_log_price"]
    df.loc[df["comp_n"] < C.MIN_COMP_SET, ["price_premium", "comp_median_log_price"]] = np.nan
    return df


def build_frame(listings: pd.DataFrame, aspects: pd.DataFrame,
                min_reviews: int) -> pd.DataFrame:
    df = comp_set_premium(listings).merge(
        aspects, left_on="id", right_on="listing_id", how="inner")
    df = df[df["n_reviews_scored"] >= min_reviews].copy()

    df["log_accommodates"] = np.log(df["accommodates"])
    df["log_reviews"] = np.log(df["n_reviews_scored"])
    df["host_tenure_years"] = df["host_tenure_days"] / 365.25
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["min_nights"] = df["minimum_nights"].clip(upper=30)
    for rt in ("Private room", "Shared room"):
        df[f"room_{rt}"] = (df["room_type"] == rt).astype(float)
    return df


def aspect_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Two features per aspect, and an account of what is imputed.

    - `sent_<a>`  mean VADER sentiment over the listing's mentions. Undefined
      below MIN_ASPECT_MENTIONS mentions, and filled with the column mean,
      which standardises to zero, so the coefficient is estimated off the
      listings that have the measurement.
    - `talk_<a>`  share of the listing's reviews mentioning the aspect at all.
      Always defined. It carries what the imputation throws away: a listing
      nobody mentions value for money on differs from one where everybody
      does.
    """
    out = pd.DataFrame(index=df.index)
    coverage: dict[str, float] = {}
    for a in ASPECTS:
        s = df[f"sentiment__{a}"]
        coverage[a] = float(s.notna().mean())
        out[f"sent_{a}"] = s.fillna(s.mean())
        out[f"talk_{a}"] = df[f"mention_rate__{a}"]
    return out, coverage


def control_features(df: pd.DataFrame, include_price: bool) -> pd.DataFrame:
    cols = ["log_accommodates", "bedrooms", "log_reviews", "host_tenure_years",
            "n_amenities", "min_nights", "room_Private room", "room_Shared room"]
    if include_price:
        cols.append("log_price")
    X = df[cols].copy()
    # Neighbourhood group as dummies. Dropped for the premium model. The comp
    # set already differences it out, and putting it back nets out the same
    # variation twice.
    if include_price:
        d = pd.get_dummies(df["neighbourhood_group_cleansed"],
                           prefix="area", drop_first=True, dtype=float)
        X = pd.concat([X, d], axis=1)
    return X.astype(float)


def fit_ridge(X: pd.DataFrame, y: pd.Series, seed: int
              ) -> tuple[dict, pd.DataFrame, float, float]:
    """
    RidgeCV on standardised features, with bootstrap confidence intervals.

    y stays in native units, so a coefficient reads as "outcome units per one
    standard deviation of this feature" - rating points, or log price points.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=seed)

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    cv = RidgeCV(alphas=ALPHAS).fit(Ztr, ytr)
    alpha = float(cv.alpha_)
    r2_test = float(r2_score(yte, cv.predict(Zte)))
    r2_train = float(r2_score(ytr, cv.predict(Ztr)))

    # Bootstrap the whole standardise-and-fit on the training rows. Alpha is
    # held at the cross-validated value. Re-tuning inside every resample would
    # widen the intervals with tuning noise rather than sampling noise.
    rng = np.random.default_rng(seed)
    boots = np.empty((C.N_BOOTSTRAP, X.shape[1]))
    Xtr_np, ytr_np = Xtr.to_numpy(), ytr.to_numpy()
    for b in range(C.N_BOOTSTRAP):
        idx = rng.integers(0, len(Xtr_np), len(Xtr_np))
        sc_b = StandardScaler().fit(Xtr_np[idx])
        boots[b] = Ridge(alpha=alpha).fit(sc_b.transform(Xtr_np[idx]),
                                          ytr_np[idx]).coef_

    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef": cv.coef_,
        "ci_lo": np.percentile(boots, 2.5, axis=0),
        "ci_hi": np.percentile(boots, 97.5, axis=0),
    })
    coefs["significant"] = (coefs["ci_lo"] > 0) | (coefs["ci_hi"] < 0)
    info = {"alpha": alpha, "r2_test": r2_test, "r2_train": r2_train,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    return info, coefs, r2_test, r2_train


def ols_crosscheck(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Unregularised OLS with heteroskedasticity-robust (HC3) standard errors.

    Ridge shrinks coefficients towards zero, and shrinkage can manufacture a
    ranking. The same design without a penalty checks the ranking is in the
    data. Where the two disagree, the disagreement is reported, not resolved.
    """
    Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns,
                     index=X.index)
    m = sm.OLS(y.to_numpy(), sm.add_constant(Z)).fit(cov_type="HC3")
    ci = m.conf_int()
    return pd.DataFrame({
        "feature": X.columns,
        "ols_coef": m.params.drop("const").to_numpy(),
        "ols_ci_lo": ci.drop("const")[0].to_numpy(),
        "ols_ci_hi": ci.drop("const")[1].to_numpy(),
        "ols_p": m.pvalues.drop("const").to_numpy(),
    })


def fit_gbm(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[float, pd.Series]:
    """
    Gradient boosting on the same design, scored on the same held-out rows.

    Early stopping runs against a validation slice cut from the *training*
    rows. Stopping on the test set is a leak that flatters the reported R²,
    and the sklearn wrapper's `eval_set` makes it easy to do by accident. The
    core `lgb.train` API is stable across LightGBM 4.x, where the wrapper's
    eval arguments were renamed.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=seed)
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed)

    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15,
              "min_data_in_leaf": 30, "bagging_fraction": 0.85, "bagging_freq": 1,
              "feature_fraction": 0.8, "lambda_l2": 1.0, "verbosity": -1,
              "seed": seed, "deterministic": True, "force_row_wise": True}
    booster = lgb.train(
        params, lgb.Dataset(Xf, yf), num_boost_round=1200,
        valid_sets=[lgb.Dataset(Xv, yv)],
        callbacks=[lgb.early_stopping(60, verbose=False)])

    r2 = float(r2_score(yte, booster.predict(Xte)))
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns)
    return r2, imp / imp.sum()


def run_outcome(name: str, df: pd.DataFrame, y: pd.Series, include_price: bool,
                seed: int) -> dict:
    """Controls-only, then controls+aspects, for both model families."""
    A, coverage = aspect_features(df)
    K = control_features(df, include_price=include_price)
    full = pd.concat([A, K], axis=1)

    ctrl_info, _, r2_ctrl, _ = fit_ridge(K, y, seed)
    info, coefs, r2_full, r2_train = fit_ridge(full, y, seed)
    gbm_ctrl, _ = fit_gbm(K, y, seed)
    gbm_full, gbm_imp = fit_gbm(full, y, seed)

    coefs = coefs.merge(ols_crosscheck(full, y), on="feature", how="left")
    coefs["outcome"] = name
    coefs["is_aspect"] = coefs["feature"].str.startswith(("sent_", "talk_"))
    coefs["gbm_gain_share"] = coefs["feature"].map(gbm_imp)
    asp = coefs[coefs["is_aspect"]]
    agree_sign = float((np.sign(asp["coef"]) == np.sign(asp["ols_coef"])).mean())
    agree_rank = float(asp["coef"].corr(asp["ols_coef"], method="spearman"))

    aspect_gain = float(gbm_imp[[c for c in gbm_imp.index
                                 if c.startswith(("sent_", "talk_"))]].sum())
    return {
        "name": name, "n": int(len(y)), "coefs": coefs,
        "alpha": info["alpha"],
        "ridge_r2_controls_only": r2_ctrl,
        "ridge_r2_full": r2_full,
        "ridge_r2_train": r2_train,
        "ridge_r2_uplift_from_aspects": r2_full - r2_ctrl,
        "gbm_r2_controls_only": gbm_ctrl,
        "gbm_r2_full": gbm_full,
        "gbm_r2_uplift_from_aspects": gbm_full - gbm_ctrl,
        "gbm_aspect_gain_share": aspect_gain,
        "ols_sign_agreement_on_aspects": agree_sign,
        "ols_rank_agreement_on_aspects": agree_rank,
        "ols_significant_aspects": int((asp["ols_p"] < 0.05).sum()),
        "ridge_significant_aspects": int(asp["significant"].sum()),
        "aspect_coverage": coverage,
        "ridge_alpha_controls": ctrl_info["alpha"],
    }


def pretty(feature: str) -> str:
    if feature.startswith("sent_"):
        return f"{LABELS[feature[5:]]} - sentiment"
    if feature.startswith("talk_"):
        return f"{LABELS[feature[5:]]} - how often mentioned"
    if feature.startswith("area_"):
        return f"Area: {feature[5:]}"
    return CONTROL_LABELS.get(feature, feature)


def coefficient_chart(res: dict, path, headline: str, sub: str, unit: str,
                      source: str) -> None:
    c = res["coefs"]
    c = c[c["is_aspect"]].sort_values("coef")
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    y = np.arange(len(c))
    cols = [(viz.BLUE if v > 0 else viz.ORANGE) if sig else viz.INK_MUTED
            for v, sig in zip(c["coef"], c["significant"], strict=True)]
    ax.hlines(y, c["ci_lo"], c["ci_hi"], color=cols, linewidth=2.4, alpha=0.5)
    ax.scatter(c["coef"], y, s=46, color=cols, zorder=3,
               edgecolor=viz.SURFACE, linewidth=1.2)
    ax.axvline(0, color=viz.INK, linewidth=1.0)
    ax.set_yticks(y, [pretty(f) for f in c["feature"]])
    ax.set_xlabel(unit)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.035)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=viz.BLUE),
               plt.Line2D([], [], marker="o", linestyle="none", color=viz.ORANGE),
               plt.Line2D([], [], marker="o", linestyle="none", color=viz.INK_MUTED)]
    ax.legend(handles, ["Positive, CI excludes zero",
                        "Negative, CI excludes zero",
                        "CI includes zero"],
              loc="lower right", ncols=1)
    viz.title(ax, headline, sub)
    # viz.title clears the subtitle by 1.5% of the axes height. On an axes this
    # tall that is under a point, so the bold title lands on top of it. The
    # title is re-set with a larger pad rather than editing the house module.
    ax.set_title(headline, loc="left", pad=22)
    viz.source(fig, textwrap.fill(source, 108))
    viz.save(fig, path)


def main() -> None:
    listings = pd.read_parquet(C.LISTINGS_PARQUET)
    aspects = pd.read_parquet(C.ASPECTS_LISTING_PARQUET)

    df = build_frame(listings, aspects, C.MIN_REVIEWS_FOR_MODEL)
    df.to_parquet(C.MODEL_FRAME_PARQUET, index=False)

    comp_tab = (df.groupby(["neighbourhood_group_cleansed", "room_type",
                            "accommodates_band"], observed=True)
                  .agg(listings=("id", "size"),
                       median_price=("price", "median"),
                       usable=("price_premium", lambda s: int(s.notna().sum())))
                  .reset_index().sort_values("listings", ascending=False))
    comp_tab.to_csv(C.TAB / "comp_sets.csv", index=False)

    rating = df[df["review_scores_rating"].notna()]
    prem = df[df["price_premium"].notna()]
    print(f"Rating model n={len(rating):,}; premium model n={len(prem):,} "
          f"({1 - len(prem) / len(df):.1%} of listings have no comp set of "
          f"{C.MIN_COMP_SET}+)")

    res_rating = run_outcome("rating", rating, rating["review_scores_rating"],
                             include_price=True, seed=C.RANDOM_STATE)
    res_prem = run_outcome("price_premium", prem, prem["price_premium"],
                           include_price=False, seed=C.RANDOM_STATE)

    all_coefs = pd.concat([res_rating["coefs"], res_prem["coefs"]])
    all_coefs["label"] = all_coefs["feature"].map(pretty)
    all_coefs.to_csv(C.TAB / "driver_coefficients.csv", index=False)

    # fig 6
    top = res_rating["coefs"].query("is_aspect").reindex(
        res_rating["coefs"].query("is_aspect")["coef"].abs().sort_values(
            ascending=False).index).iloc[0]
    coefficient_chart(
        res_rating, C.FIG / "06_rating_drivers.png",
        f"{pretty(top['feature']).split(' - ')[0]} carries the largest single "
        f"association with rating",
        "Rating points per 1 SD, holding property, area, price and review count fixed",
        "Change in overall rating (points) per 1 SD",
        f"Association, not effect. Ridge on {res_rating['n']:,} listings; bars are "
        f"95% bootstrap intervals over {C.N_BOOTSTRAP} resamples of the training "
        "rows. Property and area controls are fitted but not shown - see "
        "outputs/tables/driver_coefficients.csv.")

    # fig 7
    # Both model families are shown with and without the aspect block. Pairing
    # a Ridge controls-only bar against a LightGBM full bar would flatter the
    # text. On price, LightGBM is *worse* with the aspects in.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), sharey=True)
    panels = [("Overall rating (0-100)", res_rating),
              ("Price premium vs comp set", res_prem)]
    families = ["Ridge", "LightGBM"]
    for ax, (heading, res) in zip(axes, panels, strict=True):
        vals = {
            "Controls only": [res["ridge_r2_controls_only"],
                              res["gbm_r2_controls_only"]],
            "Controls + review text": [res["ridge_r2_full"], res["gbm_r2_full"]],
        }
        x = np.arange(len(families))
        w = 0.33
        for i, ((k, v), col) in enumerate(zip(vals.items(),
                                              [viz.INK_MUTED, viz.BLUE],
                                              strict=True)):
            pos = x + (i - 0.5) * w
            ax.bar(pos, v, width=w * 0.88, color=col, label=k)
            for px, pv in zip(pos, v, strict=True):
                ax.text(px, max(pv, 0) + 0.008, f"{pv:.2f}", ha="center",
                        fontsize=9, color=viz.INK_2)
        ax.set_xticks(x, families)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 0.54)
        viz.title(ax, heading, f"n = {res['n']:,} listings")
    axes[0].set_ylabel("Out-of-sample R-squared")
    axes[0].legend(loc="upper left")
    up_r = res_rating["ridge_r2_full"] - res_rating["ridge_r2_controls_only"]
    up_p = res_prem["gbm_r2_full"] - res_prem["gbm_r2_controls_only"]
    fig.suptitle(f"Review text adds {up_r:+.2f} R-squared on rating and nothing on price",
                 x=0.0, ha="left", fontsize=13.5, fontweight="bold",
                 color=viz.INK, y=1.10)
    viz.source(fig, textwrap.fill(
        f"Held-out {C.TEST_SIZE:.0%} of listings, split once at seed "
        f"{C.RANDOM_STATE}. On price the gradient-boosted model is "
        f"{abs(up_p):.2f} R-squared *worse* with the text features in, which "
        "is the honest reading: there is no price signal here for it to "
        "find.", 108))
    fig.subplots_adjust(wspace=0.10)
    viz.save(fig, C.FIG / "07_model_comparison.png")

    # fig 8
    ptop = res_prem["coefs"].query("is_aspect and significant")
    lead = (ptop.reindex(ptop["coef"].abs().sort_values(ascending=False).index)
            .iloc[0] if len(ptop) else None)
    if lead is None:
        head = "No aspect clears its confidence interval on price"
    else:
        pct_move = float(np.expm1(abs(lead["coef"])))
        spread = float(np.expm1(prem["price_premium"].std()))
        head = (f"The biggest text effect on price is {pct_move:.1%} per SD, "
                f"against a {spread:.0%} price spread")
    coefficient_chart(
        res_prem, C.FIG / "08_premium_drivers.png", head,
        "Log price premium points per 1 SD",
        "Change in log price premium per 1 SD",
        f"Premium is log price minus the leave-one-out median log price of the "
        f"same neighbourhood group x room type x capacity band. Ridge on "
        f"{res_prem['n']:,} listings with a comp set of {C.MIN_COMP_SET}+ peers.")

    # sensitivity
    # The MIN_REVIEWS_FOR_MODEL threshold is a judgement call. It is swept.
    sens = {}
    for k in (C.MIN_REVIEWS_SENSITIVITY[0], C.MIN_REVIEWS_FOR_MODEL,
              C.MIN_REVIEWS_SENSITIVITY[1]):
        d = build_frame(listings, aspects, k)
        d = d[d["review_scores_rating"].notna()]
        r = run_outcome(f"rating_min{k}", d, d["review_scores_rating"],
                        include_price=True, seed=C.RANDOM_STATE)
        top_asp = (r["coefs"].query("is_aspect")
                   .set_index("feature")["coef"].abs().idxmax())
        sens[f"min_reviews_{k}"] = {
            "n": r["n"], "ridge_r2_full": r["ridge_r2_full"],
            "ridge_r2_controls_only": r["ridge_r2_controls_only"],
            "top_aspect_feature": top_asp,
            "top_aspect_coef": float(
                r["coefs"].set_index("feature").loc[top_asp, "coef"]),
        }

    # metrics
    def coef_dict(res: dict) -> dict:
        c = res["coefs"].set_index("feature")
        return {f: {"coef": float(r["coef"]), "ci_lo": float(r["ci_lo"]),
                    "ci_hi": float(r["ci_hi"]),
                    "significant": bool(r["significant"]),
                    "ols_coef": float(r["ols_coef"]),
                    "ols_p": float(r["ols_p"]),
                    "gbm_gain_share": (None if pd.isna(r["gbm_gain_share"])
                                       else float(r["gbm_gain_share"]))}
                for f, r in c.iterrows()}

    metrics = {
        "min_reviews_for_model": C.MIN_REVIEWS_FOR_MODEL,
        "listings_in_frame": int(len(df)),
        "rating": {k: v for k, v in res_rating.items() if k != "coefs"},
        "price_premium": {k: v for k, v in res_prem.items() if k != "coefs"},
        "rating_coefficients": coef_dict(res_rating),
        "premium_coefficients": coef_dict(res_prem),
        "comp_sets_total": int(len(comp_tab)),
        "listings_without_comp_set": int(df["price_premium"].isna().sum()),
        "listings_without_comp_set_share": float(df["price_premium"].isna().mean()),
        "premium_sd": float(prem["price_premium"].std()),
        "rating_sd": float(rating["review_scores_rating"].std()),
        "sensitivity_min_reviews": sens,
    }
    (C.OUT / "driver_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items()
                      if not k.endswith("coefficients")}, indent=2)[:3000])


if __name__ == "__main__":
    main()
