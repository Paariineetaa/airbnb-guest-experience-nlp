"""
Step 3. Aspect-based sentiment. 83,000 paragraphs into eight numbers per
listing, checked against an independent measurement.

How it works
------------
1. Split each review into sentences. Guests use line breaks as punctuation,
   so the splitter treats them as terminators.
2. Match each sentence against the eight aspect patterns in `lexicon.py`. Most
   sentences carry one aspect or none.
3. Score the *matching sentences only* with VADER, not the whole review. "The
   flat was spotless but the street was deafening" is positive on cleanliness
   and negative on noise.
4. Roll up to listing level: mention rate, mean sentiment, negative-mention
   rate.

Then **convergent validity**. Six of the eight aspects have an Airbnb subscore
measuring the same construct, collected separately from the free text. If the
derived cleanliness sentiment does not track `review_scores_cleanliness`, the
extraction is not measuring cleanliness. Those correlations are reported
either way.

Outputs
  data/processed/aspect_review.parquet
  data/processed/aspect_listing.parquet
  figures/04_aspect_prevalence.png
  figures/05_aspect_validation.png
  tables/aspect_validation.csv
  tables/lexicon_discovery.csv
  aspect_metrics.json
"""
from __future__ import annotations

import json
import textwrap
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config as C
import text_utils as T
import viz
from lexicon import (
    ASPECTS,
    GENERIC_HEADS,
    LABELS,
    SUBSCORE,
    TERMS,
    compile_patterns,
)

viz.use_style()


def score_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (review, aspect) mention.

    VADER is deterministic and stateless, so identical sentences are cached.
    Roughly one sentence in eight is a repeat ("Great place."). The cache
    keeps this step under a minute on 83,000 reviews.
    """
    patterns = compile_patterns()
    analyser = SentimentIntensityAnalyzer()
    cache: dict[str, float] = {}

    def compound(sent: str) -> float:
        hit = cache.get(sent)
        if hit is None:
            hit = analyser.polarity_scores(sent)["compound"]
            cache[sent] = hit
        return hit

    rows: list[tuple] = []
    n_sentences = 0
    n_matched_sentences = 0
    scoreable = reviews.loc[~reviews["is_short"]]

    for rid, lid, text in zip(scoreable["id"], scoreable["listing_id"],
                              scoreable["comments"], strict=True):
        sents = T.sentences(text)
        n_sentences += len(sents)
        found: dict[str, list[float]] = {}
        for sent in sents:
            hits = [a for a in ASPECTS if patterns[a].search(sent)]
            if not hits:
                continue
            n_matched_sentences += 1
            c = compound(sent)
            for a in hits:
                found.setdefault(a, []).append(c)
        for a, vals in found.items():
            rows.append((rid, lid, a, float(np.mean(vals)), len(vals)))

    out = pd.DataFrame(rows, columns=["review_id", "listing_id", "aspect",
                                      "sentiment", "n_sentences"])
    out.attrs["n_sentences"] = n_sentences
    out.attrs["n_matched_sentences"] = n_matched_sentences
    out.attrs["n_scoreable_reviews"] = int(len(scoreable))
    out.attrs["n_cached_sentences"] = len(cache)
    return out


def to_listing(mentions: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Listing x aspect -> mention rate, mean sentiment, negative-mention rate.

    The mention-rate denominator is *scoreable reviews for that listing*, so
    a listing with many one-line reviews is not credited with a mention rate
    it did not earn.
    """
    denom = (reviews.loc[~reviews["is_short"]]
             .groupby("listing_id").size().rename("n_reviews_scored"))

    mentions = mentions.assign(
        is_neg=mentions["sentiment"] <= C.VADER_NEG_THRESHOLD,
        is_pos=mentions["sentiment"] >= C.VADER_POS_THRESHOLD,
    )
    g = mentions.groupby(["listing_id", "aspect"]).agg(
        mentions=("sentiment", "size"),
        sentiment=("sentiment", "mean"),
        neg_rate=("is_neg", "mean"),
        pos_rate=("is_pos", "mean"),
    ).reset_index()

    g = g.join(denom, on="listing_id")
    g["mention_rate"] = g["mentions"] / g["n_reviews_scored"]

    wide = g.pivot(index="listing_id", columns="aspect")
    wide.columns = [f"{stat}__{asp}" for stat, asp in wide.columns]
    wide = wide.join(denom)

    # A listing that never mentions an aspect has a mention rate of zero (a
    # fact) but no sentiment (not measured). Filling the second with zero
    # would invent a neutral opinion. It stays NaN, and each downstream model
    # decides what to do with it.
    for a in ASPECTS:
        wide[f"mentions__{a}"] = wide[f"mentions__{a}"].fillna(0)
        wide[f"mention_rate__{a}"] = wide[f"mention_rate__{a}"].fillna(0.0)
        thin = wide[f"mentions__{a}"] < C.MIN_ASPECT_MENTIONS
        for stat in ("sentiment", "neg_rate", "pos_rate"):
            wide.loc[thin, f"{stat}__{a}"] = np.nan
    return wide.reset_index()


# lexicon audit - what is the frozen lexicon missing?
def mine_vocabulary(texts: list[str]) -> tuple[Counter, str]:
    """
    Re-derive the corpus vocabulary of *things guests talk about*.

    Preferred path: spaCy dependency parse over noun-chunk heads, which is
    what seeded the committed lexicon. Without spaCy or its English model it
    falls back to scikit-learn TF-IDF over noun-ish unigrams, which cannot
    tell a noun from a verb. The method used is recorded in
    aspect_metrics.json.
    """
    try:
        import spacy
        # The lemmatizer stays on. It turns "buses"/"bus" and
        # "restaurants"/"restaurant" into one entry. Naive suffix-stripping
        # produced "bu" and "acce".
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:                                     # pragma: no cover
        return _mine_tfidf(texts), "tfidf-fallback"

    heads: Counter = Counter()
    for doc in nlp.pipe(texts, batch_size=64):
        for chunk in doc.noun_chunks:
            root = chunk.root
            if root.pos_ != "NOUN":
                continue
            heads[(root.lemma_ or root.text).lower()] += 1
    return heads, "spacy-en_core_web_sm-noun-chunks"


def _mine_tfidf(texts: list[str]) -> Counter:             # pragma: no cover
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=4000, stop_words="english",
                         token_pattern=r"(?u)\b[a-z][a-z]{2,}\b", lowercase=True)
    X = cv.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    return Counter({t: int(c)
                    for t, c in zip(cv.get_feature_names_out(), counts, strict=True)})


def audit_lexicon(heads: Counter) -> pd.DataFrame:
    """Which mined terms the frozen lexicon covers, and which not."""
    patterns = compile_patterns()
    rows = []
    for term, n in heads.most_common(C.LEXICON_AUDIT_TOP_N):
        if n < C.LEXICON_AUDIT_MIN_COUNT:
            break
        matched = [a for a in ASPECTS if patterns[a].search(term)]
        rows.append({"term": term, "mentions_in_sample": int(n),
                     "generic": term in GENERIC_HEADS,
                     "covered": bool(matched),
                     "aspects": "|".join(matched) if matched else ""})
    return pd.DataFrame(rows)


def validate(listing_aspects: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """
    Convergent validity: derived aspect sentiment vs Airbnb's own subscore.

    Restricted to listings with enough reviews for either measurement to be
    stable. Pearson and Spearman are both reported: the subscores are ordinal
    and heaped at 10, where Pearson understates a monotone relationship.
    """
    df = listing_aspects.merge(listings, left_on="listing_id", right_on="id",
                               how="inner")
    df = df[df["n_reviews_scored"] >= C.MIN_REVIEWS_FOR_MODEL]

    rows = []
    for a in ASPECTS:
        sub = SUBSCORE[a]
        base = {"aspect": a, "label": LABELS[a],
                "subscore": sub or "", "n": 0,
                "pearson_r": np.nan, "spearman_rho": np.nan, "p_value": np.nan,
                "neg_rate_pearson_r": np.nan}
        if sub is not None:
            d = df[[f"sentiment__{a}", f"neg_rate__{a}", sub]].dropna()
            if len(d) > 30:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, p = stats.pearsonr(d[f"sentiment__{a}"], d[sub])
                    rho, _ = stats.spearmanr(d[f"sentiment__{a}"], d[sub])
                    rn, _ = stats.pearsonr(d[f"neg_rate__{a}"], d[sub])
                base |= {"n": int(len(d)), "pearson_r": float(r),
                         "spearman_rho": float(rho), "p_value": float(p),
                         "neg_rate_pearson_r": float(rn)}
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    reviews = pd.read_parquet(C.REVIEWS_PARQUET)
    listings = pd.read_parquet(C.LISTINGS_PARQUET)

    print(f"Scoring {len(reviews):,} reviews across {len(ASPECTS)} aspects ...")
    mentions = score_reviews(reviews)
    mentions.to_parquet(C.ASPECTS_REVIEW_PARQUET, index=False)
    print(f"  {len(mentions):,} (review, aspect) mentions from "
          f"{mentions.attrs['n_matched_sentences']:,} of "
          f"{mentions.attrs['n_sentences']:,} sentences")

    la = to_listing(mentions, reviews)
    la.to_parquet(C.ASPECTS_LISTING_PARQUET, index=False)

    val = validate(la, listings)
    val.to_csv(C.TAB / "aspect_validation.csv", index=False)
    print(val.to_string(index=False))

    print("Re-mining corpus vocabulary to audit the frozen lexicon ...")
    sample = (reviews.loc[~reviews["is_short"], "comments"]
              .sample(min(C.LEXICON_AUDIT_SAMPLE, len(reviews)),
                      random_state=C.RANDOM_STATE).tolist())
    heads, mine_method = mine_vocabulary(sample)
    audit = audit_lexicon(heads)
    audit.to_csv(C.TAB / "lexicon_discovery.csv", index=False)
    # Coverage is reported on the terms an aspect lexicon could own. "place",
    # "stay" and "apartment" are the three commonest nouns and none names an
    # attribute. Scoring against them would understate recall by construction.
    specific = audit[~audit["generic"]]
    covered_share = float(specific["covered"].mean())
    covered_mass = float(specific.loc[specific["covered"], "mentions_in_sample"].sum()
                         / specific["mentions_in_sample"].sum())
    print(f"  method={mine_method}; lexicon covers {covered_share:.0%} of the "
          f"{len(specific)} non-generic terms in the top {len(audit)} "
          f"({covered_mass:.0%} by frequency)")

    # fig 4
    rate = pd.Series({a: float((mentions["aspect"] == a).sum()) for a in ASPECTS})
    rate = rate / mentions.attrs["n_scoreable_reviews"]
    neg = pd.Series({
        a: float((mentions.loc[mentions["aspect"] == a, "sentiment"]
                  <= C.VADER_NEG_THRESHOLD).mean()) for a in ASPECTS})
    order = rate.sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.barh(range(len(order)), rate[order].to_numpy(), color=viz.BLUE, height=0.62)
    viz.bar_ends(ax, horizontal=True)
    ax.set_yticks(range(len(order)), [LABELS[a] for a in order])
    for i, a in enumerate(order):
        ax.text(rate[a] + 0.012, i, f"{rate[a]:.0%}", va="center",
                fontsize=9, color=viz.INK_2)
    ax.set_xlim(0, float(rate.max()) * 1.18)
    ax.grid(axis="y", visible=False)
    viz.pct(ax, "x")
    ax.set_xlabel("Share of scoreable reviews mentioning the aspect")
    head_l = f"{LABELS[order[-1]]} is mentioned in {rate[order[-1]]:.0%} of reviews"
    viz.title(ax, head_l, f"{len(mentions):,} aspect mentions across "
                          f"{mentions.attrs['n_scoreable_reviews']:,} reviews")
    ax.set_title(head_l, loc="left", pad=20)

    ax = axes[1]
    ax.barh(range(len(order)), neg[order].to_numpy(), color=viz.ORANGE, height=0.62)
    viz.bar_ends(ax, horizontal=True)
    ax.set_yticks(range(len(order)), [LABELS[a] for a in order])
    for i, a in enumerate(order):
        ax.text(neg[a] + float(neg.max()) * 0.03, i, f"{neg[a]:.1%}", va="center",
                fontsize=9, color=viz.INK_2)
    ax.set_xlim(0, float(neg.max()) * 1.28)
    ax.grid(axis="y", visible=False)
    viz.pct(ax, "x", decimals=0)
    ax.set_xlabel("Share of mentions scored negative by VADER")
    worst = neg.idxmax()
    head = f"The sorest aspect is negative in {neg[worst]:.0%} of its mentions"
    viz.title(ax, head,
              f"Share of mentions with VADER compound <= {C.VADER_NEG_THRESHOLD}")
    ax.set_title(head, loc="left", pad=20)
    viz.source(fig, textwrap.fill(
        "Negative mentions are rare in absolute terms. That is the finding, "
        "not a failure of the extractor: guests who had a bad stay mostly "
        "do not review at all. Note this is per mention, not per review - "
        "value for money looks worst here only because nobody raises price "
        "unless something is wrong with it, which is why priority_matrix.py "
        "uses complaints per review instead.", 108))
    fig.subplots_adjust(wspace=0.42)
    viz.save(fig, C.FIG / "04_aspect_prevalence.png")

    # fig 5
    v = val[val["subscore"] != ""].sort_values("spearman_rho")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    y = np.arange(len(v))
    ax.barh(y - 0.185, v["pearson_r"].to_numpy(), height=0.33, color=viz.BLUE)
    ax.barh(y + 0.185, v["spearman_rho"].to_numpy(), height=0.33, color=viz.AQUA)
    viz.bar_ends(ax, horizontal=True)
    ax.set_yticks(y, v["label"].tolist())
    for i, (r, rho) in enumerate(zip(v["pearson_r"], v["spearman_rho"], strict=True)):
        ax.text(r + 0.008, i - 0.185, f"{r:.2f}", va="center", fontsize=8.4,
                color=viz.INK_2)
        ax.text(rho + 0.008, i + 0.185, f"{rho:.2f}", va="center", fontsize=8.4,
                color=viz.INK_2)
    ax.set_xlim(0, float(max(v["pearson_r"].max(), v["spearman_rho"].max())) * 1.24)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Correlation with Airbnb's own subscore")
    # bar_ends replaces the original patches, so the legend needs explicit
    # handles or it renders as labels with no swatches.
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=viz.BLUE),
               plt.Rectangle((0, 0), 1, 1, color=viz.AQUA)],
              ["Pearson r", "Spearman rho"], loc="lower right")
    viz.title(ax, "Agreement with the subscores is real but weak",
              f"Listings with {C.MIN_REVIEWS_FOR_MODEL}+ scoreable reviews "
              f"and {C.MIN_ASPECT_MENTIONS}+ mentions of the aspect")

    # A scatter for the strongest pair. It shows what an r of this size looks
    # like.
    best = v.iloc[-1]
    d = (la.merge(listings, left_on="listing_id", right_on="id")
           .query("n_reviews_scored >= @C.MIN_REVIEWS_FOR_MODEL")
           [[f"sentiment__{best['aspect']}", best["subscore"]]].dropna())
    ax = axes[1]
    jitter = np.random.default_rng(C.RANDOM_STATE).normal(0, 0.055, len(d))
    ax.scatter(d[f"sentiment__{best['aspect']}"], d[best["subscore"]] + jitter,
               s=8, color=viz.BLUE, alpha=0.28, linewidths=0)
    fit = np.polyfit(d[f"sentiment__{best['aspect']}"], d[best["subscore"]], 1)
    xs = np.linspace(d[f"sentiment__{best['aspect']}"].min(),
                     d[f"sentiment__{best['aspect']}"].max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=viz.ORANGE, linewidth=2.0)
    ax.set_xlabel(f"Derived {best['label'].lower()} sentiment (VADER, -1 to +1)")
    ax.set_ylabel(f"Airbnb {best['label'].lower()} subscore (jittered)")
    ax.legend([plt.Line2D([], [], color=viz.ORANGE, lw=2)], ["Linear fit"],
              loc="lower right")
    viz.title(ax,
              f"Even the closest pair leaves {1 - best['pearson_r']**2:.0%} unexplained",
              f"{best['label']}: r = {best['pearson_r']:.2f}, "
              f"rho = {best['spearman_rho']:.2f}, n = {int(best['n']):,}")
    viz.source(fig, textwrap.fill(
        "Convergent validity, not accuracy: the subscore is itself a noisy, "
        "compressed measurement, so a perfect extractor could not correlate "
        "at 1.0 with it.", 108))
    fig.subplots_adjust(wspace=0.30)
    viz.save(fig, C.FIG / "05_aspect_validation.png")

    # metrics
    metrics = {
        "aspects": list(ASPECTS),
        "reviews_scoreable": int(mentions.attrs["n_scoreable_reviews"]),
        "sentences_total": int(mentions.attrs["n_sentences"]),
        "sentences_with_aspect": int(mentions.attrs["n_matched_sentences"]),
        "sentence_match_rate": float(mentions.attrs["n_matched_sentences"]
                                     / mentions.attrs["n_sentences"]),
        "unique_sentences_scored": int(mentions.attrs["n_cached_sentences"]),
        "mentions_total": int(len(mentions)),
        "reviews_with_no_aspect": int(mentions.attrs["n_scoreable_reviews"]
                                      - mentions["review_id"].nunique()),
        "reviews_with_no_aspect_share": float(
            1 - mentions["review_id"].nunique() / mentions.attrs["n_scoreable_reviews"]),
        "mention_rate": {a: float(rate[a]) for a in ASPECTS},
        "negative_mention_rate": {a: float(neg[a]) for a in ASPECTS},
        # Complaints per review, not per mention. Per-mention rates are
        # confounded by how often a topic is raised at all. Value for money
        # comes up in 7% of reviews and is usually raised *because* something
        # is wrong: worst per mention, one of the rarest complaints made.
        "complaint_incidence": {a: float(rate[a] * neg[a]) for a in ASPECTS},
        "mean_sentiment": {a: float(mentions.loc[mentions["aspect"] == a,
                                                 "sentiment"].mean())
                           for a in ASPECTS},
        "most_mentioned_aspect": str(rate.idxmax()),
        "most_negative_aspect": str(neg.idxmax()),
        "validation": val.set_index("aspect").to_dict(orient="index"),
        "validation_mean_pearson": float(val["pearson_r"].mean()),
        "validation_mean_spearman": float(val["spearman_rho"].mean()),
        "validation_best_aspect_by_spearman": str(best["aspect"]),
        "validation_best_pearson": float(best["pearson_r"]),
        "validation_best_spearman": float(best["spearman_rho"]),
        "validation_worst_aspect_by_spearman": str(v.iloc[0]["aspect"]),
        "validation_worst_pearson": float(v.iloc[0]["pearson_r"]),
        "validation_worst_spearman": float(v.iloc[0]["spearman_rho"]),
        "validation_max_pearson": float(val["pearson_r"].max()),
        "validation_neg_rate_pearson": {
            r["aspect"]: (None if pd.isna(r["neg_rate_pearson_r"])
                          else float(r["neg_rate_pearson_r"]))
            for _, r in val.iterrows()},
        "lexicon_mining_method": mine_method,
        "lexicon_terms": {a: len(TERMS[a]) for a in ASPECTS},
        "lexicon_terms_total": int(sum(len(TERMS[a]) for a in ASPECTS)),
        "lexicon_audit_terms_checked": int(len(audit)),
        "lexicon_audit_terms_specific": int(len(specific)),
        "lexicon_audit_covered_share": covered_share,
        "lexicon_audit_covered_by_frequency": covered_mass,
        "lexicon_audit_top_uncovered": specific.loc[~specific["covered"], "term"]
                                               .head(15).tolist(),
    }
    (C.OUT / "aspect_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v_ for k, v_ in metrics.items()
                      if k != "validation"}, indent=2))


if __name__ == "__main__":
    main()
