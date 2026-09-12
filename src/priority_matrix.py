"""
Step 6. Which of the eight attributes deserve money.

An importance-performance matrix. Importance is the estimated association
between an attribute's sentiment and the overall rating, from `drivers.py`.
Performance is how often guests complain about it. Crossed, the portfolio
splits into four groups, one of which is "spend".

Two things sit in the chart rather than underneath it:

- An attribute whose importance interval crosses zero is drawn hollow. The
  quadrant it lands in is not a finding, and the table says so.
- The second panel converts each attribute into the rating points a
  bottom-quartile listing would gain by reaching the portfolio median on it.
  The quadrant says where to look, the number how much is there.

Outputs
  figures/11_priority_matrix.png
  tables/priority_actions.csv
  priority_metrics.json
"""
from __future__ import annotations

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import viz
from lexicon import ASPECTS, LABELS

viz.use_style()

QUADRANTS = {
    ("high", "low"): ("Fix first",
                      "Matters, and currently goes wrong most often"),
    ("high", "high"): ("Maintain",
                       "Matters and is already working - protect it, do not fund it"),
    ("low", "low"): ("Low priority",
                     "Goes wrong, but moving it barely moves the rating"),
    ("low", "high"): ("Possible over-investment",
                      "Already excellent, and improving it further buys little"),
}
QUAD_COLOUR = {"Fix first": viz.RED, "Maintain": viz.GREEN,
               "Low priority": viz.INK_MUTED, "Possible over-investment": viz.YELLOW}


def place_labels(ax, xs, ys, texts, fontsize=9.4) -> None:
    """
    Put a label next to each point without letting labels collide.

    Candidate offsets are tried in order. The first that clears every label
    already placed - and every marker - wins. The text metrics are
    approximate: the boxes only have to keep things apart.
    """
    candidates = [(0, 15), (0, -21), (14, 5), (-14, 5),
                  (14, -13), (-14, -13), (0, 27), (0, -33)]
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    taken = [(px - 9, py - 9, px + 9, py + 9) for px, py in pts]   # markers

    def overlaps(box) -> bool:
        return any(not (box[2] < b[0] or box[0] > b[2]
                        or box[3] < b[1] or box[1] > b[3]) for b in taken)

    order = np.argsort(-np.asarray(ys))            # highest importance first
    for i in order:
        px, py = pts[i]
        w, h = len(texts[i]) * fontsize * 0.58, fontsize * 1.5
        for dx, dy in candidates:
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            x0 = px + dx - (w / 2 if ha == "center" else (0 if ha == "left" else w))
            box = (x0, py + dy - h / 2, x0 + w, py + dy + h / 2)
            if not overlaps(box):
                break
        taken.append(box)
        ax.annotate(texts[i], (xs[i], ys[i]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=fontsize, color=viz.INK, fontweight="bold")


def load() -> tuple[pd.DataFrame, dict, dict]:
    frame = pd.read_parquet(C.MODEL_FRAME_PARQUET)
    frame = frame[frame["review_scores_rating"].notna()]
    drivers = json.loads((C.OUT / "driver_metrics.json").read_text())
    aspects = json.loads((C.OUT / "aspect_metrics.json").read_text())
    return frame, drivers, aspects


def build(frame: pd.DataFrame, drivers: dict, aspects: dict) -> pd.DataFrame:
    rows = []
    for a in ASPECTS:
        coef = drivers["rating_coefficients"][f"sent_{a}"]
        s = frame[f"sentiment__{a}"].dropna()
        sd = float(s.std())
        # Gap to close: how far a bottom-quartile listing sits below the
        # portfolio median on this attribute, in standard deviations. Median
        # rather than mean because the sentiment distributions are skewed.
        gap_sd = float((s.median() - s.quantile(0.25)) / sd) if sd > 0 else 0.0
        rows.append({
            "aspect": a,
            "label": LABELS[a],
            "importance": coef["coef"],
            "importance_lo": coef["ci_lo"],
            "importance_hi": coef["ci_hi"],
            "evidence": "clear" if coef["significant"] else "inconclusive",
            "complaint_rate": aspects["complaint_incidence"][a],
            "complaint_per_mention": aspects["negative_mention_rate"][a],
            "mention_rate": aspects["mention_rate"][a],
            "mean_sentiment": aspects["mean_sentiment"][a],
            "listings_measured": int(frame[f"sentiment__{a}"].notna().sum()),
            "gap_p25_to_median_sd": gap_sd,
            "rating_points_available": gap_sd * max(coef["coef"], 0.0),
        })
    df = pd.DataFrame(rows)

    # Performance is complaints per *review*, not per mention. See the note
    # in aspects.py. Per-mention rates rank value for money worst because
    # nobody mentions price unless something is wrong.
    # Split on the median of each axis across the eight attributes. A median
    # split is arbitrary, as any split is, and cannot be tuned to put a
    # favoured attribute in a favoured box.
    imp_cut = float(df["importance"].median())
    perf_cut = float(df["complaint_rate"].median())
    df["imp_side"] = np.where(df["importance"] >= imp_cut, "high", "low")
    # Low complaint rate = high performance.
    df["perf_side"] = np.where(df["complaint_rate"] <= perf_cut, "high", "low")
    df["quadrant"] = [QUADRANTS[(i, p)][0]
                      for i, p in zip(df["imp_side"], df["perf_side"], strict=True)]
    df["action"] = [QUADRANTS[(i, p)][1]
                    for i, p in zip(df["imp_side"], df["perf_side"], strict=True)]
    df.loc[df["evidence"] == "inconclusive", "action"] = (
        "Evidence inconclusive: the importance interval crosses zero")
    df.attrs["imp_cut"] = imp_cut
    df.attrs["perf_cut"] = perf_cut
    return df


def main() -> None:
    frame, drivers, aspects = load()
    df = build(frame, drivers, aspects)
    df.to_csv(C.TAB / "priority_actions.csv", index=False)
    print(df[["label", "importance", "complaint_rate", "quadrant",
              "rating_points_available", "evidence"]].to_string(index=False))

    imp_cut, perf_cut = df.attrs["imp_cut"], df.attrs["perf_cut"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.8),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    # matrix
    ax = axes[0]
    x = df["complaint_rate"].to_numpy()
    y = df["importance"].to_numpy()
    pad_x = (x.max() - x.min()) * 0.30
    pad_y = (y.max() - y.min()) * 0.34
    xlim = (x.max() + pad_x, max(x.min() - pad_x, 0.0))     # inverted: right = better
    ylim = (y.min() - pad_y, y.max() + pad_y)

    # The shaded bands start where the importance cut is, so the fraction is
    # computed from the data limits rather than guessed.
    cut_frac = (imp_cut - ylim[0]) / (ylim[1] - ylim[0])
    ax.axvspan(perf_cut, xlim[0], ymin=cut_frac, ymax=1.0,
               color=viz.RED, alpha=0.05, lw=0)
    ax.axvspan(xlim[1], perf_cut, ymin=cut_frac, ymax=1.0,
               color=viz.GREEN, alpha=0.05, lw=0)
    ax.axvspan(xlim[1], perf_cut, ymin=0.0, ymax=cut_frac,
               color=viz.YELLOW, alpha=0.05, lw=0)
    ax.axvline(perf_cut, color=viz.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.axhline(imp_cut, color=viz.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))

    for _, r in df.iterrows():
        clear = r["evidence"] == "clear"
        col = QUAD_COLOUR[r["quadrant"]] if clear else viz.INK_MUTED
        ax.vlines(r["complaint_rate"], r["importance_lo"], r["importance_hi"],
                  color=col, linewidth=1.6, alpha=0.35, zorder=2)
        ax.scatter(r["complaint_rate"], r["importance"],
                   s=90 + 900 * r["mention_rate"],
                   facecolor=col if clear else "none",
                   edgecolor=col if clear else viz.INK_MUTED,
                   linewidth=1.6, alpha=0.9 if clear else 1.0, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    place_labels(ax, df["complaint_rate"].to_numpy(), df["importance"].to_numpy(),
                 [lab + ("" if ev == "clear" else " *")
                  for lab, ev in zip(df["label"], df["evidence"], strict=True)])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v * 100:.1f}"))
    ax.set_xlabel("Complaints per 100 reviews  (worse <-------> better)")
    ax.set_ylabel("Rating points per 1 SD of sentiment")
    for (qi, qp), (name, _) in QUADRANTS.items():
        ax.text(0.985 if qp == "high" else 0.015,
                0.965 if qi == "high" else 0.035,
                name.upper(), transform=ax.transAxes,
                ha="right" if qp == "high" else "left",
                va="top" if qi == "high" else "bottom",
                fontsize=9, color=QUAD_COLOUR[name], fontweight="bold")
    fixes = df.query("quadrant == 'Fix first' and evidence == 'clear'")["label"].tolist()
    head = (f"Fix {' and '.join(fixes).lower()} first"
            if fixes else "Nothing lands in the fix-first quadrant with clear evidence")
    viz.title(ax, head,
              "Bubble size is how often the attribute is mentioned. "
              "* = importance interval crosses zero.")
    ax.set_title(head, loc="left", pad=22)

    # size of the prize
    ax = axes[1]
    p = df.sort_values("rating_points_available")
    cols = [QUAD_COLOUR[q] if e == "clear" else viz.INK_MUTED
            for q, e in zip(p["quadrant"], p["evidence"], strict=True)]
    ax.barh(range(len(p)), p["rating_points_available"], color=cols, height=0.62)
    viz.bar_ends(ax, horizontal=True)
    ax.set_yticks(range(len(p)), p["label"])
    for i, v in enumerate(p["rating_points_available"]):
        ax.text(v + p["rating_points_available"].max() * 0.03, i, f"{v:+.2f}",
                va="center", fontsize=9, color=viz.INK_2)
    ax.set_xlim(0, float(p["rating_points_available"].max()) * 1.24)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rating points, out of 100")
    best = p.iloc[-1]
    head2 = (f"The whole prize is {df['rating_points_available'].sum():.1f} rating "
             "points")
    viz.title(ax, head2,
              "Points a bottom-quartile listing would gain by reaching the "
              "portfolio median on one attribute")
    ax.set_title(head2, loc="left", pad=22)
    viz.source(fig, textwrap.fill(
        "Importance from the Ridge model in drivers.py: association, not "
        "effect, and conditional on room type, size, area, price, review "
        "count and host tenure. The prize is arithmetic on those "
        "coefficients, so it inherits every one of their assumptions.", 108))
    fig.subplots_adjust(wspace=0.34)
    viz.save(fig, C.FIG / "11_priority_matrix.png")

    metrics = {
        "importance_cut": imp_cut,
        "performance_cut": perf_cut,
        "quadrants": {r["aspect"]: {"label": r["label"], "quadrant": r["quadrant"],
                                    "action": r["action"], "evidence": r["evidence"],
                                    "importance": float(r["importance"]),
                                    "complaint_rate": float(r["complaint_rate"]),
                                    "complaint_per_mention":
                                        float(r["complaint_per_mention"]),
                                    "mention_rate": float(r["mention_rate"]),
                                    "rating_points_available":
                                        float(r["rating_points_available"])}
                      for _, r in df.iterrows()},
        "fix_first": df.query("quadrant == 'Fix first'")["label"].tolist(),
        "fix_first_clear_evidence": fixes,
        "maintain": df.query("quadrant == 'Maintain'")["label"].tolist(),
        "low_priority": df.query("quadrant == 'Low priority'")["label"].tolist(),
        "over_investment": df.query("quadrant == 'Possible over-investment'")
                             ["label"].tolist(),
        "total_rating_points_available": float(df["rating_points_available"].sum()),
        "largest_single_prize_aspect": str(best["label"]),
        "largest_single_prize_points": float(best["rating_points_available"]),
        "inconclusive_aspects": df.query("evidence == 'inconclusive'")["label"].tolist(),
    }
    (C.OUT / "priority_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
