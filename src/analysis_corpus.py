"""
Step 2. Describe the corpus and name its central constraint.

Airbnb ratings are compressed at the top. The mean listing scores in the
mid-nineties out of 100 and most sit above 90. Anything explaining "what
drives ratings" is explaining a variable with almost no room to move.

Outputs
  figures/01_review_volume.png
  figures/02_rating_compression.png
  figures/03_corpus_shape.png
  tables/reviews_by_month.csv
  corpus_metrics.json
"""
from __future__ import annotations

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import viz
from lexicon import SUBSCORE

viz.use_style()

SUBSCORE_ORDER = [
    ("review_scores_checkin", "Check-in"),
    ("review_scores_communication", "Communication"),
    ("review_scores_location", "Location"),
    ("review_scores_accuracy", "Accuracy"),
    ("review_scores_cleanliness", "Cleanliness"),
    ("review_scores_value", "Value"),
]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(C.REVIEWS_PARQUET),
            pd.read_parquet(C.LISTINGS_PARQUET))


def gini(x: np.ndarray) -> float:
    """Concentration of reviews across listings. 0 = flat, 1 = one listing has all."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    reviews, listings = load()
    rated = listings["review_scores_rating"].dropna()

    # fig 1
    monthly = (reviews.assign(month=reviews["date"].dt.to_period("M").dt.to_timestamp())
                      .groupby("month").size().rename("reviews"))
    # The last month is a stub: the scrape stopped on 4 January 2016. Shown,
    # and excluded from any growth statement.
    full = monthly.iloc[:-1]
    last_18 = full.tail(18).sum() / full.sum()
    monthly.to_frame().to_csv(C.TAB / "reviews_by_month.csv")

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.fill_between(monthly.index, monthly.to_numpy(), color=viz.BLUE, alpha=0.16)
    ax.plot(monthly.index, monthly.to_numpy(), color=viz.BLUE)
    # The final bar is four days of January, not a month. Shaded rather than
    # flagged with an arrow: at this aspect ratio a leader line crosses the
    # 2015 peak.
    ax.axvspan(monthly.index[-1] - pd.Timedelta(days=15),
               monthly.index[-1] + pd.Timedelta(days=20),
               color=viz.ORANGE, alpha=0.22, linewidth=0)
    ax.text(0.015, 0.93, "Shaded: scrape stopped 4 Jan 2016,\nso the last month is 4 days long",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.6,
            color=viz.INK_2)
    ax.margins(x=0.01)
    ax.set_ylim(0, monthly.max() * 1.22)
    ax.set_ylabel("Reviews posted")
    viz.title(ax,
              f"{last_18:.0%} of the corpus was written in the final 18 months",
              "Monthly review volume, Seattle listings, June 2009 - January 2016")
    viz.source(fig, textwrap.fill(
        f"Inside Airbnb Seattle scrape, 4 January 2016. {len(reviews):,} "
        "reviews after cleaning.", 108))
    viz.save(fig, C.FIG / "01_review_volume.png")

    # fig 2
    share_90 = float((rated >= 90).mean())
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    ax = axes[0]
    bins = np.arange(20, 102, 2)
    counts, edges = np.histogram(rated, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    cols = [viz.BLUE if c >= 90 else viz.ORANGE for c in centres]
    ax.bar(centres, counts, width=1.7, color=cols)
    ax.axvline(float(rated.mean()), color=viz.INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"Mean {rated.mean():.1f}", (float(rated.mean()), counts.max() * 0.93),
                xytext=(-8, 0), textcoords="offset points", ha="right",
                fontsize=9, color=viz.INK, fontweight="bold")
    ax.set_xlabel("Overall rating (0-100)")
    ax.set_ylabel("Listings")
    ax.set_xlim(18, 102)
    handles = [plt.Rectangle((0, 0), 1, 1, color=viz.BLUE),
               plt.Rectangle((0, 0), 1, 1, color=viz.ORANGE)]
    ax.legend(handles, [f"90 and above ({share_90:.0%})",
                        f"Below 90 ({1 - share_90:.0%})"], loc="upper left")
    head_hist = f"{share_90:.0%} of listings score 90 or better"
    viz.title(ax, head_hist, f"Overall rating, {len(rated):,} rated Seattle listings")
    ax.set_title(head_hist, loc="left", pad=18)

    # A boxplot of these subscores is unreadable: the interquartile range of
    # check-in and communication is a single point at 10.0. The right-hand
    # panel bands them instead.
    ax = axes[1]
    band_edges = [(10.0, 10.01, "Perfect 10"), (9.0, 10.0, "9 to 9.9"),
                  (8.0, 9.0, "8 to 8.9"), (0.0, 8.0, "Below 8")]
    band_cols = [viz.SEQ(0.88), viz.SEQ(0.58), viz.SEQ(0.3), viz.ORANGE]
    names = [n for _, n in SUBSCORE_ORDER]
    ypos = np.arange(len(names))
    left = np.zeros(len(names))
    shares = {}
    for (lo, hi, lbl), col in zip(band_edges, band_cols, strict=True):
        w = np.array([float(((listings[c].dropna() >= lo)
                             & (listings[c].dropna() < hi)).mean())
                      for c, _ in SUBSCORE_ORDER])
        shares[lbl] = w
        ax.barh(ypos, w, left=left, height=0.62, color=col, label=lbl)
        for y, (x0, ww) in enumerate(zip(left, w, strict=True)):
            if ww >= 0.075:
                ax.text(x0 + ww / 2, y, f"{ww:.0%}", ha="center", va="center",
                        fontsize=8.6, color=viz.SURFACE if col is not viz.ORANGE
                        else viz.INK, fontweight="bold")
        left = left + w
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.margins(y=0.06)
    ax.grid(axis="y", visible=False)
    viz.pct(ax, "x")
    ax.set_xlabel("Share of rated listings")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=4)
    perfect_min = min(shares["Perfect 10"])
    head = f"At least {perfect_min:.0%} score a perfect 10 on every subscore"
    viz.title(ax, head, "Distribution of each Airbnb subscore across rated listings")
    ax.set_title(head, loc="left", pad=18)
    viz.source(fig, textwrap.fill(
        "Compression is the central analytical constraint in this project: a "
        "model of overall rating is explaining a variable that barely moves, "
        "and every coefficient downstream should be read against a six-point "
        "interquartile range.", 108))
    fig.subplots_adjust(wspace=0.30)
    viz.save(fig, C.FIG / "02_rating_compression.png")

    # fig 3
    per_listing = reviews.groupby("listing_id").size()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))

    ax = axes[0]
    ax.hist(reviews["n_words"].clip(upper=400), bins=60, color=viz.BLUE)
    med = float(reviews["n_words"].median())
    ax.axvline(med, color=viz.ORANGE, linewidth=1.6)
    ax.annotate(f"Median {med:.0f} words", (med, ax.get_ylim()[1] * 0.88),
                xytext=(9, 0), textcoords="offset points", fontsize=9,
                color=viz.ORANGE, fontweight="bold")
    ax.set_xlabel("Words per review (clipped at 400)")
    ax.set_ylabel("Reviews")
    viz.title(ax, f"The typical review is {med:.0f} words",
              "A paragraph, not an essay: room for two or three aspect mentions")

    ax = axes[1]
    srt = np.sort(per_listing.to_numpy())[::-1]
    cum = np.cumsum(srt) / srt.sum()
    frac = np.arange(1, srt.size + 1) / srt.size
    top20 = float(np.interp(0.20, frac, cum))
    ax.plot(frac, cum, color=viz.BLUE)
    ax.fill_between(frac, cum, color=viz.BLUE, alpha=0.14)
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot([0.20, 0.20], [0, top20], color=viz.ORANGE, linewidth=1.4)
    ax.plot([0.20], [top20], marker="o", color=viz.ORANGE)
    ax.annotate(f"Busiest 20% of listings\nhold {top20:.0%} of reviews",
                (0.20, top20), xytext=(16, -30), textcoords="offset points",
                fontsize=9, color=viz.ORANGE, fontweight="bold")
    viz.pct(ax, "x")
    viz.pct(ax, "y")
    ax.set_xlabel("Listings, busiest first")
    ax.set_ylabel("Cumulative share of reviews")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend([plt.Line2D([], [], color=viz.BLUE, lw=2),
               plt.Line2D([], [], color=viz.INK_MUTED, lw=1, linestyle=(0, (4, 3)))],
              ["Actual", "Perfectly even"], loc="lower right")
    viz.title(ax, f"Review volume is concentrated: Gini {gini(per_listing.to_numpy()):.2f}",
              f"Median listing has {per_listing.median():.0f} reviews; "
              f"the busiest has {per_listing.max():,}")
    viz.source(fig, textwrap.fill(
        "Concentration matters because a listing-level mean built from 3 "
        "reviews is not the same measurement as one built from 300.", 108))
    fig.subplots_adjust(wspace=0.28)
    viz.save(fig, C.FIG / "03_corpus_shape.png")

    # metrics
    metrics = {
        "reviews": int(len(reviews)),
        "listings": int(len(listings)),
        "listings_rated": int(len(rated)),
        "reviews_last_18m_share": float(last_18),
        "reviews_2015_share": float(
            (reviews["date"].dt.year == 2015).sum() / len(reviews)),
        "median_words": float(reviews["n_words"].median()),
        "mean_words": float(reviews["n_words"].mean()),
        "p90_words": float(reviews["n_words"].quantile(0.90)),
        "median_reviews_per_listing": float(per_listing.median()),
        "max_reviews_per_listing": int(per_listing.max()),
        "reviews_gini": float(gini(per_listing.to_numpy())),
        "top20pct_listings_review_share": float(top20),
        "mean_rating": float(rated.mean()),
        "median_rating": float(rated.median()),
        "sd_rating": float(rated.std()),
        "share_rating_ge_90": float(share_90),
        "share_rating_ge_95": float((rated >= 95).mean()),
        "share_rating_100": float((rated == 100).mean()),
        "rating_iqr": [float(rated.quantile(0.25)), float(rated.quantile(0.75))],
        "subscore_medians": {n: float(listings[c].median())
                             for c, n in SUBSCORE_ORDER},
        "subscore_share_ge_9": {n: float((listings[c] >= 9).mean())
                                for c, n in SUBSCORE_ORDER},
        "subscore_share_perfect_10": {n: float((listings[c] == 10).mean())
                                      for c, n in SUBSCORE_ORDER},
        "min_subscore_share_perfect_10": float(
            min((listings[c] == 10).mean() for c, _ in SUBSCORE_ORDER)),
        "aspects_with_official_subscore": int(
            sum(v is not None for v in SUBSCORE.values())),
    }
    (C.OUT / "corpus_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
