"""
Step 7. Where in Seattle the portfolio is strong, and where it is not.

Reported at the `neighbourhood_group_cleansed` level - 17 areas - rather than
`neighbourhood_cleansed`, which has 87. A mean built on six listings has a
standard error roughly four times that of one built on a hundred, so at 87
areas a table of "best and worst neighbourhoods" is a table of which small
areas got lucky. Areas below MIN_LISTINGS_PER_AREA are excluded at either
level.

Cells are marked where the area differs from the city by more than two
standard errors, so the map shows differences that survive their own noise.

Outputs
  figures/12_neighbourhood_positioning.png
  tables/neighbourhood_profile.csv
  positioning_metrics.json
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

AREA = "neighbourhood_group_cleansed"
FINE = "neighbourhood_cleansed"


def stability_check(frame: pd.DataFrame) -> dict:
    """
    How much noisier the fine-grained level is, in numbers.

    Compares the median standard error of mean cleanliness sentiment at the
    two levels, and counts the units at each that clear the minimum listing
    count.
    """
    out = {}
    for name, col in (("neighbourhood_group", AREA), ("neighbourhood", FINE)):
        g = frame.groupby(col)["sentiment__cleanliness"]
        n = g.size()
        se = g.std() / np.sqrt(g.count())
        out[name] = {
            "units": int(frame[col].nunique()),
            "units_above_min": int((n >= C.MIN_LISTINGS_PER_AREA).sum()),
            "median_listings": float(n.median()),
            "min_listings": int(n.min()),
            "median_standard_error": float(se.median()),
        }
    out["se_ratio_fine_over_coarse"] = (
        out["neighbourhood"]["median_standard_error"]
        / out["neighbourhood_group"]["median_standard_error"])
    return out


def area_profile(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Area x aspect z-scores against the city, plus the standard-error map."""
    keep = frame.groupby(AREA)["id"].transform("size") >= C.MIN_LISTINGS_PER_AREA
    df = frame[keep]

    z = pd.DataFrame(index=sorted(df[AREA].unique()), dtype=float)
    t = pd.DataFrame(index=z.index, dtype=float)
    for a in ASPECTS:
        col = f"sentiment__{a}"
        city_mean = df[col].mean()
        city_sd = df[col].std()
        g = df.groupby(AREA)[col]
        # Effect size against the city (in listing-level SDs) and the
        # t-like statistic that says whether it is distinguishable from zero.
        z[LABELS[a]] = (g.mean() - city_mean) / city_sd
        t[LABELS[a]] = (g.mean() - city_mean) / (g.std() / np.sqrt(g.count()))

    prof = pd.DataFrame({
        "listings": df.groupby(AREA)["id"].size(),
        "median_price": df.groupby(AREA)["price"].median(),
        "mean_rating": df.groupby(AREA)["review_scores_rating"].mean(),
        "mean_price_premium": df.groupby(AREA)["price_premium"].mean(),
        "reviews": df.groupby(AREA)["n_reviews_scored"].sum(),
    })
    prof = prof.join(z.add_prefix("z_"))
    return z, t, prof


def main() -> None:
    frame = pd.read_parquet(C.MODEL_FRAME_PARQUET)
    frame = frame[frame["review_scores_rating"].notna()]

    stab = stability_check(frame)
    print(json.dumps(stab, indent=2))

    z, t, prof = area_profile(frame)
    prof.to_csv(C.TAB / "neighbourhood_profile.csv")

    # Order areas by their average standing across the eight attributes, so
    # the map reads top-to-bottom as "strongest guest experience first".
    order = z.mean(axis=1).sort_values(ascending=False).index
    z, t = z.loc[order], t.loc[order]
    cols = [LABELS[a] for a in ASPECTS]
    z, t = z[cols], t[cols]

    lim = float(np.nanmax(np.abs(z.to_numpy())))
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    # DIV runs blue -> red. Reversed so red reads as "below the city".
    im = ax.imshow(z.to_numpy(), cmap=viz.DIV.reversed(), aspect="auto",
                   vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(cols)), cols, rotation=32, ha="right")
    ax.set_yticks(range(len(z)),
                  [f"{a}  (n={int(prof.loc[a, 'listings'])})"
                   + ("  \u2190 catch-all bucket" if a in C.CATCH_ALL_AREAS else "")
                   for a in z.index])
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(z), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color=viz.SURFACE, linewidth=2.0)

    n_marked = 0
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            v = z.iat[r, c]
            strong = abs(t.iat[r, c]) > 2
            n_marked += int(strong)
            ax.text(c, r, f"{v:+.2f}" + ("*" if strong else ""),
                    ha="center", va="center", fontsize=8.2,
                    fontweight="bold" if strong else "normal",
                    color="#ffffff" if abs(v) > lim * 0.55 else viz.INK)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.026)
    cb.outline.set_visible(False)
    cb.set_label("Standard deviations from the city mean", color=viz.INK_2, fontsize=9)

    real = [a for a in z.index if a not in C.CATCH_ALL_AREAS]
    best_area = real[0]
    worst_area = real[-1]
    # The largest gap that is also statistically distinguishable.
    masked = z.where(t.abs() > 2)
    flat = masked.stack()
    gap_area, gap_aspect = flat.abs().idxmax()
    gap_val = float(flat.loc[(gap_area, gap_aspect)])

    n_below = int((t.loc[worst_area] < -2).sum())
    head = (f"{worst_area} sits significantly below the city on "
            f"{n_below} of {len(cols)} attributes")
    viz.title(ax, head,
              f"Areas with {C.MIN_LISTINGS_PER_AREA}+ modelled listings, ordered by "
              "average standing. * marks a gap over two standard errors wide.")
    ax.set_title(head, loc="left", pad=22)
    viz.source(fig, textwrap.fill(
        f"Reported at the {stab['neighbourhood_group']['units']}-area level, "
        f"not the {stab['neighbourhood']['units']}-neighbourhood level: the "
        "median neighbourhood standard error is "
        f"{stab['se_ratio_fine_over_coarse']:.1f}x the area figure, and only "
        f"{stab['neighbourhood']['units_above_min']} of "
        f"{stab['neighbourhood']['units']} neighbourhoods clear "
        f"{C.MIN_LISTINGS_PER_AREA} listings at all.", 108))
    viz.save(fig, C.FIG / "12_neighbourhood_positioning.png")

    metrics = {
        "areas_reported": int(len(z)),
        "areas_excluded_small_n": int(frame[AREA].nunique() - len(z)),
        "min_listings_per_area": C.MIN_LISTINGS_PER_AREA,
        "stability": stab,
        "cells": int(z.size),
        "cells_beyond_two_se": int(n_marked),
        "cells_beyond_two_se_share": float(n_marked / z.size),
        "strongest_area_overall": str(best_area),
        "weakest_area_overall": str(worst_area),
        "weakest_area_attributes_below": n_below,
        "attributes_below_city_by_area": {a: int((t.loc[a] < -2).sum())
                                          for a in z.index},
        "attributes_above_city_by_area": {a: int((t.loc[a] > 2).sum())
                                          for a in z.index},
        "strongest_areas_excluding_catch_all": real[:3],
        "weakest_areas_excluding_catch_all": real[-3:][::-1],
        "catch_all_areas_excluded_from_claims": list(C.CATCH_ALL_AREAS),
        "largest_gap_area": str(gap_area),
        "largest_gap_aspect": str(gap_aspect),
        "largest_gap_sd": gap_val,
        "area_mean_z": {a: float(v) for a, v in z.mean(axis=1).items()},
        "area_table": {a: {"listings": int(prof.loc[a, "listings"]),
                           "median_price": float(prof.loc[a, "median_price"]),
                           "mean_rating": float(prof.loc[a, "mean_rating"]),
                           "mean_price_premium": float(prof.loc[a, "mean_price_premium"])}
                       for a in z.index},
        "aspect_z_by_area": {a: {c: float(z.loc[a, c]) for c in z.columns}
                             for a in z.index},
    }
    (C.OUT / "positioning_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("aspect_z_by_area", "area_table")}, indent=2))


if __name__ == "__main__":
    main()
