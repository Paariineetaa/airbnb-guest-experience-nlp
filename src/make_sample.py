"""
Cut a committed extract so the tests run without the raw download.

Whole listings are kept, never random reviews. Every aspect feature is a
listing-level aggregate, so a row sample would leave the tests checking
aggregates built from fragments. Listings are drawn at random from those with
enough reviews to exercise the aggregation, not from the busiest.

Reviews in other languages and automated cancellation notices are re-injected
from the raw file, so `tests/test_cleaning.py` has something to catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import text_utils as T

N_LISTINGS = 220
MIN_REVIEWS = 8
N_AWKWARD = 60


def main() -> None:
    reviews = pd.read_parquet(C.REVIEWS_PARQUET)
    listings = pd.read_parquet(C.LISTINGS_PARQUET)

    counts = reviews.groupby("listing_id").size()
    eligible = counts[counts >= MIN_REVIEWS].index.to_numpy()
    rng = np.random.default_rng(C.RANDOM_STATE)
    keep = rng.choice(eligible, size=min(N_LISTINGS, len(eligible)), replace=False)

    rv = reviews[reviews["listing_id"].isin(keep)].copy()
    lst = listings[listings["id"].isin(keep)].copy()

    # Cleaning tests need rows that cleaning removes, and the cleaned parquet
    # has none. They come from the raw file, restricted to the sampled
    # listings, and marked so the fixture can separate them from cleaned rows.
    raw = pd.read_csv(C.RAW_REVIEWS)
    raw["comments"] = raw["comments"].fillna("").map(T.normalise)
    raw["date"] = pd.to_datetime(raw["date"])
    awkward = raw[
        raw["listing_id"].isin(keep)
        & (raw["comments"].map(T.is_cancellation)
           | ~raw["comments"].map(
               lambda t: T.looks_english(t, C.MIN_ASCII_RATIO,
                                         C.MIN_FOREIGN_STOPWORDS)))
    ]
    if len(awkward) > N_AWKWARD:
        awkward = awkward.sample(N_AWKWARD, random_state=C.RANDOM_STATE)

    rv["from_raw"] = False
    awkward = awkward.assign(from_raw=True, is_short=False,
                             n_chars=awkward["comments"].str.len(),
                             n_words=awkward["comments"].str.count(r"\s+") + 1)
    out = pd.concat([rv, awkward[rv.columns]], ignore_index=True)

    C.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.SAMPLE_REVIEWS, index=False)
    lst.to_parquet(C.SAMPLE_LISTINGS, index=False)
    print(f"{len(out):,} reviews ({int(out['from_raw'].sum())} deliberately dirty) "
          f"across {out['listing_id'].nunique()} listings "
          f"-> {C.SAMPLE_REVIEWS} ({C.SAMPLE_REVIEWS.stat().st_size / 1e6:.2f} MB)")
    print(f"{len(lst):,} listings -> {C.SAMPLE_LISTINGS} "
          f"({C.SAMPLE_LISTINGS.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
