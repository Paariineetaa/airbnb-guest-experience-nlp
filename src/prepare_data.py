"""
Step 1. Load the two raw CSVs, clean them, cache to parquet.

Every exclusion is counted and written to outputs/tables/cleaning_log.csv.

Outputs
  data/processed/reviews.parquet
  data/processed/listings.parquet
  outputs/tables/cleaning_log.csv
  outputs/data_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
import text_utils as T

SUBSCORES = [
    "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [p for p in (C.RAW_REVIEWS, C.RAW_LISTINGS) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing raw data: " + ", ".join(str(p) for p in missing)
            + "\nRun `make data` (or see data/README.md) first."
        )
    reviews = pd.read_csv(C.RAW_REVIEWS)
    listings = pd.read_csv(C.RAW_LISTINGS, low_memory=False)
    return reviews, listings


def money_to_float(s: pd.Series) -> pd.Series:
    """'$1,250.00' -> 1250.0. Blank and NaN stay NaN."""
    return pd.to_numeric(
        s.astype("string").str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def clean_reviews(raw: pd.DataFrame, listing_ids: set[int]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    log: list[dict] = []
    n0 = len(raw)
    log.append({"table": "reviews", "step": "raw rows loaded", "rows_removed": 0,
                "rows_remaining": n0,
                "rationale": "reviews.csv as published by Inside Airbnb"})

    df = raw.copy()
    df["comments"] = df["comments"].map(T.normalise)
    df["date"] = pd.to_datetime(df["date"])

    def step(name: str, keep: pd.Series, why: str) -> pd.DataFrame:
        nonlocal df
        log.append({"table": "reviews", "step": name,
                    "rows_removed": int((~keep).sum()),
                    "rows_remaining": int(keep.sum()), "rationale": why})
        df = df[keep].copy()
        return df

    # 1. Empty text. 18 rows have a null comment. They are review *events*
    #    with no opinion attached.
    step("drop empty comment", df["comments"].str.len() > 0,
         "null or whitespace-only comment: a review event with no text")

    # 2. Automated cancellation notices. Airbnb writes these itself. They are
    #    fluent English, they mention the host, and VADER reads them as mildly
    #    negative. Leaving them in would penalise the host-communication score
    #    of every listing that ever had a cancellation.
    is_auto = df["comments"].map(T.is_cancellation)
    step("drop automated cancellation notices", ~is_auto,
         "platform-generated text ('The host canceled this reservation...'), "
         "not guest opinion")

    # 3. Orphan reviews. None in this extract. The join is asserted, not
    #    assumed.
    step("drop reviews with no matching listing", df["listing_id"].isin(listing_ids),
         "review references a listing_id absent from listings.csv")

    # 4. Language screen. Documented in text_utils.looks_english. The aspect
    #    lexicon and VADER are English-only, so a Spanish review would score
    #    as neutral-with-no-mentions and dilute every mean.
    is_en = df["comments"].map(
        lambda t: T.looks_english(t, C.MIN_ASCII_RATIO, C.MIN_FOREIGN_STOPWORDS))
    step("drop non-English reviews", is_en,
         "non-Latin script, or more non-English than English function words; "
         "the aspect lexicon and VADER are English-only")

    # Two audits of the rules above, with live numbers.
    #
    # (a) The rejected language rule. The first design *required* English
    #     function words to be present. This counts what that would have cost.
    strict_drop = int((df["comments"].map(
        lambda t: T.stopword_hits(t)[0]) < 2).sum())
    log.append({"table": "reviews", "step": "language rule counterfactual (kept)",
                "rows_removed": 0, "rows_remaining": len(df),
                "rationale": f"a rule requiring 2+ English function words to be "
                             f"present would additionally drop {strict_drop:,} "
                             "surviving reviews, most of them terse English "
                             "('Great location, comfy bed'); rejected for that reason"})

    # (b) The anchored cancellation rule. Unanchored, it would also delete
    #     genuine reviews that mention a cancellation in passing.
    passing = int(df["comments"].str.contains(r"cancell?ed", case=False,
                                              regex=True).sum())
    log.append({"table": "reviews", "step": "cancellation rule counterfactual (kept)",
                "rows_removed": 0, "rows_remaining": len(df),
                "rationale": f"{passing:,} surviving reviews mention a cancellation "
                             "in passing and are guest opinion; the filter is "
                             "anchored to the start of the review to keep them"})

    # 5. Exact duplicate text from the same reviewer on the same listing.
    keep = ~df.duplicated(subset=["listing_id", "reviewer_id", "comments"], keep="first")
    step("drop duplicated reviewer/listing/text", keep,
         "same reviewer, same listing, identical text: a double-post")

    # 6. Reviews under MIN_REVIEW_CHARS stay in the corpus, flagged out of
    #    aspect scoring. "Great!" counts as volume, not as aspect content.
    df["is_short"] = df["comments"].str.len() < C.MIN_REVIEW_CHARS
    log.append({"table": "reviews", "step": "flag very short reviews (kept)",
                "rows_removed": 0, "rows_remaining": len(df),
                "rationale": f"{int(df['is_short'].sum()):,} reviews under "
                             f"{C.MIN_REVIEW_CHARS} chars kept for corpus counts, "
                             "excluded from aspect scoring"})

    # Recall audit for the language screen. A survivor carrying a non-English
    # function word and no English one is a suspected miss. An upper bound on
    # one failure mode, not a full error rate: a short foreign review with no
    # function words is invisible to it. Quoted in docs/method.md.
    _hits = df["comments"].map(T.stopword_hits)
    df["_suspect_foreign"] = [f >= 1 and e == 0 for e, f in _hits]
    log.append({"table": "reviews", "step": "language screen recall audit (kept)",
                "rows_removed": 0, "rows_remaining": len(df),
                "rationale": f"{int(df['_suspect_foreign'].sum()):,} surviving reviews "
                             "carry a non-English function word and no English one: "
                             "an upper bound on one failure mode of the screen"})

    df["n_chars"] = df["comments"].str.len()
    df["n_words"] = df["comments"].str.count(r"\s+") + 1
    df = df.sort_values("date").reset_index(drop=True)
    df.attrs["language_suspect_misses"] = int(df.pop("_suspect_foreign").sum())
    df.attrs["strict_language_rule_would_drop"] = strict_drop
    df.attrs["cancellation_mentions_kept"] = passing

    log.append({"table": "reviews", "step": "final", "rows_removed": n0 - len(df),
                "rows_remaining": len(df),
                "rationale": f"{(n0 - len(df)) / n0:.1%} of raw review rows removed"})
    return df, pd.DataFrame(log)


def clean_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log: list[dict] = []
    n0 = len(raw)
    log.append({"table": "listings", "step": "raw rows loaded", "rows_removed": 0,
                "rows_remaining": n0, "rationale": "listings.csv, 92 columns"})

    keep_cols = [
        "id", "name", "host_name", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
        "latitude", "longitude", "property_type", "room_type", "accommodates",
        "bathrooms", "bedrooms", "beds", "amenities", "price", "cleaning_fee",
        "minimum_nights", "host_since", "host_is_superhost",
        "host_listings_count", "number_of_reviews", "reviews_per_month",
        "first_review", "last_review", "review_scores_rating", *SUBSCORES,
        "instant_bookable", "cancellation_policy",
    ]
    df = raw[keep_cols].copy()
    log.append({"table": "listings", "step": "select analysis columns",
                "rows_removed": 0, "rows_remaining": len(df),
                "rationale": f"{len(keep_cols)} of {raw.shape[1]} columns kept; "
                             "the rest are URLs, free text and scrape metadata"})

    df["price"] = money_to_float(df["price"])
    df["cleaning_fee"] = money_to_float(df["cleaning_fee"]).fillna(0.0)
    for c in ("host_since", "first_review", "last_review"):
        df[c] = pd.to_datetime(df[c])
    for c in ("host_is_superhost", "instant_bookable"):
        df[c] = df[c].map({"t": True, "f": False})

    def step(name: str, keep: pd.Series, why: str) -> None:
        nonlocal df
        log.append({"table": "listings", "step": name,
                    "rows_removed": int((~keep).sum()),
                    "rows_remaining": int(keep.sum()), "rationale": why})
        df = df[keep].copy()

    step("drop non-positive price", df["price"] > 0,
         "a nightly price of zero is a placeholder, not an offer")
    step("drop missing capacity", df["accommodates"] > 0,
         "capacity is needed to build a comparable set")

    # Kept, not dropped. A listing with no rating belongs in the corpus
    # description and the neighbourhood counts. It is excluded from the
    # modelling frame later, where the outcome is undefined.
    n_no_rating = int(df["review_scores_rating"].isna().sum())
    log.append({"table": "listings", "step": "flag listings with no rating (kept)",
                "rows_removed": 0, "rows_remaining": len(df),
                "rationale": f"{n_no_rating:,} listings have no review_scores_rating "
                             "(too few reviews); kept for description, excluded "
                             "from modelling where the outcome is undefined"})

    scrape = pd.Timestamp(C.SCRAPE_DATE)
    df["host_tenure_days"] = (scrape - df["host_since"]).dt.days
    df["log_price"] = np.log(df["price"])
    df["accommodates_band"] = pd.cut(
        df["accommodates"], bins=list(C.ACCOMMODATES_BINS),
        labels=list(C.ACCOMMODATES_LABELS))
    df["n_amenities"] = (df["amenities"].fillna("{}")
                         .str.count(",").add(1)
                         .where(df["amenities"].str.len() > 2, 0))

    log.append({"table": "listings", "step": "final", "rows_removed": n0 - len(df),
                "rows_remaining": len(df),
                "rationale": f"{(n0 - len(df)) / n0:.1%} of raw listing rows removed"})
    return df.reset_index(drop=True), pd.DataFrame(log)


def main() -> None:
    print("Loading raw CSVs ...")
    raw_reviews, raw_listings = load_raw()

    listings, log_l = clean_listings(raw_listings)
    reviews, log_r = clean_reviews(raw_reviews, set(listings["id"]))

    log = pd.concat([log_l, log_r], ignore_index=True)
    print(log.to_string(index=False))

    reviews.to_parquet(C.REVIEWS_PARQUET, index=False)
    listings.to_parquet(C.LISTINGS_PARQUET, index=False)
    log.to_csv(C.TAB / "cleaning_log.csv", index=False)

    rated = listings["review_scores_rating"].dropna()
    per_listing = reviews.groupby("listing_id").size()

    summary = {
        "reviews_raw": int(len(raw_reviews)),
        "reviews_clean": int(len(reviews)),
        "reviews_removed_share": float(1 - len(reviews) / len(raw_reviews)),
        "reviews_cancellation_notices": int(
            log.loc[log["step"] == "drop automated cancellation notices",
                    "rows_removed"].iloc[0]),
        "reviews_non_english": int(
            log.loc[log["step"] == "drop non-English reviews", "rows_removed"].iloc[0]),
        "reviews_short_flagged": int(reviews["is_short"].sum()),
        "reviews_saved_by_rejecting_strict_language_rule": int(
            reviews.attrs["strict_language_rule_would_drop"]),
        "reviews_mentioning_cancellation_kept": int(
            reviews.attrs["cancellation_mentions_kept"]),
        "language_screen_suspected_misses": int(
            reviews.attrs["language_suspect_misses"]),
        "language_screen_suspected_miss_share": float(
            reviews.attrs["language_suspect_misses"] / len(reviews)),
        "listings_raw": int(len(raw_listings)),
        "listings_clean": int(len(listings)),
        "listings_with_rating": int(rated.notna().sum()),
        "listings_with_any_review": int(per_listing.gt(0).sum()),
        "reviewers": int(reviews["reviewer_id"].nunique()),
        "date_min": str(reviews["date"].min().date()),
        "date_max": str(reviews["date"].max().date()),
        "median_review_chars": float(reviews["n_chars"].median()),
        "mean_review_words": float(reviews["n_words"].mean()),
        "median_reviews_per_listing": float(per_listing.median()),
        "mean_rating": float(rated.mean()),
        "median_rating": float(rated.median()),
        "share_rating_ge_90": float((rated >= 90).mean()),
        "share_rating_ge_95": float((rated >= 95).mean()),
        "rating_p10": float(rated.quantile(0.10)),
        "rating_p25": float(rated.quantile(0.25)),
        "median_price": float(listings["price"].median()),
        "neighbourhood_groups": int(listings["neighbourhood_group_cleansed"].nunique()),
        "neighbourhoods": int(listings["neighbourhood_cleansed"].nunique()),
    }
    (C.OUT / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
