"""
The cleaning rules, tested against rows that need cleaning.

The rule that matters most is the cancellation filter. Airbnb writes those
reviews itself when a booking falls through; they are fluent English, they name
the host, and VADER scores them mildly negative. Leaving them in would attach a
host-communication complaint to every listing that ever had a cancellation -
a bias pointed in exactly the direction that would make the analysis look more
interesting than it is.
"""
from __future__ import annotations

import pandas as pd

import config as C
import prepare_data as P
import text_utils as T


def test_sample_actually_contains_rows_that_need_cleaning(sample_reviews):
    dirty = sample_reviews[sample_reviews["from_raw"]]
    assert len(dirty) > 0, "the sample has nothing for the cleaning tests to catch"
    assert dirty["comments"].map(T.is_cancellation).any()


def test_cancellation_notices_are_removed(raw_like, sample_listings):
    before = int(raw_like["comments"].map(T.is_cancellation).sum())
    assert before > 0
    out, _ = P.clean_reviews(raw_like, set(sample_listings["id"]))
    assert not out["comments"].map(T.is_cancellation).any()


def test_no_surviving_review_is_an_automated_posting(raw_like, sample_listings):
    out, _ = P.clean_reviews(raw_like, set(sample_listings["id"]))
    assert not out["comments"].str.contains("automated posting", case=False).any()
    assert not out["comments"].str.lower().str.startswith("the host canceled").any()


def test_empty_and_duplicate_reviews_are_removed(raw_like, sample_listings):
    extra = raw_like.head(3).copy()
    extra["id"] = extra["id"] + 10**9
    blank = raw_like.head(2).copy()
    blank["id"] = blank["id"] + 2 * 10**9
    blank["comments"] = ["", "   "]
    stressed = pd.concat([raw_like, extra, blank], ignore_index=True)

    out, log = P.clean_reviews(stressed, set(sample_listings["id"]))
    assert (out["comments"].str.len() > 0).all()
    assert not out.duplicated(subset=["listing_id", "reviewer_id", "comments"]).any()
    removed = log.set_index("step")["rows_removed"]
    assert removed["drop empty comment"] >= 2
    assert removed["drop duplicated reviewer/listing/text"] >= 3


def test_cleaning_log_row_counts_reconcile(raw_like, sample_listings):
    out, log = P.clean_reviews(raw_like, set(sample_listings["id"]))
    steps = log[~log["step"].isin(["raw rows loaded", "final"])]
    running = len(raw_like)
    for _, r in steps.iterrows():
        running -= int(r["rows_removed"])
        assert int(r["rows_remaining"]) == running, r["step"]
    assert int(log.iloc[-1]["rows_remaining"]) == len(out)


def test_language_screen_drops_other_scripts_but_keeps_terse_english():
    kw = {"min_ascii": C.MIN_ASCII_RATIO, "min_foreign": C.MIN_FOREIGN_STOPWORDS}
    assert not T.looks_english("房间的描述与实际相符，离华盛顿西雅图分校很近", **kw)
    assert not T.looks_english("네 모두가 정확했습니다", **kw)
    assert not T.looks_english(
        "Die Wohnung ist sehr sauber und die Lage ist perfekt fuer uns gewesen",
        **kw)
    # The failure mode this heuristic was rewritten to avoid: telegraphic
    # English reviews carry the densest aspect content in the whole corpus.
    for terse in ("Great location, comfy bed - easy check-in.",
                  "Clean, quiet, close to everything.",
                  "Perfect spot. Would return."):
        assert T.looks_english(terse, **kw), terse


def test_money_parsing_handles_thousands_and_blanks():
    s = pd.Series(["$85.00", "$1,250.00", None, "", "$20"])
    out = P.money_to_float(s)
    assert out.iloc[0] == 85.0
    assert out.iloc[1] == 1250.0
    assert out.iloc[4] == 20.0
    assert out.isna().sum() == 2


def test_sentence_splitter_treats_newlines_as_terminators():
    text = "Great place.\nHost was kind! Would return?  Yes"
    assert T.sentences(text) == ["Great place.", "Host was kind!",
                                 "Would return?", "Yes"]
    assert T.sentences("") == []
