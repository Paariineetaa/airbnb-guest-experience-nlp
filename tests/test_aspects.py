"""
The aspect extractor: does it fire on the right text, and does the sign mean
what the write-up says it means?

A lexicon-plus-VADER pipeline is easy to get subtly wrong - a stem that
swallows unrelated words, a sentence splitter that hands VADER the wrong span,
an aggregation that treats "never mentioned" as "mentioned neutrally". Each of
those would leave the charts looking entirely reasonable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aspects as A
import config as C
from lexicon import ASPECTS, TERMS, compile_patterns

# One clearly positive and one clearly negative sentence per aspect, each
# written to use vocabulary from that aspect only.
POLARITY_CASES = {
    "cleanliness": ("The apartment was spotless and beautifully clean.",
                    "The apartment was filthy and the bathroom was disgusting."),
    "location": ("Wonderful location, an easy walk to shops and restaurants.",
                 "Terrible location, a miserable walk from any shops."),
    "host_communication": ("The host was warm, helpful and answered every question.",
                           "The host was rude, unresponsive and never answered us."),
    "checkin": ("Check-in was smooth and the key instructions were perfect.",
                "Check-in was a nightmare and the key instructions were wrong."),
    "value": ("Excellent value, worth every penny of the price.",
              "Awful value, overpriced for what you get."),
    "accuracy": ("Exactly as described, the photos were completely accurate.",
                 "Nothing like the description, the photos were misleading."),
    "noise": ("Beautifully quiet, we slept wonderfully every night.",
              "Horribly noisy, we could not sleep for the awful traffic."),
    "space_comfort": ("The bed was wonderfully comfortable and the space generous.",
                      "The bed was awful and the space cramped and depressing."),
}


def _one_review(text: str, listing_id: int = 1, rid: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{"id": rid, "listing_id": listing_id, "comments": text,
                          "is_short": False}])


def test_every_aspect_has_terms_and_a_working_pattern():
    pats = compile_patterns()
    assert set(pats) == set(ASPECTS)
    for a in ASPECTS:
        assert len(TERMS[a]) >= 20, a
        assert pats[a].search(TERMS[a][0].rstrip("*")), a


@pytest.mark.parametrize("aspect", list(ASPECTS))
def test_negative_mention_scores_below_positive_mention(aspect):
    pos, neg = POLARITY_CASES[aspect]
    p = A.score_reviews(_one_review(pos))
    n = A.score_reviews(_one_review(neg))
    p = p[p["aspect"] == aspect]
    n = n[n["aspect"] == aspect]
    assert len(p) == 1, f"positive case did not trigger {aspect}"
    assert len(n) == 1, f"negative case did not trigger {aspect}"
    assert n["sentiment"].iloc[0] < p["sentiment"].iloc[0]
    assert n["sentiment"].iloc[0] <= C.VADER_NEG_THRESHOLD


def test_vader_is_blind_to_factual_praise():
    """
    A limitation, pinned down rather than hidden.

    "Exactly as described, the photos were completely accurate" scores 0.0 on
    VADER: "described" and "accurate" are not in its lexicon, because VADER
    was built for affective language and accuracy is a factual claim. The
    negative direction still works ("misleading" is scored), so the accuracy
    aspect is systematically better at detecting complaints than praise - and
    that asymmetry is the most likely reason its agreement with Airbnb's
    accuracy subscore is the second weakest of the six. Named in
    docs/method.md.
    """
    pos, neg = POLARITY_CASES["accuracy"]
    p = A.score_reviews(_one_review(pos))
    n = A.score_reviews(_one_review(neg))
    p_val = p[p["aspect"] == "accuracy"]["sentiment"].iloc[0]
    n_val = n[n["aspect"] == "accuracy"]["sentiment"].iloc[0]
    assert p_val < C.VADER_POS_THRESHOLD, (
        "if factual praise starts scoring positive, the accuracy caveat in "
        "docs/method.md is out of date")
    assert n_val <= C.VADER_NEG_THRESHOLD


def test_scoring_is_deterministic(sample_reviews):
    clean = sample_reviews[~sample_reviews["from_raw"]].head(1500)
    a = A.score_reviews(clean)
    b = A.score_reviews(clean)
    pd.testing.assert_frame_equal(a, b)


def test_sentiment_is_bounded(scored):
    assert scored["sentiment"].between(-1.0, 1.0).all()
    assert scored["sentiment"].notna().all()
    assert (scored["n_sentences"] >= 1).all()


def test_patterns_do_not_fire_on_substrings():
    pats = compile_patterns()
    # "bar" is a location cue; "barbecue" and "barn" are not.
    assert pats["location"].search("a great bar on the corner")
    assert not pats["location"].search("we used the barbecue in the barn")
    # "key" is a check-in cue; "keyboard" is not.
    assert pats["checkin"].search("the key was in the lockbox")
    assert not pats["checkin"].search("there was a keyboard on the desk")


def test_a_review_with_no_aspect_language_produces_no_rows():
    out = A.score_reviews(_one_review("Thank you so much! Five stars from us."))
    assert len(out) == 0


def test_short_reviews_are_excluded_from_scoring():
    df = _one_review("Clean!", rid=1)
    df.loc[0, "is_short"] = True
    assert len(A.score_reviews(df)) == 0
    assert A.score_reviews(df).attrs["n_scoreable_reviews"] == 0


def test_listing_aggregation_respects_the_minimum_mention_guard(sample_reviews):
    clean = sample_reviews[~sample_reviews["from_raw"]].copy()
    mentions = A.score_reviews(clean)
    la = A.to_listing(mentions, clean)

    for a in ASPECTS:
        thin = la[f"mentions__{a}"] < C.MIN_ASPECT_MENTIONS
        # Below the guard, sentiment is missing rather than zero: a listing
        # nobody talked about must not be recorded as feeling neutral.
        assert la.loc[thin, f"sentiment__{a}"].isna().all(), a
        assert la.loc[~thin, f"sentiment__{a}"].notna().all(), a
        # Mention rate is a share of that listing's own scoreable reviews.
        assert la[f"mention_rate__{a}"].between(0, 1).all(), a
        assert (la[f"mentions__{a}"] >= 0).all(), a


def test_mention_counts_reconcile_with_the_long_frame(sample_reviews):
    clean = sample_reviews[~sample_reviews["from_raw"]].copy()
    mentions = A.score_reviews(clean)
    la = A.to_listing(mentions, clean)
    for a in ASPECTS[:3]:
        expected = (mentions[mentions["aspect"] == a]
                    .groupby("listing_id").size())
        got = la.set_index("listing_id")[f"mentions__{a}"]
        common = expected.index.intersection(got.index)
        assert np.allclose(expected.loc[common], got.loc[common])


def test_negation_is_carried_into_the_score():
    """VADER handles negation; if it stopped, the whole extraction silently
    inverts for a large class of complaints."""
    pos = A.score_reviews(_one_review("The room was clean."))
    neg = A.score_reviews(_one_review("The room was not clean."))
    p = pos[pos["aspect"] == "cleanliness"]["sentiment"].iloc[0]
    n = neg[neg["aspect"] == "cleanliness"]["sentiment"].iloc[0]
    assert n < p
