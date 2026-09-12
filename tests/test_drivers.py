"""
The comp-set premium and the model frame.

The bug this file exists for: if a listing is included in the median that
defines its own benchmark, then a comp set of one produces a premium of exactly
zero - not "this listing is priced at the market" but "there is no market to
compare it to". Those rows are indistinguishable from genuinely average
listings and they pull every coefficient towards zero. The premium is therefore
built leave-one-out, and comp sets that are too thin return NaN rather than a
fabricated benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import drivers as D


def _synthetic(n_per_cell: int, prices: list[float] | None = None) -> pd.DataFrame:
    """A single comp set of `n_per_cell` listings with known prices."""
    prices = prices or list(np.linspace(50, 250, n_per_cell))
    return pd.DataFrame({
        "id": range(len(prices)),
        "price": prices,
        "log_price": np.log(prices),
        "neighbourhood_group_cleansed": ["Ballard"] * len(prices),
        "room_type": ["Entire home/apt"] * len(prices),
        "accommodates_band": pd.Categorical(
            ["1-2"] * len(prices), categories=list(C.ACCOMMODATES_LABELS)),
    })


def test_a_listing_is_never_its_own_benchmark():
    df = _synthetic(C.MIN_COMP_SET + 5)
    out = D.comp_set_premium(df)
    for i in out.index:
        peers = np.delete(df["log_price"].to_numpy(), i)
        assert np.isclose(out.loc[i, "comp_median_log_price"], np.median(peers))


def test_a_solo_listing_gets_no_premium_rather_than_a_premium_of_zero():
    """The failure mode in one line: with a plain group median this row would
    report a premium of exactly 0.0 and look perfectly average."""
    df = _synthetic(1)
    out = D.comp_set_premium(df)
    assert out["price_premium"].isna().all()
    assert not (out["price_premium"] == 0).any()


def test_thin_comp_sets_are_dropped_at_the_configured_threshold():
    just_under = D.comp_set_premium(_synthetic(C.MIN_COMP_SET))
    just_over = D.comp_set_premium(_synthetic(C.MIN_COMP_SET + 1))
    # comp_n counts peers, so a cell of MIN_COMP_SET listings has one peer
    # too few and must be dropped entirely.
    assert just_under["price_premium"].isna().all()
    assert just_over["price_premium"].notna().all()
    assert (just_over["comp_n"] == C.MIN_COMP_SET).all()


def test_premium_is_signed_the_way_the_write_up_reads_it():
    n = C.MIN_COMP_SET + 5
    prices = [50.0] * (n // 2) + [200.0] * (n - n // 2)
    out = D.comp_set_premium(_synthetic(n, prices=prices)).sort_values("price")
    assert out["price_premium"].iloc[0] < 0, "the cheap half must read as a discount"
    assert out["price_premium"].iloc[-1] > 0, "the dear half must read as a premium"


def test_identical_prices_give_a_premium_of_exactly_zero():
    """The one case where zero is the right answer, as a guard against
    'fixing' the leave-one-out logic into always returning NaN."""
    out = D.comp_set_premium(_synthetic(C.MIN_COMP_SET + 3,
                                        prices=[120.0] * (C.MIN_COMP_SET + 3)))
    assert np.allclose(out["price_premium"], 0.0)


def test_aspect_features_impute_openly_and_leave_no_gaps(sample_reviews,
                                                         sample_listings):
    import aspects as A

    clean = sample_reviews[~sample_reviews["from_raw"]].copy()
    la = A.to_listing(A.score_reviews(clean), clean)
    frame = D.build_frame(sample_listings, la, min_reviews=5)
    X, coverage = D.aspect_features(frame)

    assert not X.isna().any().any(), "an imputed matrix must have no gaps left"
    assert X.shape[1] == 16
    assert all(0.0 <= v <= 1.0 for v in coverage.values())
    # Imputation is at the column mean, so it cannot move the column mean.
    for a, cov in coverage.items():
        if 0 < cov < 1:
            observed = frame[f"sentiment__{a}"].dropna()
            assert np.isclose(X[f"sent_{a}"].mean(), observed.mean(), atol=1e-9)


def test_model_frame_enforces_the_minimum_review_count(sample_reviews,
                                                       sample_listings):
    import aspects as A

    clean = sample_reviews[~sample_reviews["from_raw"]].copy()
    la = A.to_listing(A.score_reviews(clean), clean)
    for k in (5, 10, 20):
        frame = D.build_frame(sample_listings, la, min_reviews=k)
        assert (frame["n_reviews_scored"] >= k).all()
        assert frame["log_accommodates"].notna().all()
