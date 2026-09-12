"""
Guards on the reporting layer: the minimum-n rules that stop the write-up
quoting a number built from six listings, and the rule that every quoted
number comes out of a metrics file rather than a keyboard.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config as C
import positioning as POS

METRICS = ["data_summary.json", "corpus_metrics.json", "aspect_metrics.json",
           "driver_metrics.json", "topic_metrics.json", "priority_metrics.json",
           "positioning_metrics.json"]


def _frame(counts: dict[str, int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for area, n in counts.items():
        for i in range(n):
            rows.append({"id": len(rows), "neighbourhood_group_cleansed": area,
                         "neighbourhood_cleansed": f"{area}-{i % 3}",
                         "price": 100.0, "review_scores_rating": 95.0,
                         "price_premium": 0.0, "n_reviews_scored": 12,
                         **{f"sentiment__{a}": float(rng.normal(0.5, 0.1))
                            for a in __import__("lexicon").ASPECTS}})
    return pd.DataFrame(rows)


def test_small_areas_are_excluded_from_the_neighbourhood_map():
    big, small = C.MIN_LISTINGS_PER_AREA + 10, C.MIN_LISTINGS_PER_AREA - 1
    z, t, prof = POS.area_profile(_frame({"Big": big, "Also big": big,
                                          "Tiny": small}))
    assert "Tiny" in _frame({"Tiny": small})["neighbourhood_group_cleansed"].values
    assert "Tiny" not in z.index, "an area under the minimum was reported anyway"
    assert set(z.index) == {"Big", "Also big"}
    assert (prof["listings"] >= C.MIN_LISTINGS_PER_AREA).all()


def test_area_profile_is_expressed_in_city_standard_deviations():
    z, t, _ = POS.area_profile(_frame({"A": 60, "B": 60, "Cc": 60}))
    # z-scores against a common city mean must be near-centred across areas
    # once weighted by size; with equal sizes the plain mean is enough.
    assert abs(float(z.to_numpy().mean())) < 0.35
    assert np.isfinite(t.to_numpy()).all()


@pytest.mark.parametrize("name", METRICS)
def test_every_metrics_file_exists_and_is_valid_json(name):
    path = C.OUT / name
    if not path.exists():
        pytest.skip("run `make all` first")
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict) and payload


def test_quoted_headline_numbers_are_present_in_the_metrics_files():
    """If a key the write-up reads disappears, this fails before the README
    silently starts quoting a stale number."""
    required = {
        "data_summary.json": ["reviews_raw", "reviews_clean",
                              "reviews_cancellation_notices", "mean_rating"],
        "corpus_metrics.json": ["share_rating_ge_90", "median_words",
                                "reviews_gini"],
        "aspect_metrics.json": ["mention_rate", "complaint_incidence",
                                "validation_mean_pearson"],
        "driver_metrics.json": ["rating", "price_premium", "rating_coefficients"],
        "priority_metrics.json": ["fix_first", "total_rating_points_available"],
        "positioning_metrics.json": ["weakest_area_overall", "stability"],
    }
    for name, keys in required.items():
        path = C.OUT / name
        if not path.exists():
            pytest.skip("run `make all` first")
        payload = json.loads(path.read_text())
        missing = [k for k in keys if k not in payload]
        assert not missing, f"{name} is missing {missing}"


def test_priority_quadrants_cover_every_aspect_exactly_once():
    from lexicon import ASPECTS

    path = C.OUT / "priority_metrics.json"
    if not path.exists():
        pytest.skip("run `make all` first")
    q = json.loads(path.read_text())["quadrants"]
    assert set(q) == set(ASPECTS)
    buckets = ["fix_first", "maintain", "low_priority", "over_investment"]
    payload = json.loads(path.read_text())
    labelled = [x for b in buckets for x in payload[b]]
    assert len(labelled) == len(ASPECTS)
    assert len(set(labelled)) == len(ASPECTS)
