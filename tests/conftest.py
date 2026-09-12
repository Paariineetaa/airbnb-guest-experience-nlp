"""
Fixtures for the test suite.

Everything runs off the committed sample in `data/sample/`, so the suite needs
no download and CI needs no credentials. The sample deliberately contains rows
that cleaning is supposed to delete - cancellation notices and non-English
reviews - because a cleaning test that only ever sees clean data proves
nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402


@pytest.fixture(scope="session")
def sample_reviews() -> pd.DataFrame:
    if not C.SAMPLE_REVIEWS.exists():
        pytest.skip("run `make sample` first")
    return pd.read_parquet(C.SAMPLE_REVIEWS)


@pytest.fixture(scope="session")
def sample_listings() -> pd.DataFrame:
    if not C.SAMPLE_LISTINGS.exists():
        pytest.skip("run `make sample` first")
    return pd.read_parquet(C.SAMPLE_LISTINGS)


@pytest.fixture(scope="session")
def raw_like(sample_reviews) -> pd.DataFrame:
    """The sample in the shape `prepare_data.clean_reviews` expects as input:
    cleaned rows plus the deliberately dirty ones, columns as in reviews.csv."""
    cols = ["listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments"]
    return sample_reviews[cols].copy()


@pytest.fixture(scope="session")
def scored(sample_reviews) -> pd.DataFrame:
    """Aspect mentions for the clean part of the sample. Built once: scoring
    9,000 reviews takes a few seconds and every aspect test wants the same
    frame."""
    import aspects

    clean = sample_reviews[~sample_reviews["from_raw"]].copy()
    return aspects.score_reviews(clean)
