"""Paths and analysis constants. Every tunable value lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

for _p in (DATA_RAW, DATA_PROC, DATA_SAMPLE, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

RAW_REVIEWS = DATA_RAW / "reviews.csv"
RAW_LISTINGS = DATA_RAW / "listings.csv"

REVIEWS_PARQUET = DATA_PROC / "reviews.parquet"
LISTINGS_PARQUET = DATA_PROC / "listings.parquet"
ASPECTS_REVIEW_PARQUET = DATA_PROC / "aspect_review.parquet"
ASPECTS_LISTING_PARQUET = DATA_PROC / "aspect_listing.parquet"
MODEL_FRAME_PARQUET = DATA_PROC / "model_frame.parquet"

SAMPLE_REVIEWS = DATA_SAMPLE / "reviews_sample.parquet"
SAMPLE_LISTINGS = DATA_SAMPLE / "listings_sample.parquet"

CURRENCY = "$"
SCRAPE_DATE = "2016-01-04"
CITY = "Seattle"

RANDOM_STATE = 42
TEST_SIZE = 0.30

# cleaning thresholds
# Reviews shorter than this carry no aspect content ("Great!", "A+"). Kept in
# the corpus counts, excluded from aspect scoring. The split is logged.
MIN_REVIEW_CHARS = 20

# Language screen. A review is non-English if too few of its characters are
# ASCII (a non-Latin script), or if it carries at least MIN_FOREIGN_STOPWORDS
# distinct non-English function words and more of them than English ones.
# Requiring English function words to be *present* discarded 1,720 telegraphic
# English reviews ("Great location, comfy bed - easy check-in"), the ones with
# the densest aspect content. See text_utils.looks_english.
MIN_ASCII_RATIO = 0.90
MIN_FOREIGN_STOPWORDS = 3

# modelling sample
# A listing needs enough reviews for its mean aspect sentiment to be a
# measurement. 10 is a bias/variance compromise. drivers.py re-runs the
# headline model at 5 and 20 into outputs/driver_metrics.json.
MIN_REVIEWS_FOR_MODEL = 10
MIN_REVIEWS_SENSITIVITY = (5, 20)

# A listing must mention an aspect this often before its per-aspect sentiment
# is used. Below it the mean is one offhand sentence.
MIN_ASPECT_MENTIONS = 2

# comp set (price premium)
# Comp set = same neighbourhood group x room type x capacity band. The
# baseline is a LEAVE-ONE-OUT median. A listing is never part of its own
# benchmark. A comp set of one would otherwise report a premium of exactly
# zero rather than "no comparable available".
ACCOMMODATES_BINS = (0, 2, 4, 6, 100)
ACCOMMODATES_LABELS = ("1-2", "3-4", "5-6", "7+")
MIN_COMP_SET = 8          # peers required, excluding the listing itself

# neighbourhood reporting
# Seattle has 87 `neighbourhood_cleansed` values. The smallest hold a handful
# of listings, where a neighbourhood mean is noise. Reporting happens at the
# 17-value `neighbourhood_group_cleansed` level, and a group must clear this
# many modelled listings to be shown.
MIN_LISTINGS_PER_AREA = 25

# "Other neighborhoods" is Inside Airbnb's residual bucket, not a place. A
# fifth of the portfolio, so it stays on the map. Excluded from any "best
# area" claim: an average over everywhere that fit nowhere else is not a
# location an owner can act on.
CATCH_ALL_AREAS = ("Other neighborhoods",)

# rating bands used by topics.py
# Bands are percentiles, not round numbers. The rating distribution is
# compressed at the top: "below 90" is the bottom 13% of listings.
LOW_BAND_PCTL = 25
HIGH_BAND_PCTL = 75

# topic model
N_TOPICS = 10
TOPIC_MAX_FEATURES = 12_000
TOPIC_MIN_DF = 40
TOPIC_MAX_DF = 0.25
TOPIC_MAX_DOCS_PER_BAND = 20_000

# sentiment
# VADER's own published thresholds. Not tuned.
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

N_BOOTSTRAP = 400

# lexicon audit
# aspects.py re-mines the corpus vocabulary to check what the frozen lexicon
# in lexicon.py is missing. A 6,000 review sample matches the full 83,000 at
# the frequencies that matter, and keeps the pipeline under a minute.
LEXICON_AUDIT_SAMPLE = 6_000
LEXICON_AUDIT_TOP_N = 150
LEXICON_AUDIT_MIN_COUNT = 25
