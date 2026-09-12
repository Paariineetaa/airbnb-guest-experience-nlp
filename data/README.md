# Data

## What is here

`data/raw/` is gitignored and empty on clone. `make data` checks for the two
files and prints where to get them.

| File | Rows | Used for |
|---|---|---|
| `reviews.csv` | 84,849 | The corpus. Everything in `aspects.py` and `topics.py` |
| `listings.csv` | 3,818 × 92 columns | Outcomes and controls. 32 columns are kept |
| `calendar.csv` | 1,393,570 | **Not used here.** It ships with the same scrape and is the input to the pricing project |

`data/processed/` holds the parquet caches written by `make all`; also
gitignored. `data/sample/` holds a committed 220-listing extract so the test
suite runs without a download.

## Provenance

Inside Airbnb, Seattle, scraped **4 January 2016**.

- Primary: <http://insideairbnb.com/get-the-data/> (older scrapes move to the
  archive tab)
- Mirror: <https://www.kaggle.com/datasets/airbnb/seattle> — "Seattle Airbnb
  Open Data", the copy most people can actually download today

Licensed CC BY 4.0. Scraped from public listing pages. The only personal data
is public reviewer and host first names, and both are stop-listed out of every
model in this repo.

## Schema: `reviews.csv`

| Column | Type | Notes |
|---|---|---|
| `listing_id` | int | Joins to `listings.id`. Every review in this extract has a matching listing |
| `id` | int | Review id, unique |
| `date` | date | 2009-06-07 to 2016-01-03 |
| `reviewer_id` | int | 74,436 distinct reviewers |
| `reviewer_name` | str | First name only. Stop-listed in `topics.py` |
| `comments` | str | Free text. 18 nulls. Median 315 characters, 56 words |

## Schema: `listings.csv` — the columns this project uses

| Column | Type | Notes |
|---|---|---|
| `id` | int | Listing id |
| `review_scores_rating` | float 0-100 | **Outcome 1.** Present for 3,171 of 3,818. Mean 94.5, median 96 |
| `review_scores_{cleanliness,location,value,communication,checkin,accuracy}` | float 0-10 | Used for convergent validity, not as model inputs. All six have a median of 10 |
| `price` | str | Currency string, `"$85.00"`. Parsed by `prepare_data.money_to_float` |
| `neighbourhood_group_cleansed` | str | 17 values. The reporting level |
| `neighbourhood_cleansed` | str | 87 values. Too thin to report on — see `docs/method.md` |
| `room_type` | str | Entire home/apt (2,541), Private room (1,160), Shared room (117) |
| `accommodates` | int | 1 to 16. Banded 1-2 / 3-4 / 5-6 / 7+ for comp sets |
| `bedrooms`, `beds`, `bathrooms` | float | Some nulls; bedrooms median-imputed |
| `amenities` | str | Brace-delimited list. Counted, not parsed |
| `host_since` | date | Converted to tenure in days against the scrape date |
| `host_name` | str | Stop-listed in `topics.py`; not otherwise used |
| `host_is_superhost` | str t/f | **Deliberately not used as a control** — it is awarded partly on ratings, so it is downstream of the outcome |
| `number_of_reviews`, `reviews_per_month` | float | Platform counts. The models use the post-cleaning count instead |

## Known quirks

- **805 reviews are Airbnb's own cancellation notices**, not guest opinion.
  They read *"The host canceled this/my/our reservation N days before arrival.
  This is an automated posting."* They are fluent English, they name the host,
  and VADER scores them mildly negative. Removing them is the single most
  important cleaning rule in the project.
- **`price` is what the host asked on 4 January 2016**, not what they achieved.
  There is no booked-price field anywhere in this dataset.
- **`review_scores_rating` is missing for 647 listings** — those with too few
  reviews for Airbnb to publish a score. They are kept for corpus description
  and excluded from modelling.
- **"Other neighborhoods" is a residual bucket**, not a place. It is 794 raw
  listings, a fifth of the portfolio, and is excluded from every best-area or
  worst-area claim.
- **The final month is four days long.** The scrape stopped on 4 January 2016,
  so January 2016 is a stub in any time series and is excluded from growth
  statements.
- **`neighbourhood` and `neighbourhood_cleansed` disagree** for some listings;
  only the `_cleansed` fields are used, since they are the ones Inside Airbnb
  normalises.
- **1.0% of reviews are not in English.** Chinese, German, French, Spanish,
  Dutch and Portuguese all appear. See `docs/method.md` for how they are
  detected and what the screen misses.
