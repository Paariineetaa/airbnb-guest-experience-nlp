"""
Point at the raw data. Nothing here is committed. Raw data does not belong in
a git repository.

The canonical home is the Inside Airbnb archive, which republishes Seattle on
a rolling basis and retires old scrapes. The 4 January 2016 vintage this
project uses is mirrored on Kaggle as "Seattle Airbnb Open Data". Both are
listed so a reader can tell which one they got.
"""
from __future__ import annotations

import csv
import sys

import config as C

SOURCES = [
    ("reviews.csv", C.RAW_REVIEWS, 84_849,
     "listing_id, id, date, reviewer_id, reviewer_name, comments"),
    ("listings.csv", C.RAW_LISTINGS, 3_818,
     "id, price, review_scores_*, neighbourhood_group_cleansed, room_type, ..."),
]

INSTRUCTIONS = """
Download the Seattle 4 January 2016 scrape and put the two CSVs in data/raw/.

  Primary   Inside Airbnb archive   http://insideairbnb.com/get-the-data/
            (Seattle, 2016-01-04; older scrapes move to the archive tab)

  Mirror    Kaggle, "Seattle Airbnb Open Data"
            https://www.kaggle.com/datasets/airbnb/seattle
            kaggle datasets download -d airbnb/seattle -p data/raw --unzip

calendar.csv ships with both and is not used here - it is the input to the
pricing project, not this one.

Licence: Inside Airbnb publishes under CC BY 4.0. The data is scraped from
public listing pages; no private or personally identifying information beyond
public reviewer first names is included, and reviewer names are stop-listed
out of every model in this repo.
"""


def count_records(path) -> int:
    """
    Number of CSV data rows, not the number of lines in the file.

    Airbnb's free-text columns - `description`, `house_rules`, `comments` -
    hold newlines inside quoted fields, so `wc -l` reports 10,466 for a
    3,818-row listings file. The csv module respects the quoting. Field size
    is raised because one review can exceed the 128 KB default.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def main() -> None:
    problems = []
    for name, path, expected, cols in SOURCES:
        if not path.exists():
            problems.append(f"  {name}: MISSING - expected {expected:,} rows: {cols}")
            continue
        rows = count_records(path)
        if abs(rows - expected) > expected * 0.01:
            problems.append(f"  {name}: {rows:,} rows, expected {expected:,} "
                            "- looks truncated or from a different scrape")
        else:
            print(f"  {name}: {rows:,} rows  OK "
                  f"({path.stat().st_size / 1e6:.1f} MB)")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(INSTRUCTIONS, file=sys.stderr)
        raise SystemExit(1)
    print("\nAll raw files present and the right size. Run `make all`.")


if __name__ == "__main__":
    main()
