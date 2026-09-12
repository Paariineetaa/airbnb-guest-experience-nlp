# This folder is empty on purpose

Raw data is **not committed** to the repository. It is large, it is not ours to
redistribute, and a repo full of CSVs reads as a data dump rather than an
analysis. `.gitignore` excludes everything in here.

## What belongs here

    listings.csv  and  reviews.csv

## How to get it

download manually (see data/README.md), then `make data` verifies them

Full provenance, expected row counts and known quirks: `../README.md`.

Once the files are in place, `make all` from the repo root rebuilds every
figure and every metric. Nothing else in the repo needs editing.
