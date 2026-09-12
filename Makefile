.PHONY: help data all test lint sample clean

PY := python
SRC := src

help:
	@echo "make data   - check for the raw CSVs and print where to get them"
	@echo "make all    - run the full pipeline, rebuilding every figure and metric"
	@echo "make test   - cleaning, extraction, comp-set and minimum-n checks"
	@echo "make lint   - ruff"
	@echo "make sample - re-cut the committed test sample from processed data"
	@echo "make clean  - remove generated outputs and processed data"

data:
	cd $(SRC) && $(PY) fetch_data.py

all:
	cd $(SRC) && $(PY) run_all.py

test:
	$(PY) -m pytest tests -q

lint:
	ruff check $(SRC) tests

sample:
	cd $(SRC) && $(PY) make_sample.py

clean:
	rm -rf outputs/figures/*.png outputs/tables/*.csv outputs/*.json \
	       data/processed/* .pytest_cache
