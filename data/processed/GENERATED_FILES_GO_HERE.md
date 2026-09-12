# This folder is empty on purpose

The pipeline writes its cached intermediate tables here on the first run.
Nothing in it is committed, and nothing in it needs to be backed up - `make all`
regenerates all of it from `data/raw/`.

Safe to delete at any time.
