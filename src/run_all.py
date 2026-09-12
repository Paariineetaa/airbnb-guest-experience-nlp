"""Run the pipeline in order. `make all` calls this."""
from __future__ import annotations

import importlib
import time

STEPS = ["prepare_data", "analysis_corpus", "aspects", "drivers",
         "topics", "priority_matrix", "positioning"]


def main() -> None:
    t0 = time.time()
    for i, name in enumerate(STEPS, 1):
        t = time.time()
        print(f"\n{'=' * 66}\n[{i}/{len(STEPS)}] {name}\n{'=' * 66}")
        importlib.import_module(name).main()
        print(f"-- {name} finished in {time.time() - t:.0f}s")
    print(f"\nDone in {time.time() - t0:.0f}s. See outputs/.")


if __name__ == "__main__":
    main()
