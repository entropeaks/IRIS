"""Run one or more experiment configs, appending a record per (seed, fold).

    python scripts/evaluate.py configs/hsv.yaml configs/hsv_reranked.yaml

Each config is evaluated on its own draws and its records appended to the same
file. Nothing is aggregated here: `scripts/report.py` reads the file afterwards,
so a new question costs a read rather than a re-run.

Sweeping is a shell loop or a list of files -- there is one pipeline shape, so
the variation lives in the configs.
"""

import argparse
import os
import sys
from pathlib import Path

from src.experiments import ExperimentConfig, append_records, run


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="YAML experiment configs")
    parser.add_argument("--out", default="results/records.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for path in args.configs:
        config = ExperimentConfig.from_yaml(path)
        draws = config.data.k_folds * len(config.data.seeds)
        print(f"{config.name}  [{config.fingerprint()}]  {draws} draws", flush=True)

        records = run(config, quiet=not args.verbose)
        append_records(records, args.out)

        best = max(config.recall_k)
        mean = sum(r["recall"]["1"] for r in records) / len(records)
        print(f"  R@1 {mean:.3f} over {len(records)} draws -> {args.out}", flush=True)


if __name__ == "__main__":
    # src.data walks sets of class names, so the split depends on string hashing
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
