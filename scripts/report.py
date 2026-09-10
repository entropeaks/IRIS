"""Aggregate experiment records: recall, cost, paired comparisons, confusions.

    python scripts/report.py results/records.jsonl
    python scripts/report.py results/records.jsonl --against hsv --groups miscellaneous/groups.json

Reads only. Every question asked here costs a read of the records rather than a
re-run, which is why `evaluate.py` writes one line per (seed, fold) instead of an
average.

Comparisons are paired on the draws two experiments actually share. An unpaired
difference means little on this dataset: a single draw carries several points of
recall noise, so two configurations must be judged on the same splits.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments import load_records


def by_experiment(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["experiment"]].append(record)
    return dict(grouped)


def recall_table(grouped: dict, recall_k: list[str]) -> str:
    header = f"{'experiment':<24}{'draws':>7}" + "".join(f"{'R@' + k:>14}" for k in recall_k)
    lines = [header, "-" * len(header)]
    for name, records in sorted(grouped.items(),
                                key=lambda kv: -np.mean([r["recall"]["1"] for r in kv[1]])):
        cells = ""
        for k in recall_k:
            values = np.array([r["recall"][k] for r in records]) * 100
            cells += f"{values.mean():>8.1f}+/-{values.std():<5.1f}"
        lines.append(f"{name:<24}{len(records):>7}{cells}")
    return "\n".join(lines)


def cost_table(grouped: dict) -> str:
    stages = sorted({s for records in grouped.values() for s in records[0]["costs"]})
    header = f"{'experiment':<24}" + "".join(f"{s + ' (ms)':>22}" for s in stages)
    lines = [header, "-" * len(header)]
    for name, records in sorted(grouped.items()):
        cells = ""
        for stage in stages:
            per_call = [r["costs"][stage]["seconds_per_call"] * 1000
                        for r in records if stage in r["costs"]]
            cells += f"{np.mean(per_call):>22.1f}" if per_call else f"{'-':>22}"
        lines.append(f"{name:<24}{cells}")
    return "\n".join(lines)


def paired_table(grouped: dict, baseline: str, recall_k: list[str]) -> str:
    """Compare each experiment to the baseline on the draws they share."""
    reference = {(r["seed"], r["fold"]): r for r in grouped[baseline]}
    header = (f"{'experiment':<24}{'shared':>8}{'gap R@1':>10}"
              f"{'win/loss':>11}{'p':>10}")
    lines = [header, "-" * len(header)]
    for name, records in sorted(grouped.items()):
        if name == baseline:
            continue
        pairs = [(r, reference[(r["seed"], r["fold"])]) for r in records
                 if (r["seed"], r["fold"]) in reference]
        if not pairs:
            lines.append(f"{name:<24}{'0':>8}   no shared draw")
            continue
        gaps = np.array([(a["recall"]["1"] - b["recall"]["1"]) * 100 for a, b in pairs])
        p = wilcoxon(gaps).pvalue if np.any(gaps != 0) else 1.0
        flag = " *" if p < 0.05 else ""
        lines.append(f"{name:<24}{len(pairs):>8}{gaps.mean():>+10.2f}"
                     f"{f'{(gaps > 0).sum()}/{(gaps < 0).sum()}':>11}{p:>10.2g}{flag}")
    return "\n".join(lines)


def confusion_table(grouped: dict, groups_path: str=None, top: int=8) -> str:
    """Which classes get mistaken for which, and how much of that is near-duplicates."""
    families = {}
    if groups_path:
        for family, members in json.loads(Path(groups_path).read_text()).items():
            for member in members:
                families[int(member)] = family

    lines = []
    for name, records in sorted(grouped.items()):
        pairs = [tuple(p) for r in records for p in r["confusion"] if p[0] != p[1]]
        if not pairs:
            lines.append(f"{name}: no errors"); continue

        same_family = sum(1 for true, pred in pairs
                          if families.get(true) and families[true] == families.get(pred))
        note = (f", {same_family}/{len(pairs)} inside a near-duplicate family"
                if families else "")
        lines.append(f"\n{name}: {len(pairs)} errors{note}")
        for (true, pred), count in Counter(pairs).most_common(top):
            family = families.get(true)
            tag = f"  [{family}]" if family and family == families.get(pred) else ""
            lines.append(f"  {true:>4} taken for {pred:<4} x{count}{tag}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", help="JSONL written by scripts/evaluate.py")
    parser.add_argument("--against", help="experiment to compare the others against")
    parser.add_argument("--groups", help="near-duplicate families, for error analysis")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    records = load_records(args.records)
    grouped = by_experiment(records)
    recall_k = sorted(records[0]["recall"], key=int)

    print(f"{len(records)} records, {len(grouped)} experiments\n")
    print(recall_table(grouped, recall_k))
    print("\nCost per call, averaged over draws\n")
    print(cost_table(grouped))

    if args.against:
        if args.against not in grouped:
            raise SystemExit(f"no experiment named {args.against!r}; have {sorted(grouped)}")
        print(f"\nPaired against {args.against}, Wilcoxon signed-rank, * = p<0.05\n")
        print(paired_table(grouped, args.against, recall_k))

    print("\nMost frequent confusions")
    print(confusion_table(grouped, args.groups, args.top))


if __name__ == "__main__":
    main()
