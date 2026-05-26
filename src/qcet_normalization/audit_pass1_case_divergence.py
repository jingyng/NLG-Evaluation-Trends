#!/usr/bin/env python3
"""Compute the Pass-1 case-divergence diagnostic reported in the paper:

  "Among the 780 lowercase groups containing >= 2 case variants, Pass-1
   classifications disagreed on the assigned QCET target in 35% of groups."

Reads `outputs/stage4_classifications_simple.csv`, which carries both
`stage1_qcet_id` (Pass-1 verdict per raw variant) and `chosen_id`
(post-Pass-2-and-collapse final target). For Pass-1 case-divergence we look
only at `stage1_qcet_id` across raw_string variants that share a lowercase
key.

Run from `paper_code/`:

    python 05_criteria_normalization/audit_pass1_case_divergence.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "outputs" / "stage4_classifications_simple.csv"


def main(csv_path: Path = DEFAULT_CSV) -> None:
    if not csv_path.exists():
        raise SystemExit(f"Stage-4 CSV not found: {csv_path}")

    by_lower: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            raw  = row["raw_string"]
            s1id = row.get("stage1_qcet_id", "") or ""
            by_lower[raw.strip().lower()].append((raw, s1id))

    multi = {k: v for k, v in by_lower.items() if len(v) >= 2}
    diverged = {
        k: v for k, v in multi.items()
        if len({s1id for _raw, s1id in v}) > 1
    }

    n_multi    = len(multi)
    n_diverged = len(diverged)
    pct = (n_diverged / n_multi * 100) if n_multi else 0.0

    print(f"Lowercase groups with >= 2 case variants: {n_multi}")
    print(f"Of those, diverging on stage-1 qcet_id:    {n_diverged}  ({pct:.1f}%)")

    if diverged:
        # Print the first 10 examples for spot-checking
        print()
        print("Sample diverging groups (first 10):")
        for k, items in list(diverged.items())[:10]:
            ids = sorted({s1id or "(none)" for _raw, s1id in items})
            print(f"  {k!r:35s}  {len(items)} variants  →  {ids}")


if __name__ == "__main__":
    main()
