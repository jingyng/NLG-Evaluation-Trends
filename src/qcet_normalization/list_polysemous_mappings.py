#!/usr/bin/env python3
"""List Pass-1 mappings of variants of polysemous root words, BEFORE the
case-collapse / force-merge step that happens in stage 4.

Each row corresponds to one case-sensitive raw variant and shows its
independent Pass-1 verdict alongside the final stage-4 target. Divergence
between the two columns flags variants where the case-collapse step
overwrote a different Pass-1 assignment with the dominant variant's target.

Reviewer fills `proposed_qcet_id` when a per-variant override is desired;
leave blank to keep the current (stage-4) mapping. The override CSV is
then consumed by `apply_polysemous_overrides.py`.

Run from `paper_code/`:

    python 05_criteria_normalization/list_polysemous_mappings.py
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE1_CSV = HERE / "outputs" / "stage1_classifications.csv"
STAGE4_CSV = HERE / "outputs" / "stage4_classifications_simple.csv"
OUT_CSV    = HERE / "polysemous_mappings_review.csv"

# Polysemous root words: criterion strings whose intended QCET target depends
# on paper context (task, co-occurring criteria) rather than the surface form.
# Each entry's `roots` is matched as a substring against the lowercased raw
# string, so e.g. "accuracy" catches "factual accuracy", "translation accuracy",
# "classification accuracy", etc.
POLYSEMOUS_ROOTS: list[str] = [
    "accuracy",
    "faithful",
    "consistency",
    "correctness",
    "alignment",
    "quality",
    "completeness",
    "coverage",
]


def main() -> None:
    for p in (STAGE1_CSV, STAGE4_CSV):
        if not p.exists():
            raise SystemExit(f"Required input not found: {p}")

    # Build {raw_string: stage4_target} lookup so we can show what the
    # force-merge eventually picked next to the Pass-1 verdict.
    stage4_target: dict[str, tuple[str, str, str]] = {}
    with STAGE4_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            stage4_target[r["raw_string"]] = (
                r["chosen_id"],
                r["chosen_name"],
                r.get("chosen_source", ""),
            )

    OCC_FLOOR = 3
    rows_out: list[dict] = []

    with STAGE1_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            raw = r["raw_string"]
            raw_lc = raw.strip().lower()
            occ_l = int(r["occurrences_llm"] or 0)
            occ_h = int(r["occurrences_human"] or 0)
            total = occ_l + occ_h
            if total < OCC_FLOOR:
                continue
            for root in POLYSEMOUS_ROOTS:
                if root in raw_lc:
                    pass1_id   = r["qcet_id"] or ""
                    pass1_name = r["qcet_name"] or ""
                    pass1_fit  = r.get("qcet_fit", "")
                    s4_id, s4_name, s4_src = stage4_target.get(raw, ("", "", ""))
                    diverged = "Y" if (pass1_id and s4_id and pass1_id != s4_id) else ""
                    rows_out.append({
                        "root_word":         root,
                        "raw_lowercase":     raw_lc,
                        "raw_string":        raw,
                        "occurrences_llm":   occ_l,
                        "occurrences_human": occ_h,
                        "total_occ":         total,
                        "pass1_qcet_id":     pass1_id,
                        "pass1_qcet_name":   pass1_name,
                        "pass1_fit":         pass1_fit,
                        "stage4_qcet_id":    s4_id,
                        "stage4_qcet_name":  s4_name,
                        "stage4_source":     s4_src,
                        "diverged":          diverged,
                        "proposed_qcet_id":  "",
                        "notes":             "",
                    })
                    break

    # Sort: by root, then lowercase form (so case variants of the same
    # variant are adjacent), then total_occ desc within case-group.
    rows_out.sort(key=lambda d: (d["root_word"], d["raw_lowercase"], -d["total_occ"]))

    fieldnames = [
        "root_word", "raw_lowercase", "raw_string",
        "occurrences_llm", "occurrences_human", "total_occ",
        "pass1_qcet_id", "pass1_qcet_name", "pass1_fit",
        "stage4_qcet_id", "stage4_qcet_name", "stage4_source",
        "diverged", "proposed_qcet_id", "notes",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows (occ >= {OCC_FLOOR}) to {OUT_CSV}")
    print()

    # Summary
    n_diverged = sum(1 for d in rows_out if d["diverged"])
    print(f"Pass-1 vs stage-4 divergence: {n_diverged} of {len(rows_out)} rows "
          f"({100*n_diverged/max(len(rows_out),1):.1f}%) had Pass-1 picking a "
          f"different target than the eventual force-merge winner.")
    print()
    print(f"{'root':14s}  {'rows':>5s}  {'div':>5s}  {'kept_occ':>8s}")
    by_root: dict[str, tuple[int, int, int]] = {}
    for d in rows_out:
        n, dv, occ = by_root.get(d["root_word"], (0, 0, 0))
        by_root[d["root_word"]] = (n + 1, dv + (1 if d["diverged"] else 0), occ + d["total_occ"])
    for root in POLYSEMOUS_ROOTS:
        if root in by_root:
            n, dv, occ = by_root[root]
            print(f"  {root:12s}  {n:5d}  {dv:5d}  {occ:8d}")


if __name__ == "__main__":
    main()
