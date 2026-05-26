#!/usr/bin/env python3
"""Polysemy diagnostic for candidate criterion root words.

Defines and applies a reproducible rule for which root words are
"high-impact polysemous" enough to warrant manual review by way of
`polysemous_overrides.csv`.

For each candidate root r:

    V(r)            = raw variants whose lowercased form contains r
    occ_per_target  = sum of (occurrences_llm + occurrences_human) per
                      Pass-1 QCET target across V(r)
    n_targets       = number of distinct QCET targets in occ_per_target
    top_share       = max(occ_per_target) / sum(occ_per_target)
    total_occ       = sum(occ_per_target)
    n80             = number of QCET targets needed to cover 80% of occ

Selection rule (ALL must hold):

    top_share < TOP_SHARE_THRESH        # genuine polysemy: dominant target
                                        # captures < 75% of occurrences
    AND
    total_occ >= TOTAL_OCC_THRESH       # impact: enough mentions that an
                                        # override moves headline counts

Defaults: TOP_SHARE_THRESH=0.80, TOTAL_OCC_THRESH=150.

We deliberately drop "n_targets >= 5" from the polysemy clause: roots whose
dominant target captures > 75% but with several singleton tail targets
(e.g., relevance, coherence) are not meaningfully polysemous — the singletons
are classifier noise, not genuine semantic alternatives.

Output:

    polysemy_diagnostic.csv
        root, n_variants, n_targets, n80, top_share, total_occ,
        qualifies, reason, top_targets

Run from `paper_code/`:

    python 05_criteria_normalization/polysemy_diagnostic.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

HERE       = Path(__file__).resolve().parent
STAGE1_CSV = HERE / "outputs" / "stage1_classifications.csv"
OUT_CSV    = HERE / "polysemy_diagnostic.csv"

# Selection-rule thresholds. Document changes here in the appendix paragraph.
TOP_SHARE_THRESH  = 0.80
TOTAL_OCC_THRESH  = 150

# Candidate root words to evaluate. Extend this list to test new candidates;
# the rule decides which qualify.
CANDIDATE_ROOTS: list[str] = [
    # WALKTHROUGH-flagged ambiguous criterion roots
    "accuracy", "faithful", "consistency", "correctness", "alignment",
    "quality", "completeness", "coverage",
    # additional high-frequency single-noun candidates worth probing
    "similarity", "fluency", "relevance", "coherence", "naturalness",
    "helpful", "truth", "factuality", "relevant", "specificity",
    "creativity", "diversity", "usefulness", "usability",
]


def load_stage1_rows():
    rows = []
    with STAGE1_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            rows.append({
                "raw":     (r["raw_string"] or "").strip().lower(),
                "qcet_id": r.get("qcet_id") or "",
                "occ":     int(r.get("occurrences_llm") or 0)
                         + int(r.get("occurrences_human") or 0),
            })
    return rows


def diag_for_root(root: str, rows: Iterable[dict]) -> dict | None:
    matches = [r for r in rows if root in r["raw"]]
    if not matches:
        return None
    by_target: Counter = Counter()
    for r in matches:
        # Pass-1 "no fit" rows are tracked under "(none)" so they count
        # as polysemy evidence (the classifier was unsure where to put them).
        tid = r["qcet_id"] or "(none)"
        by_target[tid] += r["occ"]
    total_occ = sum(by_target.values())
    if total_occ == 0:
        return None
    n_targets = len(by_target)
    top_occ   = max(by_target.values())
    top_share = top_occ / total_occ
    # n80 — minimum targets needed to cover 80% of occurrences
    cum = 0
    n80 = 0
    for v in sorted(by_target.values(), reverse=True):
        cum += v
        n80 += 1
        if cum >= 0.8 * total_occ:
            break
    return {
        "root":        root,
        "n_variants":  len(matches),
        "n_targets":   n_targets,
        "n80":         n80,
        "top_share":   round(top_share, 3),
        "total_occ":   total_occ,
        "top_targets": "; ".join(f"{tid}({n})" for tid, n in by_target.most_common(3)),
    }


def applies_rule(d: dict) -> tuple[bool, str]:
    poly_ok   = d["top_share"] < TOP_SHARE_THRESH
    impact_ok = d["total_occ"] >= TOTAL_OCC_THRESH
    if poly_ok and impact_ok:
        return True, f"polysemous (top_share<{TOP_SHARE_THRESH}) AND impactful (occ>={TOTAL_OCC_THRESH})"
    parts = []
    if not poly_ok:
        parts.append(f"top_share={d['top_share']:.2f} >= {TOP_SHARE_THRESH:.2f}")
    if not impact_ok:
        parts.append(f"total_occ={d['total_occ']} < {TOTAL_OCC_THRESH}")
    return False, "; ".join(parts)


def main() -> None:
    if not STAGE1_CSV.exists():
        raise SystemExit(f"Required input not found: {STAGE1_CSV}")

    rows = load_stage1_rows()
    diagnostics = []
    for root in CANDIDATE_ROOTS:
        d = diag_for_root(root, rows)
        if d is None:
            continue
        qualifies, reason = applies_rule(d)
        d["qualifies"] = "Y" if qualifies else "N"
        d["reason"]    = reason
        diagnostics.append(d)

    # Sort: qualifying roots first (most polysemous on top), then non-qualifying.
    diagnostics.sort(key=lambda d: (d["qualifies"] == "N",
                                     d["top_share"], -d["n_targets"]))

    fieldnames = ["root", "n_variants", "n_targets", "n80", "top_share",
                  "total_occ", "qualifies", "reason", "top_targets"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(diagnostics)

    print(f"Wrote {len(diagnostics)} candidate roots to {OUT_CSV}")
    print(f"\nThresholds: top_share<{TOP_SHARE_THRESH}, total_occ>={TOTAL_OCC_THRESH}\n")
    print(f"{'root':14}  {'n_var':>5}  {'n_tgt':>5}  {'top_share':>9}  "
          f"{'total_occ':>9}  qualifies   reason")
    print("-" * 110)
    for d in diagnostics:
        print(f"{d['root']:14}  {d['n_variants']:>5}  {d['n_targets']:>5}  "
              f"{d['top_share']:>9.2f}  {d['total_occ']:>9}  "
              f"{d['qualifies']:^9}   {d['reason']}")
    n_qual = sum(1 for d in diagnostics if d['qualifies'] == 'Y')
    print(f"\nQualifying roots ({n_qual}): "
          f"{', '.join(d['root'] for d in diagnostics if d['qualifies'] == 'Y')}")


if __name__ == "__main__":
    main()
