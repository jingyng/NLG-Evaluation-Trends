#!/usr/bin/env python3
"""Count distinct QCET constructs (QCET leaves + retained AUX nodes) actually
present in the top-30-tasks corpus, after the QCET-mapping pipeline.

The paper claims 116 distinct constructs across 3,334 filtered papers. This
script reproduces that number from `data/llm-merged-results-top30-tasks/`,
counting only criterion strings that successfully matched the QCET mapping
(unmapped passthrough variants from non-top-30 tasks are excluded; they
arise because the QCET classifier was run on a slightly larger raw set
during stage 1 and a small residual of raw strings remains unmapped in the
final corpus).

Run from `paper_code/`:

    python 05_criteria_normalization/count_constructs_in_corpus.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
PAPER_CODE = HERE.parent
DATA_DIR = PAPER_CODE / "data" / "llm-merged-results-top30-tasks"
LLM_MAP_CSV = PAPER_CODE / "metadata_unique_counts" / "llm_criteria_normalization_mapping.csv"
HUM_MAP_CSV = PAPER_CODE / "metadata_unique_counts" / "human_criteria_normalization_mapping.csv"


def _criteria(data: dict, key: str) -> Iterable[str]:
    block = data.get(key, {}) or {}
    crits = block.get("criteria") or []
    if isinstance(crits, str):
        return [crits]
    return [c for c in crits if isinstance(c, str) and c.strip()]


def _load_canonical_set(path: Path) -> set[str]:
    """Read the `normalized` column of a mapping CSV; skip __DROP__ rows."""
    out: set[str] = set()
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            norm = (r.get("normalized") or "").strip()
            if norm and norm != "__DROP__":
                out.add(norm)
    return out


def main() -> None:
    for p in (DATA_DIR, LLM_MAP_CSV, HUM_MAP_CSV):
        if not p.exists():
            raise SystemExit(f"Required path not found: {p}")

    canonical_llm = _load_canonical_set(LLM_MAP_CSV)
    canonical_hum = _load_canonical_set(HUM_MAP_CSV)
    canonical_any = canonical_llm | canonical_hum

    paths = sorted(DATA_DIR.rglob("*.json"))
    print(f"Reading {len(paths)} papers from {DATA_DIR.relative_to(PAPER_CODE)}")

    llm_constructs: Counter = Counter()
    hum_constructs: Counter = Counter()
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception as e:                        # pragma: no cover
            print(f"  skip {p.name}: {e}")
            continue
        for c in _criteria(data, "answer_3"):         # LaaJ
            llm_constructs[c] += 1
        for c in _criteria(data, "answer_4"):         # Human
            hum_constructs[c] += 1

    # Restrict to QCET-mapped constructs (drop unmapped passthrough strings).
    llm_q = {c for c in llm_constructs if c in canonical_any}
    hum_q = {c for c in hum_constructs if c in canonical_any}
    union_q = llm_q | hum_q

    print()
    print(f"All criteria strings present in JSONs:")
    print(f"  LaaJ unique  : {len(llm_constructs):4d}")
    print(f"  Human unique : {len(hum_constructs):4d}")
    print(f"  Union        : {len(set(llm_constructs) | set(hum_constructs)):4d}")
    print()
    print(f"Restricted to QCET-mapped constructs (drop unmapped passthroughs):")
    print(f"  LaaJ unique  : {len(llm_q):4d}")
    print(f"  Human unique : {len(hum_q):4d}")
    print(f"  Union        : {len(union_q):4d}    <-- the 112 in §4 for top-30-tasks")
    print()
    print(f"Full-corpus QCET-mapped constructs (from stats_normalized CSVs):")
    print(f"  LaaJ unique  : {len(canonical_llm):4d}")
    print(f"  Human unique : {len(canonical_hum):4d}")
    print(f"  Union        : {len(canonical_any):4d}    <-- the 116 in §4 for full corpus")


if __name__ == "__main__":
    main()
