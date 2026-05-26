#!/usr/bin/env python3
"""
Regenerate metadata_unique_counts/criteria/*.csv files using the QCET
classification results from criteria_classifications.csv.

Produces (overwrites):
  metadata_unique_counts/criteria/
    llm_criteria_normalization_mapping.csv      (original,normalized,count,qcet_id)
    human_criteria_normalization_mapping.csv
    llm_criteria_normalization_merges.csv       (normalized,total_count,num_variants,variants_with_counts,qcet_id)
    human_criteria_normalization_merges.csv
    llm_criteria_stats_normalized.csv           (criterion,count,qcet_id)
    human_criteria_stats_normalized.csv

The "normalized" column carries the QCET full name (canonical, resolved across
chosen_id ↔ chosen_name conflicts).  Rows whose chosen_id == 'AUX-Other' use
the sentinel "__DROP__" so the JSON-normalization step can drop them.

QCET short labels for figure rendering live in `qcet_labels.py` (the
SHORT_LABELS dict) and are no longer exported to a separate CSV.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from qcet_labels import CANONICAL_NAME_OVERRIDES

HERE = Path(__file__).parent                  # src/qcet_normalization/
REPO_ROOT = HERE.parent.parent                # repo root
QCET_CSV = HERE / "outputs" / "criteria_classifications.csv"
POLY_OVERRIDES_CSV = HERE / "polysemous_overrides.csv"
OUT_DIR = REPO_ROOT / "metadata_unique_counts" / "criteria"

DROP_SENTINEL = "__DROP__"


def load_qcet_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with QCET_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_polysemous_overrides() -> Dict[str, Tuple[str, str]]:
    """Return {raw_lowercase: (new_qcet_id, new_qcet_name)}.

    The overrides file is produced by `extract_polysemous_overrides.py` from
    a hand-annotated review workbook. They take precedence over the QCET
    classifier's verdict, and apply to ALL case variants of the lowercased
    raw string (matching how normalize_merged_results.py looks up mappings).
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not POLY_OVERRIDES_CSV.exists():
        return out
    with POLY_OVERRIDES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("raw_lowercase") or "").strip().lower()
            new_id = (row.get("new_qcet_id") or "").strip()
            new_name = (row.get("new_qcet_name") or "").strip()
            if key and new_id and new_name:
                out[key] = (new_id, new_name)
    return out


def apply_polysemous_overrides(
    rows: List[Dict[str, str]],
    overrides: Dict[str, Tuple[str, str]],
) -> int:
    """Apply per-lowercase-key overrides in place.  Returns count of rows changed."""
    if not overrides:
        return 0
    n = 0
    for row in rows:
        key = row["raw_string"].strip().lower()
        if key in overrides:
            new_id, new_name = overrides[key]
            if row["chosen_id"] != new_id:
                row["chosen_id"]   = new_id
                row["chosen_name"] = new_name
                row["chosen_source"] = "polysemous_override"
                n += 1
    return n


def resolve_canonical_names(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Return {chosen_id: canonical_name}.  When the final classification has
    multiple chosen_name values for the same chosen_id, prefer the override,
    else the dominant name."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cid = row["chosen_id"]
        name = row["chosen_name"]
        n = int(row["occurrences_llm"] or 0) + int(row["occurrences_human"] or 0)
        counts[cid][name] += n
    canonical: Dict[str, str] = {}
    for cid, name_counts in counts.items():
        if cid in CANONICAL_NAME_OVERRIDES:
            canonical[cid] = CANONICAL_NAME_OVERRIDES[cid]
        else:
            canonical[cid] = max(name_counts.items(), key=lambda x: x[1])[0]
    return canonical


def build_mapping(rows: List[Dict[str, str]], canonical: Dict[str, str], occ_field: str) -> List[Tuple[str, str, int, str]]:
    """Return list of (original, normalized, count, qcet_id) sorted by count desc."""
    entries: List[Tuple[str, str, int, str]] = []
    for row in rows:
        n = int(row[occ_field] or 0)
        if n == 0:
            continue
        cid = row["chosen_id"]
        if cid == "AUX-Other":
            normalized = DROP_SENTINEL
        else:
            normalized = canonical[cid]
        entries.append((row["raw_string"], normalized, n, cid))
    entries.sort(key=lambda x: -x[2])
    return entries


def write_mapping(path: Path, entries: List[Tuple[str, str, int, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "normalized", "count", "qcet_id"])
        for orig, norm, n, cid in entries:
            w.writerow([orig, norm, n, cid])


def write_merges(path: Path, entries: List[Tuple[str, str, int, str]]) -> None:
    """Aggregate by normalized name, recording variants with their counts."""
    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    name_to_id: Dict[str, str] = {}
    for orig, norm, n, cid in entries:
        if norm == DROP_SENTINEL:
            continue
        grouped[norm].append((orig, n))
        name_to_id.setdefault(norm, cid)
    rows = []
    for norm, variants in grouped.items():
        variants.sort(key=lambda x: -x[1])
        total = sum(n for _, n in variants)
        rows.append((norm, total, len(variants), variants, name_to_id[norm]))
    rows.sort(key=lambda x: -x[1])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalized", "total_count", "num_variants", "variants_with_counts", "qcet_id"])
        for norm, total, k, variants, cid in rows:
            variants_str = "; ".join(f"{v} ({n})" for v, n in variants)
            w.writerow([norm, total, k, variants_str, cid])


def write_stats_normalized(path: Path, entries: List[Tuple[str, str, int, str]]) -> None:
    """Aggregate counts by normalized name (drops the sentinel entries)."""
    counts: Dict[str, int] = defaultdict(int)
    name_to_id: Dict[str, str] = {}
    for _orig, norm, n, cid in entries:
        if norm == DROP_SENTINEL:
            continue
        counts[norm] += n
        name_to_id.setdefault(norm, cid)
    rows = sorted(counts.items(), key=lambda x: -x[1])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "count", "qcet_id"])
        for name, n in rows:
            w.writerow([name, n, name_to_id[name]])


def main() -> None:
    rows = load_qcet_rows()
    overrides = load_polysemous_overrides()
    n_overridden = apply_polysemous_overrides(rows, overrides)
    canonical = resolve_canonical_names(rows)

    print(f"Loaded {len(rows)} classifications.")
    if overrides:
        print(f"Loaded {len(overrides)} polysemous overrides; "
              f"applied to {n_overridden} classification rows.")
    print(f"Canonical QCET names: {len(canonical)} (excluding AUX-Other).")

    for source, occ_field in [("llm", "occurrences_llm"), ("human", "occurrences_human")]:
        entries = build_mapping(rows, canonical, occ_field)
        n_dropped = sum(1 for e in entries if e[1] == DROP_SENTINEL)
        n_kept    = len(entries) - n_dropped
        print(f"  {source}: {len(entries)} mapping rows  ({n_kept} kept, {n_dropped} AUX-Other → __DROP__)")

        write_mapping(OUT_DIR / f"{source}_criteria_normalization_mapping.csv", entries)
        write_merges(OUT_DIR / f"{source}_criteria_normalization_merges.csv", entries)
        write_stats_normalized(OUT_DIR / f"{source}_criteria_stats_normalized.csv", entries)


if __name__ == "__main__":
    main()
