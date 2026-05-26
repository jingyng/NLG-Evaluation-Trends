"""
sample_validation_set.py

Build a stratified validation sample on top of the override-corrected
Stage-4 frame `outputs/criteria_classifications_final.csv`
(produced by applying `polysemous_overrides.csv` on top of
`outputs/criteria_classifications.csv`). Pass `--final-csv outputs/
criteria_classifications.csv` to sample against the pre-override frame
instead.

The output CSV is annotation-ready: for each sampled variant we record the
predicted classification + LLM justification, and add empty `gold_*` columns
for the annotator to fill in. A separate metadata JSON records the stratum
recipe and random seed for reproducibility.

Defaults give 155 rows across 8 strata; sizes are CLI-tunable.

Calibration-pairs (gold pairs from the calibration step) and rows curated via manual_override are
excluded so the validation set is independent of prior curation steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_FINAL_CSV = HERE / "outputs" / "criteria_classifications_final.csv"
DEFAULT_CALIBRATION = HERE / "calibration_pairs.csv"
DEFAULT_OUT_CSV = HERE / "outputs" / "validation_sample.csv"
DEFAULT_META_JSON = HERE / "outputs" / "validation_sample_metadata.json"

DEFAULT_STRATUM_SIZES = {
    "A_qcet_strong_agree": 25,
    "B_qcet_strong_rescued": 20,
    "C_qcet_partial": 25,
    "D_qcet_disagreement": 20,
    "E_aux_specific": 10,   # reduced: only AUX-OverallQuality remains (32 variants)
    "F_aux_other": 25,
    "G_stage3_decisions": 10,
    "H_new_qcet_nodes": 20, # final-reclassification items mapped to one of the 8 new QCET nodes
}


def load_calibration_strings(p: Path) -> set[str]:
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("raw_string,"):
            continue
        out.add(line.split(",", 1)[0].strip().lower())
    return out


def occ_total(r: dict[str, str]) -> int:
    try:
        return int(r.get("occurrences_llm") or 0) + int(r.get("occurrences_human") or 0)
    except ValueError:
        return 0


def assign_stratum(r: dict[str, str]) -> str | None:
    cid = r.get("chosen_id") or ""
    ctype = r.get("chosen_type") or ""
    csource = r.get("chosen_source") or ""
    cfit = r.get("fit") or ""
    s1 = r.get("stage1_qcet_id") or ""
    if not cid:
        return None
    if csource == "manual_override":
        return None

    if csource.startswith("stage3_"):
        return "G_stage3_decisions"

    # Final-reclassification LLM items that landed on one of the 6 new QCET nodes
    NEW_QCET_NODES = {
        "QOC-c-2", "QOF-w-6",
        "QEC-c-3", "QEC-w-2", "QEF-w-10", "QEF-w-11",
    }
    if csource == "stage4_llm" and cid in NEW_QCET_NODES:
        return "H_new_qcet_nodes"

    if cid == "AUX-Other":
        return "F_aux_other"

    if ctype == "aux":
        return "E_aux_specific"  # now only AUX-OverallQuality

    # remaining cases are QCET
    if ctype == "qcet":
        if cfit == "strong":
            if s1 and s1 == cid:
                return "A_qcet_strong_agree"
            if not s1:
                return "B_qcet_strong_rescued"
            return "D_qcet_disagreement"
        if cfit == "partial":
            if s1 and s1 != cid:
                return "D_qcet_disagreement"
            return "C_qcet_partial"
    return None


def stratified_sample(
    rows_by_stratum: dict[str, list[dict[str, str]]],
    sizes: dict[str, int],
    rng: random.Random,
    top_frac: float = 0.4,
) -> dict[str, list[dict[str, str]]]:
    """For each stratum, pick `top_frac` of slots from the highest-occurrence
    items and the rest uniformly at random from the remaining items."""
    out: dict[str, list[dict[str, str]]] = {}
    for stratum, want in sizes.items():
        pool = rows_by_stratum.get(stratum, [])
        if want <= 0 or not pool:
            out[stratum] = []
            continue

        n_top = max(1, int(round(want * top_frac)))
        n_rand = want - n_top

        ranked = sorted(pool, key=lambda r: -occ_total(r))
        top = ranked[: min(n_top, len(ranked))]
        remaining = ranked[len(top):]

        if n_rand > 0 and remaining:
            random_pick = rng.sample(remaining, k=min(n_rand, len(remaining)))
        else:
            random_pick = []

        if len(top) + len(random_pick) < want and remaining:
            extra_pool = [r for r in remaining if r not in random_pick]
            need = want - (len(top) + len(random_pick))
            if extra_pool:
                random_pick.extend(rng.sample(extra_pool, k=min(need, len(extra_pool))))

        out[stratum] = top + random_pick
    return out


def build_annotation_row(stratum: str, r: dict[str, str]) -> dict[str, str]:
    return {
        "stratum": stratum,
        "raw_string": r.get("raw_string", ""),
        "occurrences_llm": r.get("occurrences_llm", "0"),
        "occurrences_human": r.get("occurrences_human", "0"),
        "predicted_id": r.get("chosen_id", ""),
        "predicted_name": r.get("chosen_name", ""),
        "predicted_type": r.get("chosen_type", ""),
        "predicted_source": r.get("chosen_source", ""),
        "predicted_fit": r.get("fit", ""),
        "stage1_qcet_id": r.get("stage1_qcet_id", ""),
        "stage1_qcet_fit": r.get("stage1_qcet_fit", ""),
        "stage2_cluster_id": r.get("stage2_cluster_id", ""),
        "construct": r.get("construct", ""),
        "justification": r.get("justification", ""),
        "gold_id": "",
        "gold_name": "",
        "gold_type": "",
        "verdict": "",
        "notes": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    ap.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-frac", type=float, default=0.4,
                    help="Fraction of each stratum filled from highest-occurrence variants; rest is uniform random.")
    for stratum, n in DEFAULT_STRATUM_SIZES.items():
        ap.add_argument(f"--n-{stratum}", type=int, default=n,
                        help=f"sample size for stratum {stratum} (default {n})")
    args = ap.parse_args()

    sizes = {s: getattr(args, f"n_{s}") for s in DEFAULT_STRATUM_SIZES}
    calibration_strings = load_calibration_strings(args.calibration_csv)

    with args.final_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Keep only one representative per case-insensitive group (highest occurrence).
    seen_lower: dict[str, dict[str, str]] = {}
    for r in rows:
        key = (r.get("raw_string") or "").strip().lower()
        if key not in seen_lower or occ_total(r) > occ_total(seen_lower[key]):
            seen_lower[key] = r
    rows = list(seen_lower.values())

    pool: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped = Counter()
    for r in rows:
        if (r.get("raw_string") or "").lower().strip() in calibration_strings:
            skipped["calibration_pair"] += 1
            continue
        stratum = assign_stratum(r)
        if stratum is None:
            skipped["unstratified"] += 1
            continue
        pool[stratum].append(r)

    rng = random.Random(args.seed)
    sampled = stratified_sample(pool, sizes, rng, top_frac=args.top_frac)

    # Guarantee at least one item per new QCET node in stratum H.
    # For nodes not yet represented, force-add the highest-occurrence variant.
    NEW_QCET_NODES = [
        "QOC-c-2", "QOF-w-6",
        "QEC-c-3", "QEC-w-2", "QEF-w-10", "QEF-w-11",
    ]
    h_sampled_ids = {r["chosen_id"] for r in sampled.get("H_new_qcet_nodes", [])}
    h_pool_by_node: dict[str, list[dict]] = defaultdict(list)
    for r in pool.get("H_new_qcet_nodes", []):
        h_pool_by_node[r["chosen_id"]].append(r)
    for node in NEW_QCET_NODES:
        if node not in h_sampled_ids and h_pool_by_node.get(node):
            best = max(h_pool_by_node[node], key=occ_total)
            sampled.setdefault("H_new_qcet_nodes", []).append(best)

    annotation_rows: list[dict[str, str]] = []
    actual_sizes: dict[str, int] = {}
    for stratum in DEFAULT_STRATUM_SIZES:
        items = sampled.get(stratum, [])
        actual_sizes[stratum] = len(items)
        for r in items:
            annotation_rows.append(build_annotation_row(stratum, r))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(annotation_rows[0].keys()) if annotation_rows else []
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotation_rows)

    metadata = {
        "seed": args.seed,
        "top_frac": args.top_frac,
        "source_csv": str(args.final_csv.relative_to(HERE)),
        "calibration_excluded_n": sum(
            1 for r in rows if (r.get("raw_string") or "").lower().strip() in calibration_strings
        ),
        "manual_override_excluded_n": sum(
            1 for r in rows if r.get("chosen_source") == "manual_override"
        ),
        "input_rows": len(rows),
        "skipped": dict(skipped),
        "stratum_definitions": {
            "A_qcet_strong_agree": "QCET strong fit, stage1_qcet_id == chosen_id (sanity floor).",
            "B_qcet_strong_rescued": "QCET strong fit, no initial-classification fit (final reclassification rescued).",
            "C_qcet_partial": "QCET partial fit (low-confidence final mapping).",
            "D_qcet_disagreement": "QCET id changed between initial and final classification (top movers).",
            "E_aux_specific": "AUX-OverallQuality (composite/holistic labels that do not map to a single criterion).",
            "F_aux_other": "AUX-Other residual bucket (extraction noise vs missed signal).",
            "G_stage3_decisions": "Cluster-level aux-category decisions (fold/aux/split/drop).",
            "H_new_qcet_nodes": "Final-reclassification items assigned to one of the 6 new QCET leaf nodes.",
        },
        "stratum_pool_sizes": {s: len(pool.get(s, [])) for s in DEFAULT_STRATUM_SIZES},
        "stratum_target_sizes": sizes,
        "stratum_actual_sizes": actual_sizes,
    }
    args.meta_json.write_text(json.dumps(metadata, indent=2))

    total = sum(actual_sizes.values())
    print(f"wrote {args.out_csv}  ({total} rows)")
    print(f"wrote {args.meta_json}")
    print()
    print("stratum sizes (target -> actual, pool):")
    for s in DEFAULT_STRATUM_SIZES:
        print(f"  {s:<28} {sizes[s]:>3} -> {actual_sizes[s]:>3}   (pool {len(pool.get(s, [])):>5})")
    print()
    print(f"excluded {metadata['calibration_excluded_n']} calibration pair rows; "
          f"{metadata['manual_override_excluded_n']} manual_override rows; "
          f"{skipped.get('unstratified', 0)} unclassifiable rows.")


if __name__ == "__main__":
    main()
