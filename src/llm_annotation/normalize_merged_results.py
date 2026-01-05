#!/usr/bin/env python3
"""
Normalize merged paper JSON files using the prepared mapping CSVs.

For each paper in ``llm-merged-results`` it creates a normalized copy in
``llm-merged-results-normalized`` (by default), preserving the folder layout.
Only the metadata fields with mappings are touched; everything else is passed
through unchanged.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Set


MAPPING_FILES = {
    "tasks": "tasks_normalization_mapping.csv",
    "datasets": "datasets_normalization_mapping.csv",
    "models": "models_normalization_mapping.csv",
    "languages": "languages_normalization_mapping.csv",
    "automatic_metrics": "automatic_metrics_normalization_mapping.csv",
    "llm_criteria": "llm_criteria_normalization_mapping.csv",
    "human_criteria": "human_criteria_normalization_mapping.csv",
}


def load_mapping(path: Path) -> Dict[str, str]:
    """Load a mapping CSV into a case-insensitive lookup."""
    mapping: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "original" not in reader.fieldnames or "normalized" not in reader.fieldnames:
            raise ValueError(f"Mapping file {path} is missing required columns.")
        for row in reader:
            orig = (row.get("original") or "").strip()
            norm = (row.get("normalized") or "").strip()
            if not orig or not norm:
                continue
            mapping[orig.lower()] = norm
            # Also allow normalized values to be idempotent.
            mapping.setdefault(norm.lower(), norm)
    return mapping


def _strip_other_prefix(text: str) -> str:
    """Remove leading 'Other:' prefix (case-insensitive) if present."""
    lowered = text.lower()
    if lowered.startswith("other:"):
        return text.split(":", 1)[1].strip()
    return text


def normalize_list(
    values: Iterable[str] | None,
    mapping: Mapping[str, str],
    missing: Set[str],
    preprocessor=None,
) -> List[str]:
    """Normalize a list of strings using the provided mapping."""
    if values is None:
        return []
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)

    normalized: List[str] = []
    seen: Set[str] = set()

    for raw in items:
        if raw is None:
            continue
        text = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if preprocessor:
            text = preprocessor(text)
        if not text:
            continue
        norm = mapping.get(text.lower())
        if norm is None:
            missing.add(text)
            norm = text
        if norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    return normalized


def normalize_record(
    data: MutableMapping[str, object],
    mappings: Mapping[str, Mapping[str, str]],
    missing: Mapping[str, Set[str]],
) -> None:
    """Apply normalizations in-place to the relevant answer sections."""
    answer_1 = data.get("answer_1")
    if isinstance(answer_1, dict):
        for field, map_key in [
            ("tasks", "tasks"),
            ("datasets", "datasets"),
            ("languages", "languages"),
            ("models", "models"),
        ]:
            if field in answer_1:
                answer_1[field] = normalize_list(
                    answer_1.get(field),
                    mappings[map_key],
                    missing[map_key],
                    preprocessor=_strip_other_prefix if field == "tasks" else None,
                )

    answer_2 = data.get("answer_2")
    if isinstance(answer_2, dict) and "automatic_metrics" in answer_2:
        answer_2["automatic_metrics"] = normalize_list(
            answer_2.get("automatic_metrics"),
            mappings["automatic_metrics"],
            missing["automatic_metrics"],
        )

    answer_3 = data.get("answer_3")
    if isinstance(answer_3, dict):
        if "criteria" in answer_3:
            answer_3["criteria"] = normalize_list(
                answer_3.get("criteria"), mappings["llm_criteria"], missing["llm_criteria"]
            )
        if "models" in answer_3:
            answer_3["models"] = normalize_list(
                answer_3.get("models"), mappings["models"], missing["models"]
            )

    answer_4 = data.get("answer_4")
    if isinstance(answer_4, dict) and "criteria" in answer_4:
        answer_4["criteria"] = normalize_list(
            answer_4.get("criteria"), mappings["human_criteria"], missing["human_criteria"]
        )


def gather_mappings(base_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load all mapping CSVs into memory."""
    mappings: Dict[str, Dict[str, str]] = {}
    for key, filename in MAPPING_FILES.items():
        path = base_dir / filename
        mappings[key] = load_mapping(path)
    return mappings


def run(input_dir: Path, output_dir: Path, mappings_dir: Path) -> None:
    mappings = gather_mappings(mappings_dir)
    missing: Dict[str, Set[str]] = defaultdict(set)

    json_files = list(input_dir.rglob("*.json"))
    for idx, path in enumerate(json_files, start=1):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        normalize_record(data, mappings, missing)

        out_path = output_dir / path.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if idx % 500 == 0:
            print(f"Processed {idx} / {len(json_files)} files...")

    print(f"\nWrote {len(json_files)} normalized files to {output_dir}")
    for key in sorted(missing):
        if missing[key]:
            print(f"- Unmapped {key}: {len(missing[key])} unique values (left as-is)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize merged paper metadata using mapping CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("llm-merged-results"),
        help="Directory containing merged paper JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm-merged-results-normalized"),
        help="Where to write normalized JSON files.",
    )
    parser.add_argument(
        "--mappings-dir",
        type=Path,
        default=Path("metadata_unique_counts"),
        help="Directory containing *_normalization_mapping.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent.parent

    input_dir = args.input_dir if args.input_dir.is_absolute() else project_root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    mappings_dir = args.mappings_dir if args.mappings_dir.is_absolute() else project_root / args.mappings_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run(input_dir, output_dir, mappings_dir)


if __name__ == "__main__":
    main()
