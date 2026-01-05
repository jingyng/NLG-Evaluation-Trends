#!/usr/bin/env python3
"""Normalize task stats with fresh, mapping-free rules."""

import csv
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path(__file__).parent.parent
INPUT = BASE / "metadata_unique_counts" / "tasks_stats.csv"
OUTPUT = BASE / "metadata_unique_counts" / "tasks_stats_normalized.csv"
MAP_OUTPUT = BASE / "metadata_unique_counts" / "tasks_normalization_mapping.csv"
FUZZY_MERGES_OUTPUT = BASE / "metadata_unique_counts" / "tasks_normalization_fuzzy_merges.csv"

# Lightweight aliases (no external mapping files)
ALIASES = {
    "qa": "Question Answering",
    "q a": "Question Answering",
    "mt": "Machine Translation",
    "nmt": "Machine Translation",
    "asr": "Automatic Speech Recognition",
    "ic": "Image Captioning",
    "ner": "Named Entity Recognition",
    "gpt": "Language Modeling",
    "lm": "Language Modeling",
}


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def canonical(text: str) -> str:
    text = strip_accents(text or "")
    text = text.lower()
    text = re.sub(r"[/_]", " ", text)
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"\\btask(s)?\\b", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    return text


def normalize_task(name: str) -> str | None:
    if not name:
        return None
    key = canonical(name)
    if not key:
        return None
    if key in ALIASES:
        return ALIASES[key]
    tokens = key.split()
    tokens = ["question answering" if t in {"qa", "q", "q a"} else t for t in tokens]
    key = " ".join(tokens)
    to_match = re.match(r"^(.*? to .*?)(?: generation)?$", key)
    if to_match:
        key = f"{to_match.group(1)} generation"
    return key.title()


def group_by_similarity(counts: Counter, threshold: float = 0.9):
    """Merge near-duplicates by similarity on canonical keys."""
    items = sorted(counts.items(), key=lambda x: -x[1])
    groups = []
    used = set()
    for i, (k, c) in enumerate(items):
        if k in used:
            continue
        bucket = [(k, c)]
        used.add(k)
        for j in range(i + 1, len(items)):
            k2, c2 = items[j]
            if k2 in used:
                continue
            if SequenceMatcher(None, k, k2).ratio() >= threshold:
                bucket.append((k2, c2))
                used.add(k2)
        groups.append(bucket)
    return groups


def main():
    raw_counts = Counter()
    mapping_rows = []
    with open(INPUT, newline="") as f:
        for row in csv.DictReader(f):
            orig = row.get("task", "")
            count = int(row.get("count", 0) or 0)
            norm = normalize_task(orig)
            if not norm:
                continue
            raw_counts[norm] += count
            mapping_rows.append((orig, norm, count))

    # Build fuzzy merge mapping: intermediate_normalized -> final_normalized
    fuzzy_merge_map = {}
    merged_counts = Counter()
    fuzzy_buckets = []
    for bucket in group_by_similarity(raw_counts, threshold=0.9):
        bucket = sorted(bucket, key=lambda x: (-x[1], x[0]))
        label = bucket[0][0]
        total = sum(c for _, c in bucket)
        merged_counts[label] += total

        # Map all variants in this bucket to the label
        for variant_name, variant_count in bucket:
            fuzzy_merge_map[variant_name] = label

        if len(bucket) > 1:
            fuzzy_buckets.append((label, total, bucket))

    # Apply fuzzy merge mapping to create final complete mapping
    final_mapping_rows = []
    for orig, intermediate_norm, count in mapping_rows:
        final_norm = fuzzy_merge_map.get(intermediate_norm, intermediate_norm)
        final_mapping_rows.append((orig, final_norm, count))

    rows = sorted(merged_counts.items(), key=lambda x: (-x[1], x[0]))

    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "count"])
        w.writerows(rows)

    with open(MAP_OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original", "normalized", "count"])
        w.writerows(final_mapping_rows)

    with open(FUZZY_MERGES_OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["merged_label", "total_count", "num_variants", "variants_with_counts"])
        for label, total, bucket in sorted(fuzzy_buckets, key=lambda x: (-x[1], -len(x[2]), x[0])):
            variants_text = "; ".join([f"{name} ({cnt})" for name, cnt in bucket])
            w.writerow([label, total, len(bucket), variants_text])

    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Wrote {len(final_mapping_rows)} mapping rows (with fuzzy merges applied) to {MAP_OUTPUT}")
    print(f"Wrote {len(fuzzy_buckets)} fuzzy merge groups to {FUZZY_MERGES_OUTPUT}")
    print("\nTop 15 normalized tasks:")
    for name, cnt in rows[:15]:
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    main()
