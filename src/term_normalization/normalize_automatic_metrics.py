#!/usr/bin/env python3
"""Normalize automatic evaluation metrics with aliases + fuzzy grouping."""

import csv
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


BASE = Path(__file__).parent.parent
INPUT = BASE / "metadata_unique_counts" / "automatic_metrics_stats.csv"
OUTPUT = BASE / "metadata_unique_counts" / "automatic_metrics_stats_normalized.csv"
MAP_OUTPUT = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_mapping.csv"
MERGES_OUTPUT = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"
FUZZY_MERGES_OUTPUT = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_fuzzy_merges.csv"


def canonical(text: str) -> str:
    """Lowercase, keep alnum/+/@, normalize separators to space, strip k-values."""
    text = text.lower().strip()
    text = re.sub(r"[_/|-]", " ", text)
    text = re.sub(r"[^a-z0-9+@]+", " ", text)
    text = " ".join(text.split())

    # Strip @k/@1/@5/@10 etc. and suffixes to group variants
    # Examples: "pass@1" -> "pass", "recall@k" -> "recall", "bleu 4" -> "bleu"
    text = re.sub(r"@\d+|@k\b", "", text)  # Remove @1, @5, @k, etc.
    text = re.sub(r"\s+[0-9a-z]$", "", text)  # Remove trailing single char/digit like "rouge l", "rouge 1", "f1 5"
    text = re.sub(r"\s+\d+$", "", text)    # Remove trailing numbers like "rouge 10"

    # Strip common metric modifiers (as both prefix and suffix)
    # Examples: "f1 score" -> "f1", "macro f1" -> "f1", "f1 macro" -> "f1"
    modifiers = r"\b(score|metric|avg|average|mean|macro|micro|weighted|binary)\b"
    text = re.sub(modifiers, "", text)
    text = " ".join(text.split())  # Clean up extra spaces

    return text


def normalize_metric(name: str) -> str | None:
    if not name or not name.strip():
        return None
    key = canonical(name)
    if not key:
        return None
    if re.match(r"^bleu(\\b|\\d|\\s|@|-|$)", key):
        return "BLEU"
    if re.match(r"^rouge(\\b|\\d|\\s|@|-|$)", key):
        return "ROUGE"

    # Common abbreviations → full names
    if key == "em":
        return "EXACT MATCH"

    # Fallback: uppercase tokens joined by space, keep '@' and '+' within tokens
    tokens = key.split()
    normalized = " ".join(t.upper() for t in tokens)
    return normalized


def is_distinct_metric_pair(k1: str, k2: str) -> bool:
    """Check if two metrics are actually distinct despite surface similarity."""
    lk1, lk2 = k1.lower(), k2.lower()

    # BLEU vs BLEURT/BLEUR/BLERU - these are different metrics
    bleu_variants = ["bleurt", "bleur", "bleru"]
    if "bleu" in lk1 or "bleu" in lk2:
        for variant in bleu_variants:
            if (variant in lk1 and "bleu" == lk2.replace(" ", "")) or \
               (variant in lk2 and "bleu" == lk1.replace(" ", "")):
                return True

    # BERTScore vs BERT - BERTScore is a specific metric, plain BERT is different
    if ("bertscore" in lk1 and lk2 == "bert") or \
       ("bertscore" in lk2 and lk1 == "bert"):
        return True

    # Add more exclusions as needed
    return False


def group_by_similarity(counts: Counter, threshold: float = 0.9):
    """Merge near-duplicates by similarity on normalized labels."""
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

            # Quick filter: first 2 chars must match (more lenient than 3)
            if len(k) >= 2 and len(k2) >= 2:
                if k[:2].lower() != k2[:2].lower():
                    continue

            # Remove strict length filter - let similarity score decide
            # (but keep a very loose filter for performance)
            if abs(len(k) - len(k2)) > 15:
                continue

            # Check for known distinct metrics
            if is_distinct_metric_pair(k, k2):
                continue

            # Calculate similarity
            sim = SequenceMatcher(None, k.lower(), k2.lower()).ratio()
            if sim >= threshold:
                bucket.append((k2, c2))
                used.add(k2)

        groups.append(bucket)
    return groups


def main():
    raw_counts = Counter()
    mapping_rows = []
    norm_to_orig = {}
    with open(INPUT, newline="") as f:
        for row in csv.DictReader(f):
            norm = normalize_metric(row.get("metric", ""))
            if norm:
                count = int(row.get("count", 0) or 0)
                raw_counts[norm] += count
                mapping_rows.append((row.get("metric", "").strip(), norm, count))
                norm_to_orig.setdefault(norm, Counter()).update({row.get("metric", "").strip(): count})

    # Build fuzzy merge mapping
    fuzzy_merge_map = {}
    canonical_counts = Counter()
    fuzzy_buckets = []
    sim_groups = group_by_similarity(raw_counts, threshold=0.9)
    for bucket in sim_groups:
        bucket = sorted(bucket, key=lambda x: (-x[1], x[0]))
        label = bucket[0][0]
        total = sum(c for _, c in bucket)
        canonical_counts[label] += total

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

    rows = sorted(canonical_counts.items(), key=lambda x: (-x[1], x[0]))
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count"])
        writer.writerows(rows)

    with open(MAP_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "normalized", "count"])
        writer.writerows(final_mapping_rows)

    merges_rows = []
    for norm, ctr in norm_to_orig.items():
        if len(ctr) <= 1:
            continue
        total = sum(ctr.values())
        variants_text = "; ".join(
            f"{o} ({c})" for o, c in sorted(ctr.items(), key=lambda x: (-x[1], x[0]))
        )
        merges_rows.append((norm, total, len(ctr), variants_text))
    merges_rows.sort(key=lambda x: (-x[1], -x[2], x[0]))

    with open(MERGES_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized", "total_count", "num_variants", "variants_with_counts"])
        writer.writerows(merges_rows)

    with open(FUZZY_MERGES_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["merged_label", "total_count", "num_variants", "variants_with_counts"])
        for label, total, bucket in sorted(fuzzy_buckets, key=lambda x: (-x[1], -len(x[2]), x[0])):
            variants_text = "; ".join([f"{name} ({cnt})" for name, cnt in bucket])
            writer.writerow([label, total, len(bucket), variants_text])

    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Wrote {len(final_mapping_rows)} mapping rows (with fuzzy merges applied) to {MAP_OUTPUT}")
    print(f"Wrote {len(merges_rows)} exact merge groups to {MERGES_OUTPUT}")
    print(f"Wrote {len(fuzzy_buckets)} fuzzy merge groups to {FUZZY_MERGES_OUTPUT}")
    print("\nTop 15 normalized metrics:")
    for name, cnt in rows[:15]:
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    main()
