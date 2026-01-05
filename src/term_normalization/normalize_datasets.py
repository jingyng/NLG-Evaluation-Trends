#!/usr/bin/env python3
"""Normalize dataset stats with aliases + cautious fuzzy grouping."""

import csv
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


BASE = Path(__file__).parent.parent
INPUT = BASE / "metadata_unique_counts" / "datasets_stats.csv"
OUTPUT = BASE / "metadata_unique_counts" / "datasets_stats_normalized.csv"
MAP_OUTPUT = BASE / "metadata_unique_counts" / "datasets_normalization_mapping.csv"
MERGES_OUTPUT = BASE / "metadata_unique_counts" / "datasets_normalization_merges.csv"
FUZZY_MERGES_OUTPUT = BASE / "metadata_unique_counts" / "datasets_normalization_fuzzy_merges.csv"


ALIASES = {
    "cnn dailymail": "CNN/DailyMail",
    "cnn/dailymail": "CNN/DailyMail",
    "cnn-dailymail": "CNN/DailyMail",
    "cnn daily mail": "CNN/DailyMail",
    "cnn dm": "CNN/DailyMail",
    "squad": "SQuAD",
    "squad1": "SQuAD",
    "squad 1": "SQuAD",
    "squadv2": "SQuAD2",
    "squad2": "SQuAD2",
    "squad 2": "SQuAD2",
    "squad v2": "SQuAD2",
    "multinli": "MultiNLI",
    "multi nli": "MultiNLI",
    "gsm8k": "GSM8K",
    "gsm-8k": "GSM8K",
    "gsm 8k": "GSM8K",
    "mbpp": "MBPP",
    "hellaswag": "HellaSwag",
    "svamp": "SVAMP",
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "humaneval": "HumanEval",
    "mt bench": "MT-Bench",
    "mmlu": "MMLU",
    "truthfulqa": "TruthfulQA",
    "winogrande": "WinoGrande",
    "commonsenseqa": "CommonsenseQA",
    "piqa": "PIQA",
    "boolq": "BoolQ",
}


def normalize_wmt_language_pair(lang_pair: str) -> str:
    """
    Normalize language pair codes to standard 2-letter format.

    Examples:
        "English-German" → "en-de"
        "En-De" → "en-de"
        "EN→DE" → "en-de"
        "German-English" → "de-en"
    """
    # Language code mappings
    lang_map = {
        "english": "en",
        "german": "de",
        "french": "fr",
        "romanian": "ro",
        "chinese": "zh",
        "czech": "cs",
        "russian": "ru",
        "turkish": "tr",
        "finnish": "fi",
        "spanish": "es",
        "italian": "it",
    }

    # Remove arrows and normalize separators to hyphen
    lang_pair = lang_pair.lower().strip()
    lang_pair = re.sub(r'[→↔]', '-', lang_pair)
    lang_pair = re.sub(r'\s+', '-', lang_pair)

    # Split on hyphen
    parts = lang_pair.split('-')
    if len(parts) != 2:
        return lang_pair

    src, tgt = parts[0].strip(), parts[1].strip()

    # Map full names to codes
    src = lang_map.get(src, src)
    tgt = lang_map.get(tgt, tgt)

    return f"{src}-{tgt}"


def normalize_wmt_dataset(name: str) -> str:
    """
    Normalize WMT dataset names with year and language pair.

    Examples:
        "WMT'14 English-German" → "WMT14 En-De"
        "WMT 2014 En-De" → "WMT14 En-De"
        "WMT14 DE-EN" → "WMT14 De-En"
    """
    # Check if it's a WMT dataset (don't use \b because WMT14 has no word boundary)
    if not re.search(r'wmt', name, re.IGNORECASE):
        return name

    # Extract year: WMT14, WMT'14, WMT 2014, WMT-2014, etc.
    year_match = re.search(r"wmt['\s-]*(\d{2,4})", name, re.IGNORECASE)
    year = ""
    if year_match:
        year_num = year_match.group(1)
        # Convert 2-digit to 4-digit if needed
        if len(year_num) == 2:
            year = year_num
        elif len(year_num) == 4:
            year = year_num[-2:]  # Take last 2 digits
        else:
            year = year_num

    # Extract language pair
    lang_pair = ""
    # Pattern: language names or codes separated by hyphen/arrow/space
    lang_pattern = r'\b(english|german|french|romanian|chinese|czech|russian|turkish|finnish|en|de|fr|ro|zh|cs|ru|tr|fi)\s*[-→↔\s]+\s*(english|german|french|romanian|chinese|czech|russian|turkish|finnish|en|de|fr|ro|zh|cs|ru|tr|fi)\b'
    lang_match = re.search(lang_pattern, name, re.IGNORECASE)

    if lang_match:
        lang_pair = normalize_wmt_language_pair(lang_match.group(0))

    # Construct normalized name (only year, ignore language pairs)
    if year:
        return f"WMT{year}"
    else:
        return "WMT"


def canonical(text: str) -> str:
    """Normalize to lowercase, replace separators with spaces."""
    text = text.lower().strip()
    # Fixed regex: properly escape the hyphen
    text = re.sub(r"[_/|-]", " ", text)
    text = re.sub(r"[^a-z0-9+ ]+", " ", text)

    # Strip version suffixes like "+" or "++"
    text = re.sub(r'\+{1,2}$', '', text.strip())

    return " ".join(text.split())


def normalize_name(name: str) -> str | None:
    """Normalize dataset name using aliases and standard casing."""
    if not name or not name.strip():
        return None

    # Special handling for WMT datasets
    if re.search(r'wmt', name, re.IGNORECASE):
        return normalize_wmt_dataset(name)

    key = canonical(name)
    if key in ALIASES:
        return ALIASES[key]
    # Preserve casing for known tokens: keep alnum tokens, uppercase if all letters, keep numbers.
    tokens = key.split()
    normalized = " ".join(t.upper() if t.isalpha() else t for t in tokens)
    return normalized


def has_different_numbers(a: str, b: str) -> bool:
    """Check if two strings have different numbers."""
    nums_a = re.findall(r"\d+", a)
    nums_b = re.findall(r"\d+", b)
    return nums_a != nums_b


def group_by_similarity(counts: Counter, threshold: float = 0.9):
    """Merge near-duplicates by similarity on normalized labels with number check."""
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

            # Improved filters (less restrictive than before)
            # Prefix filter: first 2 chars (more lenient)
            if len(k) >= 2 and len(k2) >= 2:
                if k[:2].lower() != k2[:2].lower():
                    continue

            # Length filter: more lenient (allow up to 15 char difference)
            if abs(len(k) - len(k2)) > 15:
                continue

            # Don't merge if they have different numbers (e.g., SQuAD vs SQuAD2)
            if has_different_numbers(k, k2):
                continue

            # Calculate similarity
            if SequenceMatcher(None, k.lower(), k2.lower()).ratio() >= threshold:
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
            norm = normalize_name(row.get("dataset", ""))
            if norm:
                count = int(row.get("count", 0) or 0)
                raw_counts[norm] += count
                mapping_rows.append((row.get("dataset", "").strip(), norm, count))
                norm_to_orig.setdefault(norm, Counter()).update({row.get("dataset", "").strip(): count})

    # Build fuzzy merge mapping
    fuzzy_merge_map = {}
    canonical_counts = Counter()
    fuzzy_buckets = []
    canonical_to_originals = {}  # Track originals for each canonical form

    for bucket in group_by_similarity(raw_counts, threshold=0.9):
        bucket = sorted(bucket, key=lambda x: (-x[1], x[0]))
        label = bucket[0][0]  # Most common normalized form
        total = sum(c for _, c in bucket)
        canonical_counts[label] += total

        # Map all variants in this bucket to the label
        for variant_name, variant_count in bucket:
            fuzzy_merge_map[variant_name] = label

        # Collect all original names for this canonical form
        all_originals = []
        for norm_name, _ in bucket:
            if norm_name in norm_to_orig:
                all_originals.extend(norm_to_orig[norm_name].items())
        canonical_to_originals[label] = all_originals

        if len(bucket) > 1:
            fuzzy_buckets.append((label, total, bucket))

    # For each canonical form, find the original name with highest count
    # BUT: for WMT datasets, use the normalized form (e.g., "WMT14" not "WMT14 English-German")
    canonical_names = {}
    for canonical_label, originals in canonical_to_originals.items():
        # For WMT datasets with years, always use normalized form like "WMT14"
        if re.search(r'^WMT\d', canonical_label):
            canonical_names[canonical_label] = canonical_label
        elif originals:
            # For other datasets, use the most common original variant
            # Sort by count (descending), then alphabetically
            originals_sorted = sorted(originals, key=lambda x: (-x[1], x[0]))
            canonical_names[canonical_label] = originals_sorted[0][0]
        else:
            canonical_names[canonical_label] = canonical_label

    # Apply fuzzy merge mapping to create final complete mapping
    final_mapping_rows = []
    for orig, intermediate_norm, count in mapping_rows:
        final_norm = fuzzy_merge_map.get(intermediate_norm, intermediate_norm)
        final_mapping_rows.append((orig, final_norm, count))

    # Save normalized datasets (using original name with highest count)
    rows = sorted(canonical_counts.items(), key=lambda x: (-x[1], x[0]))
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "count"])
        for normalized_name, count in rows:
            canonical_name = canonical_names.get(normalized_name, normalized_name)
            writer.writerow([canonical_name, count])

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
    print(f"Wrote {len(mapping_rows)} mapping rows to {MAP_OUTPUT}")
    print(f"Wrote {len(merges_rows)} exact merge groups to {MERGES_OUTPUT}")
    print(f"Wrote {len(fuzzy_buckets)} fuzzy merge groups to {FUZZY_MERGES_OUTPUT}")
    print("\nTop 15 normalized datasets:")
    for normalized_name, cnt in rows[:15]:
        canonical_name = canonical_names.get(normalized_name, normalized_name)
        print(f"  {canonical_name}: {cnt}")


if __name__ == "__main__":
    main()
