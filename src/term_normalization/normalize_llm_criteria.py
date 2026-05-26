#!/usr/bin/env python3
"""Normalize LLM criteria stats with canonicalization + fuzzy grouping."""

import csv
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


BASE = Path(__file__).parent.parent
SUBDIR = BASE / "metadata_unique_counts" / "criteria"
INPUT = SUBDIR / "llm_criteria_stats.csv"
OUTPUT = SUBDIR / "llm_criteria_stats_normalized.csv"
MAP_OUTPUT = SUBDIR / "llm_criteria_normalization_mapping.csv"
MERGES_OUTPUT = SUBDIR / "llm_criteria_normalization_merges.csv"
FUZZY_MERGES_OUTPUT = SUBDIR / "llm_criteria_normalization_fuzzy_merges.csv"


def canonical(text: str) -> str:
    """Lowercase, strip punctuation, and standardize separators."""
    text = text.lower().strip()
    text = re.sub(r"[_/]", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(level of\b|\bscore\b)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


QUALITY_NOUNS = {
    # Core qualities
    "accuracy",
    "adequacy",
    "bias",
    "clarity",
    "coherence",
    "completeness",
    "conciseness",
    "consistency",
    "correctness",
    "coverage",
    "creativity",
    "depth",
    "diversity",
    "empathy",
    "fairness",
    "faithfulness",
    "fluency",
    "factuality",
    "grammaticality",
    "harmfulness",
    "harmlessness",
    "helpfulness",
    "informativeness",
    "naturalness",
    "novelty",
    "originality",
    "preference",
    "quality",
    "readability",
    "relevance",
    "safety",
    "similarity",
    "specificity",
    "toxicity",
    "truthfulness",
}

MANUAL_MAP = {
    "safe": "safety",
    "unsafe": "safety",
    "broad": "breadth",
    "deep": "depth",
    "true": "truth",
    "strong": "strength",
    "long": "length",
    "wide": "width",
    "high": "height",
    "poor": "quality",
    "good": "quality",
    "bad": "quality",
    "harm": "harmfulness",
    "harmless": "harmlessness",
    "honest": "honesty",
    "ethical": "ethics",
    "legal": "legality",
    "fair": "fairness",
    "bias": "bias",
    "biased": "bias",
    "unbiased": "bias",
    "toxic": "toxicity",
    "funny": "humor",
    "creative": "creativity",
    "original": "originality",
    "complex": "complexity",
    "simple": "simplicity",
    "clear": "clarity",
    "fluent": "fluency",
    "consistent": "consistency",
    "complete": "completeness",
    "valid": "validity",
    "reliable": "reliability",
    "factual": "factuality",
    "precise": "precision",
    "concise": "conciseness",
    "polite": "politeness",
    "formal": "formality",
    "neutral": "neutrality",
    "objective": "objectivity",
    "subjective": "subjectivity",
    "informative": "informativeness",
    "relevant": "relevance",
    "irrelevant": "relevance",
    "helpful": "helpfulness",
    "unhelpful": "helpfulness",
    "useless": "utility",
    "useful": "utility",
    "correct": "correctness",
    "incorrect": "correctness",
    "accurate": "accuracy",
    "inaccurate": "accuracy",
    "coherent": "coherence",
    "incoherent": "coherence",
    "logical": "logic",
    "illogical": "logic",
    "rational": "rationality",
    "sensible": "sensibility",
    "specific": "specificity",
    "vague": "vagueness",
    "detailed": "detail",
    "natural": "naturalness",
    "unnatural": "naturalness",
    "human": "humanness",
    "robotic": "roboticness",
    "engaging": "engagement",
    "boring": "boredom",
    "interesting": "interest",
    "repetitive": "repetitiveness",
    "redundant": "redundancy",
    "hallucination": "hallucination",
    "hallucinations": "hallucination",
    "emotional": "emotion",
    "grammar": "grammaticality",
    "grammatical": "grammaticality",
    "ungrammatical": "grammaticality",
    "spelled correctly": "spelling",
    "typo": "typos",
}

MANUAL_TARGETS = set(MANUAL_MAP.values())

STOPWORDS = {
    "of",
    "in",
    "on",
    "for",
    "to",
    "by",
    "with",
    "and",
    "or",
    "per",
    "overall",
    "total",
    "level",
    "score",
    "all",
    "any",
    "the",
    "a",
    "an",
    "at",
    "as",
    "within",
    "between",
    "across",
    "over",
    "under",
    "into",
    "from",
    "following",
}


def pick_quality_noun(tokens: list[str]) -> str | None:
    """Choose a single-word noun describing the quality."""
    filtered = [t for t in tokens if t and t not in STOPWORDS]
    if not filtered:
        return None

    # 1) Prefer known quality nouns in order of appearance.
    for t in filtered:
        if t in QUALITY_NOUNS:
            return t

    # 2) Prefer nouns by suffix patterns.
    suffixes = ("ness", "ity", "tion", "sion", "ment", "ence", "ance", "acy", "ship", "acy", "acy", "ure", "al")
    for t in filtered:
        if t.endswith(suffixes):
            return t

    # 3) Fall back to the last token (often the head noun in English compounds).
    return filtered[-1]


def normalize_raw(criterion: str) -> str | None:
    if not criterion or not criterion.strip():
        return None
    raw = criterion.strip()
    key = canonical(raw)
    if key in MANUAL_MAP:
        return MANUAL_MAP[key].title()
    tokens = key.split()
    for t in tokens:
        if t in MANUAL_MAP:
            return MANUAL_MAP[t].title()
    single = pick_quality_noun(tokens)
    if not single:
        return None
    single = MANUAL_MAP.get(single, single)
    return single.title()


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
            sim = SequenceMatcher(None, k, k2).ratio()
            if sim >= threshold:
                bucket.append((k2, c2))
                used.add(k2)
        groups.append(bucket)
    return groups


def print_summary(counts: Counter):
    print("\nTop 15 normalized criteria:")
    for name, cnt in counts.most_common(15):
        print(f"  {name}: {cnt}")

    suspect = []
    for name, cnt in counts.items():
        low = name.lower()
        if low not in QUALITY_NOUNS and low not in MANUAL_TARGETS:
            suspect.append((name, cnt))
    suspect = sorted(suspect, key=lambda x: -x[1])[:15]
    if suspect:
        print("\nTop 15 non-quality/alias outputs (check manually):")
        for name, cnt in suspect:
            print(f"  {name}: {cnt}")


def main():
    raw_counts = Counter()
    mapping_rows = []
    norm_to_orig = {}
    with open(INPUT) as f:
        for row in csv.DictReader(f):
            norm = normalize_raw(row.get("criterion", ""))
            if norm:
                count = int(row.get("count", 0) or 0)
                raw_counts[norm] += count
                mapping_rows.append((row.get("criterion", "").strip(), norm, count))
                norm_to_orig.setdefault(norm, Counter()).update({row.get("criterion", "").strip(): count})

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
        writer.writerow(["criterion", "count"])
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
    print(f"Wrote {len(merges_rows)} merge groups to {MERGES_OUTPUT}")
    print(f"Wrote {len(fuzzy_buckets)} fuzzy merge groups to {FUZZY_MERGES_OUTPUT}")
    print_summary(canonical_counts)


if __name__ == "__main__":
    main()
