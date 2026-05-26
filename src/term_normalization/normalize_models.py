#!/usr/bin/env python3
"""
Normalize model names to unified format: family_version_size_variant
"""
import csv
import re
from pathlib import Path
from collections import Counter


def normalize_model_name(model_name):
    """
    Normalize model name to: family_version_size_variant (all lowercase, underscore-separated)

    Examples:
        "Llama-3.1-8B-Instruct" → "llama_3.1_8b_instruct"
        "Llama2-7B" → "llama_2_7b"
        "GPT-4o-mini" → "gpt_4o_mini"
        "BART-large" → "bart_large"
        "ChatGPT (gpt-3.5-turbo)" → "gpt_3.5_turbo"
        "ChatGPT" → "chatgpt"

    Returns: normalized_name
    """
    original = model_name

    # Step 0: Handle parentheses
    # Case 1: Parentheses contain only size info like (13B), (14.7B) → append to model name
    # "Llama-2 (13B)" → "Llama-2 13B"
    # Case 2: Parentheses contain model identifier → use content inside
    # "ChatGPT (gpt-3.5-turbo)" → "gpt-3.5-turbo"
    paren_match = re.search(r'\(([^)]+)\)', model_name)
    if paren_match:
        paren_content = paren_match.group(1).strip()
        # Check if parentheses contain only size info (e.g., "13B", "14.7B", "7B")
        if re.match(r'^\d+\.?\d*[BM]$', paren_content, re.IGNORECASE):
            # Append size to model name (remove parentheses)
            model_name = re.sub(r'\s*\([^)]+\)', '', model_name) + ' ' + paren_content
        else:
            # Use content inside parentheses (it's a model identifier)
            model_name = paren_content

    # Step 1: Convert to lowercase first
    text = model_name.lower()

    # Step 2: Add space before version numbers that are directly attached
    # "llama2" → "llama 2", "qwen2.5" → "qwen 2.5", "gpt4" → "gpt 4"
    # But preserve "gpt4o" → "gpt4o" (4o is the version, not 4)
    # Pattern: word followed immediately by digit (with optional dots/decimals)
    text = re.sub(r'([a-z]+)(\d+(?:\.\d+)?)', r'\1 \2', text)

    # Step 3: Replace hyphens, slashes, underscores with spaces
    text = re.sub(r'[-_/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 4: Convert spaces to underscores
    text = text.replace(' ', '_')

    # Step 5: Clean up multiple underscores
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')

    return text


def process_models(input_file, output_dir):
    """Process models CSV and create normalized versions"""

    # Read models
    models = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append({
                'name': row['model'],
                'count': int(row['count'])
            })

    # Normalize and aggregate
    normalized_counts = Counter()
    mapping_rows = []
    merges = {}  # Track which normalized names have multiple originals

    for model in models:
        name = model['name']
        count = model['count']

        normalized = normalize_model_name(name)

        normalized_counts[normalized] += count
        mapping_rows.append((name, normalized, count))

        # Track merges
        if normalized not in merges:
            merges[normalized] = []
        merges[normalized].append((name, count))

    # For each normalized name, find the original name with the highest count
    canonical_names = {}
    for normalized, originals in merges.items():
        # Sort by count (descending), then alphabetically
        originals_sorted = sorted(originals, key=lambda x: (-x[1], x[0]))
        canonical_names[normalized] = originals_sorted[0][0]  # Most common original name

    # Save normalized models (using original name with highest count)
    normalized_file = output_dir / 'models_stats_normalized.csv'
    with open(normalized_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'count'])

        sorted_models = sorted(normalized_counts.items(), key=lambda x: (-x[1], x[0]))
        for normalized_name, count in sorted_models:
            canonical_name = canonical_names[normalized_name]
            writer.writerow([canonical_name, count])

    # Save model name mapping
    mapping_file = output_dir / 'models_normalization_mapping.csv'
    with open(mapping_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original', 'normalized', 'count'])
        for row in sorted(mapping_rows, key=lambda x: (x[1], -x[2], x[0])):
            writer.writerow(row)

    # Save merge details (only models that had multiple variants)
    merge_rows = []
    for normalized, originals in merges.items():
        if len(originals) > 1:
            total = sum(c for _, c in originals)
            variants_text = "; ".join(
                f"{o} ({c})" for o, c in sorted(originals, key=lambda x: (-x[1], x[0]))
            )
            merge_rows.append((normalized, total, len(originals), variants_text))

    merge_rows.sort(key=lambda x: (-x[1], -x[2], x[0]))

    merges_file = output_dir / 'models_normalization_merges.csv'
    with open(merges_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['normalized', 'total_count', 'num_variants', 'variants_with_counts'])
        writer.writerows(merge_rows)

    return {
        'total_original': len(models),
        'total_normalized': len(normalized_counts),
        'mappings': len(mapping_rows),
        'merges': len(merge_rows),
        'total_occurrences': sum(normalized_counts.values()),
    }


def main():
    script_dir = Path(__file__).parent.parent
    input_file = script_dir / 'metadata_unique_counts' / 'models' / 'models_stats.csv'
    output_dir = script_dir / 'metadata_unique_counts' / 'models'

    print("="*80)
    print("Model Name Normalization (Unified Format)")
    print("="*80)

    print(f"\nInput: {input_file}")

    stats = process_models(input_file, output_dir)

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Original unique models:         {stats['total_original']:,}")
    print(f"Normalized unique models:       {stats['total_normalized']:,}")
    print(f"Models merged:                  {stats['total_original'] - stats['total_normalized']:,}")
    print(f"Reduction:                      {(stats['total_original'] - stats['total_normalized']) / stats['total_original'] * 100:.2f}%")
    print(f"\nModels with multiple variants:  {stats['merges']:,}")
    print(f"Total model occurrences:        {stats['total_occurrences']:,}")

    print("\n" + "="*80)
    print("Files created:")
    print(f"  1. models_stats_normalized.csv - Normalized model counts")
    print(f"  2. models_normalization_mapping.csv - Original → Normalized mapping")
    print(f"  3. models_normalization_merges.csv - Models with multiple variants")
    print("="*80)


if __name__ == "__main__":
    main()
