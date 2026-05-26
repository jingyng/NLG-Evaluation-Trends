import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from data_loader import normalize_criteria
#!/usr/bin/env python3
"""
Alternative visualizations for term counts by year
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def spelling_unify(s: str) -> str:
    """Surface-level normalization applied uniformly to criteria strings:
    NFKD-fold accents, lowercase, replace structural separators (-_/\\&) with
    space, drop remaining non-word characters, collapse whitespace.
    Phrases stay phrases (no head-word extraction, no fuzzy matching).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[\-_/\\&]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


PAPER_ROOT = Path(__file__).parent.parent   # paper_code/
BASE = PAPER_ROOT / "data"
DEFAULT_INPUT = BASE / "llm-merged-results-normalized"
# Criteria are read from the RAW (pre-normalization) corpus so the figure
# shows actual variant proliferation, not the QCET-bucket count (which would
# be capped by the taxonomy and hide the proliferation pattern we want to
# illustrate).  Tasks/datasets/models/etc. continue to use the normalized
# corpus because their normalization is just spelling unification.
DEFAULT_RAW_INPUT = BASE / "llm-merged-results"
RAW_CRITERIA_CATEGORIES = {"LLM Criteria", "Human Criteria"}
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "outputs" / "figures"


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v is not None]
    return []


def get_year_from_path(path: Path) -> int | None:
    for part in path.parts[::-1]:
        if "-" in part:
            maybe_year = part.split("-")[-1]
            if maybe_year.isdigit():
                return int(maybe_year)
    return None


def extractor_tasks(data):
    return _as_list(data.get("answer_1", {}).get("tasks"))

def extractor_datasets(data):
    return _as_list(data.get("answer_1", {}).get("datasets"))

def extractor_languages(data):
    return _as_list(data.get("answer_1", {}).get("languages"))

def extractor_models(data):
    vals = set(_as_list(data.get("answer_1", {}).get("models")))
    vals.update(_as_list(data.get("answer_3", {}).get("models")))
    return list(vals)

def extractor_automatic_metrics(data):
    return _as_list(data.get("answer_2", {}).get("automatic_metrics"))

def extractor_llm_criteria(data):
    return _as_list(data.get("answer_3", {}).get("criteria"))

def extractor_human_criteria(data):
    return _as_list(data.get("answer_4", {}).get("criteria"))

def extractor_llm_eval_models(data):
    """Extract LLM models used for evaluation (answer_3 only)"""
    return _as_list(data.get("answer_3", {}).get("models"))


CATEGORY_EXTRACTORS: dict[str, Callable[[dict], list[str]]] = {
    "Tasks": extractor_tasks,
    "Datasets": extractor_datasets,
    "Languages": extractor_languages,
    "Models": extractor_models,
    "Automatic Metrics": extractor_automatic_metrics,
    "LLM Criteria": extractor_llm_criteria,
    "Human Criteria": extractor_human_criteria,
    "LLM Eval Models": extractor_llm_eval_models,
}


def collect_counts(
    input_dir: Path,
    categories: list[str],
    raw_input_dir: Path | None = None,
) -> dict[str, Counter]:
    """Return per-category Counter of unique terms per year.

    Categories in RAW_CRITERIA_CATEGORIES are read from raw_input_dir
    (pre-normalization corpus) so that variant proliferation is preserved.
    All other categories are read from input_dir (normalized corpus)."""
    seen: dict[str, dict[int, set[str]]] = {c: defaultdict(set) for c in categories}

    norm_categories = [c for c in categories if c not in RAW_CRITERIA_CATEGORIES]
    raw_categories = [c for c in categories if c in RAW_CRITERIA_CATEGORIES]

    def _walk(dir_: Path, cats: list[str]) -> None:
        if not cats:
            return
        for path in dir_.rglob("*.json"):
            year = get_year_from_path(path)
            if year is None:
                continue
            data = json.loads(path.read_text())
            for cat in cats:
                # Apply spelling unification to criteria so the count reflects
                # conceptual variants rather than case/punctuation noise; other
                # categories are already spelling-unified by the upstream
                # mapping CSVs in metadata_unique_counts/.
                if cat in RAW_CRITERIA_CATEGORIES:
                    terms = {spelling_unify(t) for t in CATEGORY_EXTRACTORS[cat](data)}
                else:
                    terms = set(CATEGORY_EXTRACTORS[cat](data))
                for term in terms:
                    if term:
                        seen[cat][year].add(term)

    _walk(input_dir, norm_categories)
    if raw_input_dir is not None and raw_categories:
        _walk(raw_input_dir, raw_categories)
    elif raw_categories:
        # Fall back to input_dir if no raw dir was provided.
        _walk(input_dir, raw_categories)

    counts: dict[str, Counter] = {c: Counter() for c in categories}
    for cat in categories:
        for year, terms in seen[cat].items():
            counts[cat][year] = len(terms)
    return counts


def count_total_papers(papers_dir: Path) -> Counter:
    """Count total papers per year from papers/*/merged directories"""
    year_counts = Counter()

    for merged_dir in papers_dir.glob("*/merged"):
        # Extract year from directory name (e.g., ACL-2023)
        parent_name = merged_dir.parent.name
        if "-" in parent_name:
            year_str = parent_name.split("-")[-1]
            if year_str.isdigit():
                year = int(year_str)
                # Count JSON files in merged directory
                json_files = list(merged_dir.glob("*.json"))
                # Add to the year total (aggregate across all conferences)
                year_counts[year] += len(json_files)

    return year_counts


def count_nlg_papers(llm_results_dir: Path) -> Counter:
    """Count NLG papers per year from llm-merged-results directories"""
    year_counts = Counter()

    for conf_dir in llm_results_dir.iterdir():
        if not conf_dir.is_dir() or conf_dir.name.startswith('.'):
            continue

        # Extract year from directory name (e.g., ACL-2023)
        if "-" in conf_dir.name:
            year_str = conf_dir.name.split("-")[-1]
            if year_str.isdigit():
                year = int(year_str)
                # Count JSON files
                json_files = list(conf_dir.glob("*.json"))
                year_counts[year] += len(json_files)

    return year_counts


def plot_small_multiples(counts: dict[str, Counter], total_papers: Counter,
                         nlg_papers: Counter, output: Path) -> None:
    """Option 1: Small multiples - one subplot per category"""
    years = sorted(set(y for c in counts.values() for y in c))

    # Separate out special categories for combined plots
    regular_categories = [cat for cat in counts.keys()
                         if cat not in ['LLM Criteria', 'Human Criteria', 'LLM Eval Models']]

    n_cats = len(regular_categories) + 3  # +1 for papers, +1 for criteria, +1 for LLM eval models

    fig, axes = plt.subplots(1, n_cats, figsize=(15, 2.5), sharex=True)

    colors_papers = ['#2E4057', '#048A81']  # Total and NLG papers
    colors_criteria = ['#72B7B2', '#9C755F']  # LLM and Human criteria
    colors_rest = ['#4C78A8', '#B279A2', '#54A24B', '#F58518', '#E45756']
    color_llm_eval = '#E67E22'  # Orange for LLM eval models

    # First subplot: Combined papers
    ax = axes[0]

    # Plot Total Papers
    vals_total = [total_papers.get(y, 0) for y in years]
    ax.plot(years, vals_total, marker='o', linewidth=2, markersize=4,
           color=colors_papers[0], alpha=0.9, label='Total')
    ax.fill_between(years, vals_total, alpha=0.2, color=colors_papers[0])

    # Plot NLG Papers
    vals_nlg = [nlg_papers.get(y, 0) for y in years]
    ax.plot(years, vals_nlg, marker='s', linewidth=2, markersize=4,
           color=colors_papers[1], alpha=0.9, label='NLG')
    ax.fill_between(years, vals_nlg, alpha=0.2, color=colors_papers[1])

    ax.set_title('Papers', fontsize=9, fontweight='bold', pad=4)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=6, loc='best', framealpha=0.9)

    # Regular category subplots
    for i, (cat, color) in enumerate(zip(regular_categories, colors_rest), start=1):
        ax = axes[i]
        counter = counts[cat]
        vals = [counter.get(y, 0) for y in years]
        ax.plot(years, vals, marker='o', linewidth=2, markersize=4,
               color=color, alpha=0.9)
        ax.fill_between(years, vals, alpha=0.2, color=color)

        # Use "NLG Models" instead of "Models" for display
        title = "NLG Models" if cat == "Models" else cat
        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Second-to-last subplot: Combined criteria
    ax = axes[-2]

    # Plot LLM Criteria
    if 'LLM Criteria' in counts:
        vals_llm = [counts['LLM Criteria'].get(y, 0) for y in years]
        ax.plot(years, vals_llm, marker='o', linewidth=2, markersize=4,
               color=colors_criteria[0], alpha=0.9, label='LLM')
        ax.fill_between(years, vals_llm, alpha=0.2, color=colors_criteria[0])

    # Plot Human Criteria
    if 'Human Criteria' in counts:
        vals_human = [counts['Human Criteria'].get(y, 0) for y in years]
        ax.plot(years, vals_human, marker='s', linewidth=2, markersize=4,
               color=colors_criteria[1], alpha=0.9, label='Human')
        ax.fill_between(years, vals_human, alpha=0.2, color=colors_criteria[1])

    ax.set_title('Criteria', fontsize=9, fontweight='bold', pad=4)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=6, loc='best', framealpha=0.9)

    # Last subplot: LaaJ Models
    ax = axes[-1]

    if 'LLM Eval Models' in counts:
        vals = [counts['LLM Eval Models'].get(y, 0) for y in years]
        ax.plot(years, vals, marker='o', linewidth=2, markersize=4,
               color=color_llm_eval, alpha=0.9)
        ax.fill_between(years, vals, alpha=0.2, color=color_llm_eval)

    ax.set_title('LaaJ Models', fontsize=9, fontweight='bold', pad=4)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.text(0.5, -0.02, 'Year', ha='center', fontsize=9, fontweight='bold')
    fig.tight_layout(pad=0.4)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def plot_normalized_growth(counts: dict[str, Counter], output: Path) -> None:
    """Option 2: Normalized growth from baseline year"""
    years = sorted(set(y for c in counts.values() for y in c))
    baseline_year = min(years)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#4C78A8', '#B279A2', '#54A24B', '#F58518', 
              '#E45756', '#72B7B2', '#9C755F']
    
    for cat, color in zip(counts.keys(), colors):
        counter = counts[cat]
        vals = [counter.get(y, 0) for y in years]
        baseline = vals[0] if vals[0] > 0 else 1  # Avoid division by zero
        
        # Calculate percentage change from baseline
        normalized = [(v / baseline - 1) * 100 for v in vals]
        
        ax.plot(years, normalized, marker='o', linewidth=2.5, markersize=6,
               label=cat, color=color, alpha=0.85)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Growth from {baseline_year} (%)', fontsize=11, fontweight='bold')
    ax.set_title('Normalized Growth in Unique Terms Over Time', 
                fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def plot_heatmap(counts: dict[str, Counter], output: Path) -> None:
    """Option 3: Heatmap showing intensity"""
    years = sorted(set(y for c in counts.values() for y in c))
    categories = list(counts.keys())
    
    # Build matrix
    matrix = []
    for cat in categories:
        counter = counts[cat]
        vals = [counter.get(y, 0) for y in years]
        matrix.append(vals)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Normalize each row independently for better color contrast
    matrix_normalized = matrix / matrix.max(axis=1, keepdims=True)
    
    im = ax.imshow(matrix_normalized, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    
    # Add text annotations with actual counts
    for i, cat in enumerate(categories):
        for j, year in enumerate(years):
            text = ax.text(j, i, str(matrix[i, j]),
                          ha="center", va="center", color="black", 
                          fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_title('Unique Term Counts by Category and Year', 
                fontsize=12, fontweight='bold', pad=10)
    
    cbar = plt.colorbar(im, ax=ax, label='Relative Intensity', shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alternative visualizations")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Normalized corpus (used for tasks/datasets/models/etc.).")
    parser.add_argument("--raw-input-dir", type=Path, default=DEFAULT_RAW_INPUT,
                        help="Raw (pre-QCET) corpus, used for criteria so variant "
                             "proliferation is preserved instead of being collapsed "
                             "into ~120 QCET buckets.")
    args = parser.parse_args()

    categories = list(CATEGORY_EXTRACTORS.keys())
    counts = collect_counts(args.input_dir, categories, raw_input_dir=args.raw_input_dir)

    # Count total and NLG papers
    papers_dir = BASE / "papers"
    llm_results_dir = BASE / "llm-merged-results"  # full unfiltered set

    print("Counting papers...")
    total_papers = count_total_papers(papers_dir)
    nlg_papers = count_nlg_papers(llm_results_dir)

    print(f"Total papers by year: {dict(sorted(total_papers.items()))}")
    print(f"NLG papers by year: {dict(sorted(nlg_papers.items()))}")

    print("\nGenerating alternative visualizations...")

    # Generate all three options
    plot_small_multiples(counts, total_papers, nlg_papers,
                        DEFAULT_OUTPUT_DIR / "term_counts_option1_small_multiples.png")
    plot_normalized_growth(counts, DEFAULT_OUTPUT_DIR / "term_counts_option2_normalized.png")
    plot_heatmap(counts, DEFAULT_OUTPUT_DIR / "term_counts_option3_heatmap.png")

    print("\nGenerated 3 alternative visualizations:")
    print("  Option 1: Small multiples (recommended for clarity)")
    print("  Option 2: Normalized growth (shows relative trends)")
    print("  Option 3: Heatmap (most compact)")


if __name__ == "__main__":
    main()
