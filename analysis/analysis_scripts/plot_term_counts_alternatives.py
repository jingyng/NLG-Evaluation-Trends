#!/usr/bin/env python3
"""
Alternative visualizations for term counts by year
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


BASE = Path(__file__).parent.parent
DEFAULT_INPUT = BASE / "llm-merged-results-normalized"
DEFAULT_OUTPUT_DIR = BASE / "analysis" / "figures"


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


def collect_counts(input_dir: Path, categories: list[str]) -> dict[str, Counter]:
    """Return per-category Counter of unique terms per year."""
    seen: dict[str, dict[int, set[str]]] = {c: defaultdict(set) for c in categories}

    for path in input_dir.rglob("*.json"):
        year = get_year_from_path(path)
        if year is None:
            continue
        data = json.loads(path.read_text())
        for cat in categories:
            terms = set(CATEGORY_EXTRACTORS[cat](data))
            for term in terms:
                if term:
                    seen[cat][year].add(term)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alternative visualizations")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    categories = list(CATEGORY_EXTRACTORS.keys())
    counts = collect_counts(args.input_dir, categories)

    # Count total and NLG papers
    papers_dir = BASE / "papers"
    llm_results_dir = BASE / "llm-merged-results"

    print("Counting papers...")
    total_papers = count_total_papers(papers_dir)
    nlg_papers = count_nlg_papers(llm_results_dir)

    print(f"Total papers by year: {dict(sorted(total_papers.items()))}")
    print(f"NLG papers by year: {dict(sorted(nlg_papers.items()))}")

    print("\nGenerating alternative visualizations...")

    # Generate all three options
    plot_small_multiples(counts, total_papers, nlg_papers,
                        DEFAULT_OUTPUT_DIR / "term_counts_option1_small_multiples.png")

    print("\nGenerated 3 alternative visualizations:")
    print("  Option 1: Small multiples (recommended for clarity)")
    print("  Option 2: Normalized growth (shows relative trends)")
    print("  Option 3: Heatmap (most compact)")


if __name__ == "__main__":
    main()
