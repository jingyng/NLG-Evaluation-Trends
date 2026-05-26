#!/usr/bin/env python3
"""
Analyze the relationship between automatic metrics and human evaluation.
Research question: When papers use human evaluation, which automatic metrics do they pair it with?
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from association_measures import compute_all, bh_fdr


BASE = Path(__file__).parent.parent
DEFAULT_INPUT = BASE / "data" / "llm-merged-results-top30-tasks"
DEFAULT_OUTPUT = BASE / "outputs" / "figures" / "metrics_vs_human_eval"


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v is not None]
    return []


def collect_data(input_dir: Path) -> tuple[dict, dict, dict]:
    """
    Collect data about papers, separating those with and without human evaluation.

    Returns:
        - papers_with_human: {paper_id: {metrics: [...], criteria: [...], ...}}
        - papers_without_human: {paper_id: {metrics: [...]}}
        - overall_stats: general statistics
    """
    papers_with_human = {}
    papers_without_human = {}

    files = list(input_dir.rglob("*.json"))
    print(f"Processing {len(files)} papers...")

    for path in files:
        try:
            data = json.loads(path.read_text())
            paper_id = data.get("paper_id", path.stem)

            # Check if human evaluation was used
            has_human = data.get("answer_4", {}).get("answer") == "Yes"

            # Extract automatic metrics
            metrics = _as_list(data.get("answer_2", {}).get("automatic_metrics"))

            # Extract task
            tasks = _as_list(data.get("answer_1", {}).get("tasks"))

            paper_info = {
                "metrics": metrics,
                "tasks": tasks,
                "path": str(path),
            }

            if has_human:
                # Extract human evaluation criteria
                criteria = _as_list(data.get("answer_4", {}).get("criteria"))
                guideline = _as_list(data.get("answer_4", {}).get("guideline"))
                paper_info["criteria"] = criteria
                paper_info["guideline"] = guideline
                papers_with_human[paper_id] = paper_info
            else:
                papers_without_human[paper_id] = paper_info

        except Exception as e:
            print(f"Error processing {path}: {e}")

    overall_stats = {
        "total_papers": len(files),
        "with_human": len(papers_with_human),
        "without_human": len(papers_without_human),
        "human_eval_rate": len(papers_with_human) / len(files) if files else 0,
    }

    return papers_with_human, papers_without_human, overall_stats


def analyze_metrics(papers_with_human: dict, papers_without_human: dict) -> dict:
    """Analyze automatic metric usage patterns."""

    # Count metrics in each group
    metrics_with_human = Counter()
    metrics_without_human = Counter()

    # Count papers using each metric
    for paper in papers_with_human.values():
        for metric in set(paper["metrics"]):  # Use set to count each metric once per paper
            if metric:
                metrics_with_human[metric] += 1

    for paper in papers_without_human.values():
        for metric in set(paper["metrics"]):
            if metric:
                metrics_without_human[metric] += 1

    # Get all unique metrics
    all_metrics = set(metrics_with_human.keys()) | set(metrics_without_human.keys())

    # Calculate statistics for each metric
    results = {}
    total_with_human = len(papers_with_human)
    total_without_human = len(papers_without_human)

    for metric in all_metrics:
        count_with = metrics_with_human.get(metric, 0)
        count_without = metrics_without_human.get(metric, 0)

        pct_with = (count_with / total_with_human * 100) if total_with_human > 0 else 0
        pct_without = (count_without / total_without_human * 100) if total_without_human > 0 else 0

        k11 = count_with
        k12 = total_with_human - count_with
        k21 = count_without
        k22 = total_without_human - count_without
        _s = compute_all(k11, k12, k21, k22)

        results[metric] = {
            "count_with_human": count_with,
            "count_without_human": count_without,
            "pct_with_human": pct_with,
            "pct_without_human": pct_without,
            "enrichment": _s["lr"],
            "k11": k11, "k12": k12, "k21": k21, "k22": k22,
            "g2": _s["g2"],
            "p_value": _s["p_value"],
            "total_count": count_with + count_without,
        }

    # BH-FDR correction over all metrics
    metrics_list = list(results.keys())
    pvals = [results[m]["p_value"] for m in metrics_list]
    qvals, _ = bh_fdr(pvals)
    for m, q in zip(metrics_list, qvals):
        results[m]["q_value"] = float(q)

    return results


def print_statistics(papers_with_human: dict, papers_without_human: dict,
                    metric_analysis: dict, overall_stats: dict):
    """Print summary statistics."""

    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total papers: {overall_stats['total_papers']}")
    print(f"Papers with human evaluation: {overall_stats['with_human']} ({overall_stats['human_eval_rate']*100:.1f}%)")
    print(f"Papers without human evaluation: {overall_stats['without_human']} ({(1-overall_stats['human_eval_rate'])*100:.1f}%)")

    # Metric usage statistics
    metrics_per_paper_with = [len(p["metrics"]) for p in papers_with_human.values()]
    metrics_per_paper_without = [len(p["metrics"]) for p in papers_without_human.values()]

    print(f"\nAverage metrics per paper (with human eval): {np.mean(metrics_per_paper_with):.2f} ± {np.std(metrics_per_paper_with):.2f}")
    print(f"Average metrics per paper (without human eval): {np.mean(metrics_per_paper_without):.2f} ± {np.std(metrics_per_paper_without):.2f}")

    # Top metrics by group
    print("\n" + "="*80)
    print("TOP METRICS WITH HUMAN EVALUATION")
    print("="*80)
    sorted_with = sorted(metric_analysis.items(),
                        key=lambda x: x[1]["count_with_human"],
                        reverse=True)[:15]
    print(f"{'Metric':<20} {'Count':<8} {'Percentage':<12} {'Enrichment':<12}")
    print("-"*80)
    for metric, stats in sorted_with:
        if stats["count_with_human"] > 0:
            enrich_str = f"{stats['enrichment']:.2f}x" if stats['enrichment'] != float('inf') else "N/A"
            print(f"{metric:<20} {stats['count_with_human']:<8} {stats['pct_with_human']:<11.1f}% {enrich_str:<12}")

    print("\n" + "="*80)
    print("TOP METRICS WITHOUT HUMAN EVALUATION")
    print("="*80)
    sorted_without = sorted(metric_analysis.items(),
                           key=lambda x: x[1]["count_without_human"],
                           reverse=True)[:15]
    print(f"{'Metric':<20} {'Count':<8} {'Percentage':<12}")
    print("-"*80)
    for metric, stats in sorted_without:
        if stats["count_without_human"] > 0:
            print(f"{metric:<20} {stats['count_without_human']:<8} {stats['pct_without_human']:<11.1f}%")

    # Metrics most enriched in papers with human evaluation
    print("\n" + "="*80)
    print("METRICS MOST ENRICHED WITH HUMAN EVALUATION")
    print("="*80)
    enriched = sorted(metric_analysis.items(),
                     key=lambda x: x[1]["enrichment"],
                     reverse=True)[:15]
    print(f"{'Metric':<20} {'With Human':<12} {'Without Human':<15} {'Enrichment':<12}")
    print("-"*80)
    for metric, stats in enriched:
        if stats["count_with_human"] >= 5 and stats['enrichment'] != float('inf'):  # Filter for significance
            print(f"{metric:<20} {stats['pct_with_human']:<11.1f}% {stats['pct_without_human']:<14.1f}% {stats['enrichment']:<11.2f}x")


def plot_comparison(metric_analysis: dict, output_dir: Path, top_k: int = 20):
    """Create comparison visualizations."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by total usage
    sorted_metrics = sorted(metric_analysis.items(),
                           key=lambda x: x[1]["total_count"],
                           reverse=True)[:top_k]

    metrics = [m[0] for m in sorted_metrics]
    pct_with = [m[1]["pct_with_human"] for m in sorted_metrics]
    pct_without = [m[1]["pct_without_human"] for m in sorted_metrics]

    # 1. Side-by-side bar chart
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, pct_with, width, label='With Human Evaluation', alpha=0.8)
    bars2 = ax.bar(x + width/2, pct_without, width, label='Without Human Evaluation', alpha=0.8)

    ax.set_xlabel('Automatic Metrics', fontsize=12)
    ax.set_ylabel('Percentage of Papers (%)', fontsize=12)
    ax.set_title(f'Top {top_k} Automatic Metrics: Usage in Papers With vs Without Human Evaluation',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_sidebyside.png", dpi=200)
    plt.close()
    print(f"Saved {output_dir / 'metrics_comparison_sidebyside.png'}")

    # 2. Scatter plot: enrichment vs total usage
    fig, ax = plt.subplots(figsize=(12, 8))

    all_metrics = list(metric_analysis.items())
    x_vals = [m[1]["total_count"] for m in all_metrics]
    y_vals = [m[1]["enrichment"] if m[1]["enrichment"] != float('inf') else 5.0
              for m in all_metrics]
    labels = [m[0] for m in all_metrics]

    scatter = ax.scatter(x_vals, y_vals, alpha=0.6, s=100)

    # Add labels for notable points
    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        if x > 20 or (y > 1.5 and x > 10):  # Label popular or highly enriched metrics
            ax.annotate(label, (x, y), fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No enrichment')
    ax.set_xlabel('Total Papers Using Metric', fontsize=12)
    ax.set_ylabel('Enrichment (With Human / Without Human)', fontsize=12)
    ax.set_title('Metric Enrichment vs Popularity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_enrichment_scatter.png", dpi=200)
    plt.close()
    print(f"Saved {output_dir / 'metrics_enrichment_scatter.png'}")

    # 3. Heatmap-style comparison for top metrics
    top_metrics_sorted = sorted(metric_analysis.items(),
                               key=lambda x: x[1]["total_count"],
                               reverse=True)[:25]

    metrics_names = [m[0] for m in top_metrics_sorted]
    with_human = [m[1]["pct_with_human"] for m in top_metrics_sorted]
    without_human = [m[1]["pct_without_human"] for m in top_metrics_sorted]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a 2-column matrix
    matrix = np.array([with_human, without_human]).T

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Human\nEvaluation', 'Without Human\nEvaluation'], fontsize=11)
    ax.set_yticks(range(len(metrics_names)))
    ax.set_yticklabels(metrics_names, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage of Papers (%)', fontsize=11)

    # Annotate cells
    for i in range(len(metrics_names)):
        for j in range(2):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                         ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Top 25 Automatic Metrics:\nUsage in Papers With vs Without Human Evaluation',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_heatmap.png", dpi=200)
    plt.close()
    print(f"Saved {output_dir / 'metrics_comparison_heatmap.png'}")


def export_detailed_results(metric_analysis: dict, papers_with_human: dict,
                           papers_without_human: dict, output_dir: Path):
    """Export detailed results to JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort metrics by total count
    sorted_metrics = sorted(metric_analysis.items(),
                           key=lambda x: x[1]["total_count"],
                           reverse=True)

    results = {
        "summary": {
            "total_papers_with_human": len(papers_with_human),
            "total_papers_without_human": len(papers_without_human),
            "total_unique_metrics": len(metric_analysis),
        },
        "metrics": {
            metric: {
                "count_with_human": stats["count_with_human"],
                "count_without_human": stats["count_without_human"],
                "pct_with_human": round(stats["pct_with_human"], 2),
                "pct_without_human": round(stats["pct_without_human"], 2),
                "enrichment": round(stats["enrichment"], 4) if stats["enrichment"] not in (float('inf'), float('-inf')) else None,
                "k11": stats["k11"], "k12": stats["k12"], "k21": stats["k21"], "k22": stats["k22"],
                "g2": round(stats["g2"], 4),
                "p_value": stats["p_value"],
                "q_value": round(stats["q_value"], 6),
                "total_count": stats["total_count"],
            }
            for metric, stats in sorted_metrics
        }
    }

    output_file = output_dir / "metrics_vs_human_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved detailed results to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze automatic metrics usage in papers with vs without human evaluation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing normalized JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save results and visualizations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top metrics to show in visualizations.",
    )
    args = parser.parse_args()

    # Collect data
    papers_with_human, papers_without_human, overall_stats = collect_data(args.input_dir)

    # Analyze metrics
    metric_analysis = analyze_metrics(papers_with_human, papers_without_human)

    # Print statistics
    print_statistics(papers_with_human, papers_without_human, metric_analysis, overall_stats)

    # Create visualizations
    plot_comparison(metric_analysis, args.output_dir, top_k=args.top_k)

    # Export detailed results
    export_detailed_results(metric_analysis, papers_with_human, papers_without_human,
                           args.output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
