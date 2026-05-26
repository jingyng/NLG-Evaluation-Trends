#!/usr/bin/env python3
"""
Analyze the relationship between automatic metrics and LLM-as-a-judge evaluation.
Research question: When papers use LLM-as-a-judge, which automatic metrics do they pair it with?
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
DEFAULT_OUTPUT = BASE / "outputs" / "figures" / "metrics_vs_llm_eval"


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
    Collect data about papers, separating those with and without LLM-as-a-judge evaluation.

    Returns:
        - papers_with_llm: {paper_id: {metrics: [...], llm_models: [...], ...}}
        - papers_without_llm: {paper_id: {metrics: [...]}}
        - overall_stats: general statistics
    """
    papers_with_llm = {}
    papers_without_llm = {}

    files = list(input_dir.rglob("*.json"))
    print(f"Processing {len(files)} papers...")

    for path in files:
        try:
            data = json.loads(path.read_text())
            paper_id = data.get("paper_id", path.stem)

            # Check if LLM-as-a-judge evaluation was used (answer_3)
            has_llm = data.get("answer_3", {}).get("answer") == "Yes"

            # Extract automatic metrics
            metrics = _as_list(data.get("answer_2", {}).get("automatic_metrics"))

            # Extract task
            tasks = _as_list(data.get("answer_1", {}).get("tasks"))

            paper_info = {
                "metrics": metrics,
                "tasks": tasks,
                "path": str(path),
            }

            if has_llm:
                # Extract LLM-as-a-judge details
                llm_models = _as_list(data.get("answer_3", {}).get("models"))
                llm_criteria = _as_list(data.get("answer_3", {}).get("criteria"))
                llm_methods = _as_list(data.get("answer_3", {}).get("methods"))
                paper_info["llm_models"] = llm_models
                paper_info["llm_criteria"] = llm_criteria
                paper_info["llm_methods"] = llm_methods
                papers_with_llm[paper_id] = paper_info
            else:
                papers_without_llm[paper_id] = paper_info

        except Exception as e:
            print(f"Error processing {path}: {e}")

    overall_stats = {
        "total_papers": len(files),
        "with_llm": len(papers_with_llm),
        "without_llm": len(papers_without_llm),
        "llm_eval_rate": len(papers_with_llm) / len(files) if files else 0,
    }

    return papers_with_llm, papers_without_llm, overall_stats


def analyze_metrics(papers_with_llm: dict, papers_without_llm: dict) -> dict:
    """Analyze automatic metric usage patterns."""

    # Count metrics in each group
    metrics_with_llm = Counter()
    metrics_without_llm = Counter()

    # Count papers using each metric
    for paper in papers_with_llm.values():
        for metric in set(paper["metrics"]):  # Use set to count each metric once per paper
            if metric:
                metrics_with_llm[metric] += 1

    for paper in papers_without_llm.values():
        for metric in set(paper["metrics"]):
            if metric:
                metrics_without_llm[metric] += 1

    # Get all unique metrics
    all_metrics = set(metrics_with_llm.keys()) | set(metrics_without_llm.keys())

    # Calculate statistics for each metric
    results = {}
    total_with_llm = len(papers_with_llm)
    total_without_llm = len(papers_without_llm)

    for metric in all_metrics:
        count_with = metrics_with_llm.get(metric, 0)
        count_without = metrics_without_llm.get(metric, 0)

        pct_with = (count_with / total_with_llm * 100) if total_with_llm > 0 else 0
        pct_without = (count_without / total_without_llm * 100) if total_without_llm > 0 else 0

        k11 = count_with
        k12 = total_with_llm - count_with
        k21 = count_without
        k22 = total_without_llm - count_without
        _s = compute_all(k11, k12, k21, k22)

        results[metric] = {
            "count_with_llm": count_with,
            "count_without_llm": count_without,
            "pct_with_llm": pct_with,
            "pct_without_llm": pct_without,
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


def print_statistics(papers_with_llm: dict, papers_without_llm: dict,
                    metric_analysis: dict, overall_stats: dict):
    """Print summary statistics."""

    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total papers: {overall_stats['total_papers']}")
    print(f"Papers with LLM-as-a-judge: {overall_stats['with_llm']} ({overall_stats['llm_eval_rate']*100:.1f}%)")
    print(f"Papers without LLM-as-a-judge: {overall_stats['without_llm']} ({(1-overall_stats['llm_eval_rate'])*100:.1f}%)")

    # Metric usage statistics
    metrics_per_paper_with = [len(p["metrics"]) for p in papers_with_llm.values()]
    metrics_per_paper_without = [len(p["metrics"]) for p in papers_without_llm.values()]

    print(f"\nAverage metrics per paper (with LLM eval): {np.mean(metrics_per_paper_with):.2f} ± {np.std(metrics_per_paper_with):.2f}")
    print(f"Average metrics per paper (without LLM eval): {np.mean(metrics_per_paper_without):.2f} ± {np.std(metrics_per_paper_without):.2f}")

    # Top metrics by group
    print("\n" + "="*80)
    print("TOP METRICS WITH LLM-AS-A-JUDGE EVALUATION")
    print("="*80)
    sorted_with = sorted(metric_analysis.items(),
                        key=lambda x: x[1]["count_with_llm"],
                        reverse=True)[:15]
    print(f"{'Metric':<20} {'Count':<8} {'Percentage':<12} {'Enrichment':<12}")
    print("-"*80)
    for metric, stats in sorted_with:
        if stats["count_with_llm"] > 0:
            enrich_str = f"{stats['enrichment']:.2f}x" if stats['enrichment'] != float('inf') else "N/A"
            print(f"{metric:<20} {stats['count_with_llm']:<8} {stats['pct_with_llm']:<11.1f}% {enrich_str:<12}")

    print("\n" + "="*80)
    print("TOP METRICS WITHOUT LLM-AS-A-JUDGE EVALUATION")
    print("="*80)
    sorted_without = sorted(metric_analysis.items(),
                           key=lambda x: x[1]["count_without_llm"],
                           reverse=True)[:15]
    print(f"{'Metric':<20} {'Count':<8} {'Percentage':<12}")
    print("-"*80)
    for metric, stats in sorted_without:
        if stats["count_without_llm"] > 0:
            print(f"{metric:<20} {stats['count_without_llm']:<8} {stats['pct_without_llm']:<11.1f}%")


def export_detailed_results(metric_analysis: dict, papers_with_llm: dict,
                           papers_without_llm: dict, output_dir: Path):
    """Export detailed results to JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort metrics by total count
    sorted_metrics = sorted(metric_analysis.items(),
                           key=lambda x: x[1]["total_count"],
                           reverse=True)

    results = {
        "summary": {
            "total_papers_with_llm": len(papers_with_llm),
            "total_papers_without_llm": len(papers_without_llm),
            "total_unique_metrics": len(metric_analysis),
        },
        "metrics": {
            metric: {
                "count_with_llm": stats["count_with_llm"],
                "count_without_llm": stats["count_without_llm"],
                "pct_with_llm": round(stats["pct_with_llm"], 2),
                "pct_without_llm": round(stats["pct_without_llm"], 2),
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

    output_file = output_dir / "metrics_vs_llm_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved detailed results to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze automatic metrics usage in papers with vs without LLM-as-a-judge."
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
    args = parser.parse_args()

    # Collect data
    papers_with_llm, papers_without_llm, overall_stats = collect_data(args.input_dir)

    # Analyze metrics
    metric_analysis = analyze_metrics(papers_with_llm, papers_without_llm)

    # Print statistics
    print_statistics(papers_with_llm, papers_without_llm, metric_analysis, overall_stats)

    # Export detailed results
    export_detailed_results(metric_analysis, papers_with_llm, papers_without_llm,
                           args.output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
