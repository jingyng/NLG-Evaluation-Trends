#!/usr/bin/env python3
"""Empirical signatures of "metric inertia" (paper appendix figure).

Operationalizes the term along three axes computed from the normalized
per-paper extractions in results/llm-merged-results-top30-tasks/:

  (1) Persistence   - share of papers in each of the four focus tasks using
                      a legacy metric (BLEU, ROUGE, METEOR, F1, Exact Match)
                      remains nearly flat across 2020-2025.
  (2) Displacement  - share of papers using a semantic / learned alternative
                      grows but rarely replaces the legacy metric: most
                      papers report both, and the proportion that has
                      *dropped* legacy metrics entirely stays small.
  (3) Generality    - legacy metrics have low task-conditional LR
                      (P(metric|task) / P(metric|not task)), i.e. they are
                      applied across many tasks rather than tied to any.

Outputs:
  paper/imgs/metric_inertia_appendix.pdf        (figure for appendix)
  analysis/intermediate_results/metric_inertia_summary.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "results" / "llm-merged-results-top30-tasks"
FIG_OUT = ROOT / "paper" / "imgs" / "metric_inertia_appendix.pdf"
CSV_OUT = ROOT / "analysis" / "intermediate_results" / "metric_inertia_summary.csv"

FOCUS_TASKS = [
    "Machine Translation",
    "Dialogue Generation",
    "Text Summarization",
    "Question Answering",
]

# Metrics are uppercased in the normalized data (see term-normalization step).
# Legacy = n-gram / lexical-overlap or fixed-string-match metrics that
#          predate transformer-based NLG.
LEGACY_METRICS = ["BLEU", "ROUGE", "METEOR", "F1", "EXACT MATCH"]

# Semantic / learned = embedding- or LLM-based reference-flexible metrics.
SEMANTIC_METRICS = [
    "BERTSCORE",
    "BLEURT",
    "COMET",
    "BARTSCORE",
    "MOVERSCORE",
    "UNIEVAL",
    "QUESTEVAL",
    "FACTCC",
    "SUMMAC",
    "ALIGNSCORE",
    "XCOMET",
    "G EVAL",
]

# Per-task curated shortlist for the line plots (kept readable).
TASK_LEGACY = {
    "Machine Translation": ["BLEU", "METEOR"],
    "Dialogue Generation": ["BLEU", "ROUGE"],
    "Text Summarization": ["ROUGE", "BLEU"],
    "Question Answering": ["EXACT MATCH", "F1"],
}
TASK_SEMANTIC = {
    "Machine Translation": ["COMET", "BLEURT", "BERTSCORE"],
    "Dialogue Generation": ["BERTSCORE", "BLEURT"],
    "Text Summarization": ["BERTSCORE", "BARTSCORE", "MOVERSCORE"],
    "Question Answering": ["BERTSCORE", "ALIGNSCORE"],
}


PRETTY = {
    "BLEU": "BLEU",
    "ROUGE": "ROUGE",
    "METEOR": "METEOR",
    "F1": "F1",
    "EXACT MATCH": "Exact Match",
    "BERTSCORE": "BERTScore",
    "BLEURT": "BLEURT",
    "COMET": "COMET",
    "BARTSCORE": "BARTScore",
    "MOVERSCORE": "MoverScore",
    "UNIEVAL": "UniEval",
    "QUESTEVAL": "QuestEval",
    "FACTCC": "FactCC",
    "SUMMAC": "SummaC",
    "ALIGNSCORE": "AlignScore",
    "XCOMET": "xCOMET",
    "G EVAL": "G-Eval",
}


def pretty_metric(m: str) -> str:
    return PRETTY.get(m, m.title())


def load_papers():
    """Walk the normalized top-30-tasks tree and yield {year, tasks, metrics}."""
    papers = []
    for root, _dirs, files in os.walk(DATA_DIR):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(root, fn)) as f:
                d = json.load(f)
            year = None
            pid = d.get("paper_id", "")
            if pid:
                head = pid.split(".", 1)[0]
                if head.isdigit() and len(head) == 4:
                    year = int(head)
            if year is None:
                # fall back to folder name like ACL-2023
                folder = os.path.basename(root)
                if "-" in folder:
                    try:
                        year = int(folder.rsplit("-", 1)[-1])
                    except ValueError:
                        pass
            if year is None:
                continue
            tasks = d.get("answer_1", {}).get("tasks", []) or []
            metrics = d.get("answer_2", {}).get("automatic_metrics", []) or []
            papers.append({
                "year": year,
                "tasks": tasks,
                "metrics": [m.upper() for m in metrics],
            })
    return papers


def task_year_share(papers, task, metric, years):
    """Share of papers labelled `task` in `year` whose metric list contains `metric`."""
    out = {}
    for y in years:
        denom = sum(1 for p in papers if p["year"] == y and task in p["tasks"])
        num = sum(
            1
            for p in papers
            if p["year"] == y and task in p["tasks"] and metric in p["metrics"]
        )
        out[y] = (num / denom) if denom else float("nan")
    return out


def task_lr(papers, task, metric):
    """Task-conditional LR: P(metric | task) / P(metric | not task)."""
    n_task = sum(1 for p in papers if task in p["tasks"])
    n_not_task = sum(1 for p in papers if task not in p["tasks"])
    n_metric_task = sum(
        1 for p in papers if task in p["tasks"] and metric in p["metrics"]
    )
    n_metric_not_task = sum(
        1
        for p in papers
        if task not in p["tasks"] and metric in p["metrics"]
    )
    if n_task == 0 or n_not_task == 0 or n_metric_not_task == 0:
        return float("nan")
    return (n_metric_task / n_task) / (n_metric_not_task / n_not_task)


def displacement_share(papers, task, year, legacy_set, semantic_set):
    """Of papers in (task, year): share that report any LEGACY, share that report
    any SEMANTIC, and share that report SEMANTIC without any LEGACY (true
    displacement)."""
    pool = [p for p in papers if p["year"] == year and task in p["tasks"]]
    if not pool:
        return None
    n = len(pool)
    legacy_count = sum(1 for p in pool if any(m in legacy_set for m in p["metrics"]))
    semantic_count = sum(
        1 for p in pool if any(m in semantic_set for m in p["metrics"])
    )
    semantic_only = sum(
        1
        for p in pool
        if any(m in semantic_set for m in p["metrics"])
        and not any(m in legacy_set for m in p["metrics"])
    )
    return {
        "n": n,
        "legacy_share": legacy_count / n,
        "semantic_share": semantic_count / n,
        "semantic_without_legacy": semantic_only / n,
    }


def main():
    papers = load_papers()
    print(f"Loaded {len(papers)} NLG papers", file=sys.stderr)

    years = list(range(2020, 2026))
    legacy_set = set(LEGACY_METRICS)
    semantic_set = set(SEMANTIC_METRICS)

    # ------- Compute series for the line plots and CSV summary.
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.55,
        wspace=0.35,
    )

    legacy_color = "#c0392b"  # red family
    semantic_color = "#2c7fb8"  # blue family
    legacy_palette = ["#c0392b", "#e67e22", "#d35400"]
    semantic_palette = ["#2c7fb8", "#41ab5d", "#6a51a3", "#7fbc41"]

    summary_rows = []

    for idx, task in enumerate(FOCUS_TASKS):
        r, c = idx // 2, idx % 2
        ax = fig.add_subplot(gs[r, c * 2:c * 2 + 2])

        for j, m in enumerate(TASK_LEGACY[task]):
            series = task_year_share(papers, task, m, years)
            ax.plot(
                years,
                [series[y] * 100 for y in years],
                marker="o",
                linewidth=2.0,
                color=legacy_palette[j % len(legacy_palette)],
                label=f"{pretty_metric(m)} (legacy)",
            )
        for j, m in enumerate(TASK_SEMANTIC[task]):
            series = task_year_share(papers, task, m, years)
            ax.plot(
                years,
                [series[y] * 100 for y in years],
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color=semantic_palette[j % len(semantic_palette)],
                label=f"{pretty_metric(m)} (semantic)",
            )

        # annotate sample size per year as small text under x-axis
        ns = [
            sum(1 for p in papers if p["year"] == y and task in p["tasks"])
            for y in years
        ]
        ax.set_title(
            f"{task}  (n={ns[0]} → {ns[-1]} papers)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("% of task-year papers")
        ax.set_ylim(0, 100)
        ax.set_xticks(years)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.9)

        # accumulate displacement summary
        for y in (2020, 2025):
            d = displacement_share(papers, task, y, legacy_set, semantic_set)
            if d:
                summary_rows.append({
                    "task": task,
                    "year": y,
                    "n": d["n"],
                    "legacy_share": round(d["legacy_share"], 3),
                    "semantic_share": round(d["semantic_share"], 3),
                    "semantic_without_legacy": round(
                        d["semantic_without_legacy"], 3
                    ),
                })

    # ---- Bottom row: task-LR bar chart for the top metrics.
    ax_lr = fig.add_subplot(gs[2, :])
    metrics_for_lr = [
        "BLEU", "ROUGE", "METEOR", "F1", "EXACT MATCH",
        "BERTSCORE", "BLEURT", "COMET", "BARTSCORE", "MOVERSCORE",
    ]
    width = 0.15
    x = np.arange(len(metrics_for_lr))
    task_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for ti, task in enumerate(FOCUS_TASKS):
        lrs = [task_lr(papers, task, m) for m in metrics_for_lr]
        ax_lr.bar(
            x + (ti - 1.5) * width,
            lrs,
            width=width,
            label=task,
            color=task_colors[ti],
            edgecolor="black",
            linewidth=0.4,
        )
    ax_lr.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_lr.set_yscale("log")
    ax_lr.set_xticks(x)
    ax_lr.set_xticklabels(
        [pretty_metric(m) for m in metrics_for_lr],
        rotation=20,
        ha="right",
    )
    ax_lr.set_ylabel("Task-conditional LR (log)")
    ax_lr.set_title(
        "Generality: legacy metrics LR$\\approx$1 across all tasks; "
        "semantic metrics concentrate on their target task",
        fontsize=10,
        pad=6,
    )
    ax_lr.legend(fontsize=8, loc="upper right", ncol=4)
    ax_lr.grid(axis="y", alpha=0.3, which="both")

    fig.suptitle(
        "Empirical signatures of metric inertia "
        "(persistence, displacement, generality)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"wrote {FIG_OUT}", file=sys.stderr)

    # ---- CSV summary so the prose can cite exact numbers.
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task", "year", "n", "legacy_share",
                "semantic_share", "semantic_without_legacy",
            ],
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"wrote {CSV_OUT}", file=sys.stderr)

    # ---- Also print a one-liner per task for the prose.
    print("\nHeadline numbers:")
    for task in FOCUS_TASKS:
        d20 = displacement_share(papers, task, 2020, legacy_set, semantic_set)
        d25 = displacement_share(papers, task, 2025, legacy_set, semantic_set)
        if not d20 or not d25:
            continue
        print(
            f"  {task}: legacy share {d20['legacy_share']*100:.0f}% (2020) "
            f"-> {d25['legacy_share']*100:.0f}% (2025); "
            f"semantic share {d20['semantic_share']*100:.0f}% -> "
            f"{d25['semantic_share']*100:.0f}%; "
            f"semantic-without-legacy {d20['semantic_without_legacy']*100:.0f}% "
            f"-> {d25['semantic_without_legacy']*100:.0f}% "
            f"(n: {d20['n']} -> {d25['n']})"
        )


if __name__ == "__main__":
    main()
