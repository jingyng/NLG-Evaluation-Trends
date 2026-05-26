"""
plot_lr_by_rank_dashboard.py

LR-based companion to Figure 8 (plot_frequency_by_rank_dashboard.py).

Same 7-row structure (Datasets, Models, Languages, Auto Metrics, LaaJ
Models, LaaJ Criteria, Human Criteria), but only 4 task columns (DG, MT,
TS, QA). The "All Tasks" column is dropped because task-LR is computed
*against* a specific task and has no sensible aggregate analogue.

Each cell shows the top-10 items by task-LR, restricted to those that
are statistically significant after BH-FDR ($q \le 0.05$) and have
LR$>1$. Bars are sized linearly by LR up to a cap of LR=20; cells with
fewer than 10 significant items show whatever number passes.

This figure complements Figure 8 by surfacing what's actually task-fit
(LR) versus what's actually used (frequency). The two together visualise
the metric-inertia + mapping-problem story at the category level: e.g.,
in MT, Figure 8 shows BLEU dominating by frequency, while this figure
shows COMET/xCOMET/MetricX dominating by task-LR.
"""

from __future__ import annotations

import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

import os
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data, short_label
from association_measures import compute_all, bh_fdr


# Reuse load_normalization_mappings from the frequency-dashboard script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_frequency_by_rank_dashboard import load_normalization_mappings


LR_CAP = 20.0          # bars saturate at this LR value
SIG_Q = 0.05           # BH-FDR threshold
MIN_TASK_COUNT = 5     # minimum task-frequency for an item to be a candidate

TOP_TASKS = ["dialogue generation", "machine translation",
             "text summarization", "question answering"]

CATEGORIES = [
    ("datasets", "Datasets"),
    ("models", "Models"),
    ("languages", "Languages"),
    ("auto_metrics", "Automatic Metrics"),
    ("laaj_models", "LLM-as-a-Judge Models"),
    ("laaj_criteria", "LLM-as-a-Judge Criteria"),
    ("human_criteria", "Human Evaluation Criteria"),
]

CATEGORY_COLORS = {
    "datasets": "#4e79a7",
    "models": "#f28e2b",
    "languages": "#e15759",
    "auto_metrics": "#76b7b2",
    "laaj_models": "#b07aa1",
    "laaj_criteria": "#59a14f",
    "human_criteria": "#edc948",
}


def create_lr_rank_dashboard(papers):
    print("Generating LR by Rank Dashboard...")

    print("Loading normalization mappings...")
    norm_mappings, display_mappings = load_normalization_mappings()
    print(f"Loaded normalization mappings for {len(norm_mappings)} categories")

    def get_task_papers_pure(task_name):
        return [p for p in papers
                if len(p["tasks"]) == 1
                and task_name == p["tasks"][0].lower().strip()]

    def get_top_items_by_lr(task_name, field_key, top_n=10,
                             min_count=MIN_TASK_COUNT):
        """Top items by task-LR within the (task, category) cell,
        filtered to significant (G²+BH-FDR, q<=SIG_Q) with LR>1."""
        task_papers = get_task_papers_pure(task_name)
        if not task_papers:
            return []
        other_papers = [p for p in papers if p not in task_papers]
        n_task, n_other = len(task_papers), len(other_papers)
        if n_task == 0 or n_other == 0:
            return []

        field_mapping = norm_mappings.get(field_key, {})
        display_mapping = display_mappings.get(field_key, {})

        def normalize(item):
            return field_mapping.get(item.lower().strip(), item)

        task_counts: Counter = Counter()
        other_counts: Counter = Counter()
        for paper in task_papers:
            items = {normalize(it) for it in (paper.get(field_key) or [])}
            for it in items:
                task_counts[it] += 1
        for paper in other_papers:
            items = {normalize(it) for it in (paper.get(field_key) or [])}
            for it in items:
                other_counts[it] += 1

        cand = [it for it, c in task_counts.items() if c >= min_count]
        if not cand:
            return []

        lrs, pvals = [], []
        for it in cand:
            k_w = task_counts[it]
            k_o = other_counts.get(it, 0)
            stats = compute_all(k_w, n_task - k_w, k_o, n_other - k_o)
            lr = stats.get("lr")
            p = stats.get("p_value", 1.0)
            lrs.append(lr if lr is not None and np.isfinite(lr) else 0.0)
            pvals.append(p)

        qvals, _ = bh_fdr(pvals)

        sig = [(it, lr) for it, lr, q in zip(cand, lrs, qvals)
               if q <= SIG_Q and lr > 1.0]
        sig.sort(key=lambda x: -x[1])
        top = sig[:top_n]

        result = []
        for it, lr in top:
            display = display_mapping.get(it, it)
            if field_key in ("laaj_criteria", "human_criteria"):
                display = short_label(display)
            result.append((display, lr))
        return result

    sns.set_theme(style="whitegrid")
    n_rows = len(CATEGORIES)
    n_cols = len(TOP_TASKS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 35))

    all_lrs = []
    for row_idx, (field_key, category_name) in enumerate(CATEGORIES):
        for col_idx, task in enumerate(TOP_TASKS):
            ax = axes[row_idx, col_idx]
            top_items = get_top_items_by_lr(task, field_key, top_n=10)

            if top_items:
                all_lrs.extend([lr for _, lr in top_items])

            if not top_items:
                ax.text(0.5, 0.5, "No significant items",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=12, color="#666666")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            else:
                # Order: most LR at top of the bar chart (reverse for barh)
                items = [it for it, _ in top_items][::-1]
                lrs = [lr for _, lr in top_items][::-1]
                bar_lengths = [min(lr, LR_CAP) for lr in lrs]

                y_pos = np.arange(len(items))
                ax.barh(y_pos, bar_lengths,
                        color=CATEGORY_COLORS[field_key], alpha=0.85)

                for i, (length, lr) in enumerate(zip(bar_lengths, lrs)):
                    if lr >= LR_CAP:
                        label = f"≥{int(LR_CAP)}"
                    else:
                        label = f"{lr:.1f}"
                    ax.text(length + LR_CAP * 0.015, i, label,
                            va="center", fontsize=10, fontweight="bold")

                ax.set_yticks(y_pos)
                ax.set_yticklabels(items, fontsize=11)
                ax.set_xlabel("Task-LR (capped at 20)", fontsize=11,
                              fontweight="bold")
                ax.set_xlim(0, LR_CAP * 1.18)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            if row_idx == 0:
                n_papers = len(get_task_papers_pure(task))
                ax.set_title(f"{task.title()}\n(N={n_papers})",
                             fontsize=16, fontweight="bold", pad=15)
            if col_idx == 0:
                ax.set_ylabel(category_name, fontsize=14,
                              fontweight="bold", rotation=90, labelpad=15)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if all_lrs:
        print("\nLR distribution across all significant cells:")
        print(f"  min  : {min(all_lrs):.2f}")
        print(f"  mean : {np.mean(all_lrs):.2f}")
        print(f"  max  : {max(all_lrs):.2f}")
        print(f"  n>=cap ({LR_CAP}): {sum(1 for x in all_lrs if x >= LR_CAP)}")

    out_path = Path(__file__).parent.parent / "figures" / \
        "lr_dashboard" / "lr_by_rank_dashboard.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {out_path}")

    # Also copy to paper imgs/
    paper_imgs = Path(__file__).parent.parent.parent / "my_paper" / \
        "NLG_Evaluation_Trend_and_Analysis__ACL_2026___Arxiv_" / "imgs"
    if paper_imgs.exists():
        import shutil
        target = paper_imgs / "lr_by_rank_dashboard.png"
        shutil.copy(out_path, target)
        print(f"Copied to {target}")


if __name__ == "__main__":
    papers = load_data()
    print(f"Loaded {len(papers)} papers")
    create_lr_rank_dashboard(papers)
