"""
plot_sig_items_freq_dashboard.py

Alternative view of plot_lr_by_rank_dashboard.py: same significance
filter (items must be significant via G²+BH-FDR, q<=0.05, LR>1), but
the items are RANKED and the bars are SIZED by raw frequency instead
of by LR.

Reading: "of the items that are statistically distinctive for this task,
which ones are the most frequently used?" This bridges the LR view
(task-distinctive items, regardless of how often they appear) and the
frequency view of Figure 8 (most-frequent items, regardless of whether
they are task-specific).

Layout: 7 rows (Datasets, Models, Languages, Auto Metrics, LaaJ Models,
LaaJ Criteria, Human Criteria) x 4 task columns (DG, MT, TS, QA).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_frequency_by_rank_dashboard import load_normalization_mappings


SIG_Q = 0.05
MIN_TASK_COUNT = 5

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


def create_dashboard(papers):
    print("Generating Significant-Items-by-Frequency Dashboard...")
    norm_mappings, display_mappings = load_normalization_mappings()
    print(f"Loaded normalization mappings for {len(norm_mappings)} categories")

    def get_task_papers_pure(task_name):
        return [p for p in papers
                if len(p["tasks"]) == 1
                and task_name == p["tasks"][0].lower().strip()]

    def get_top_items(task_name, field_key, top_n=10,
                       min_count=MIN_TASK_COUNT):
        """Items that are LR-significant for the task, ranked by raw
        frequency within the task. Returns (display_name, freq, lr)."""
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

        # Significant items only; rank by raw frequency descending
        sig = [(it, task_counts[it], lr)
               for it, lr, q in zip(cand, lrs, qvals)
               if q <= SIG_Q and lr > 1.0]
        sig.sort(key=lambda x: -x[1])
        top = sig[:top_n]

        result = []
        for it, freq, lr in top:
            display = display_mapping.get(it, it)
            if field_key in ("laaj_criteria", "human_criteria"):
                display = short_label(display)
            result.append((display, freq, lr))
        return result

    sns.set_theme(style="whitegrid")
    n_rows, n_cols = len(CATEGORIES), len(TOP_TASKS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 35))

    all_freqs = []
    for row_idx, (field_key, category_name) in enumerate(CATEGORIES):
        for col_idx, task in enumerate(TOP_TASKS):
            ax = axes[row_idx, col_idx]
            top_items = get_top_items(task, field_key, top_n=10)

            if top_items:
                all_freqs.extend([freq for _, freq, _ in top_items])

            if not top_items:
                ax.text(0.5, 0.5, "No significant items",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=12, color="#666666")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            else:
                items = [it for it, _, _ in top_items][::-1]
                freqs = [f for _, f, _ in top_items][::-1]
                lrs = [lr for _, _, lr in top_items][::-1]

                y_pos = np.arange(len(items))
                ax.barh(y_pos, freqs,
                        color=CATEGORY_COLORS[field_key], alpha=0.85)

                # Text label: frequency + parenthesised LR (so reader can
                # still see how task-distinctive each frequent-and-sig
                # item is)
                max_freq = max(freqs)
                for i, (f, lr) in enumerate(zip(freqs, lrs)):
                    label = f"{f}  (LR={lr:.1f})" if lr < 100 else f"{f}  (LR={lr:.0f})"
                    ax.text(f + max_freq * 0.02, i, label,
                            va="center", fontsize=9, fontweight="bold")

                ax.set_yticks(y_pos)
                ax.set_yticklabels(items, fontsize=11)
                ax.set_xlabel("Frequency (significant items only)",
                              fontsize=11, fontweight="bold")
                ax.set_xlim(0, max_freq * 1.45)
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

    if all_freqs:
        print("\nFrequency distribution across all significant cells:")
        print(f"  min  : {min(all_freqs):.0f}")
        print(f"  mean : {np.mean(all_freqs):.1f}")
        print(f"  max  : {max(all_freqs):.0f}")

    out_path = Path(__file__).parent.parent / "figures" / \
        "sig_items_freq_dashboard" / "sig_items_freq_dashboard.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {out_path}")

    paper_imgs = Path(__file__).parent.parent.parent / "my_paper" / \
        "NLG_Evaluation_Trend_and_Analysis__ACL_2026___Arxiv_" / "imgs"
    if paper_imgs.exists():
        import shutil
        target = paper_imgs / "sig_items_freq_dashboard.png"
        shutil.copy(out_path, target)
        print(f"Copied to {target}")


if __name__ == "__main__":
    papers = load_data()
    print(f"Loaded {len(papers)} papers")
    create_dashboard(papers)
