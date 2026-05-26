"""
plot_metric_method_task_dotplot.py

§4.2 figure: 2×2 grid of per-task metric-method dot plots. Each panel shows,
for one of the top four NLG tasks, how each common metric's LR with LaaJ
compares with its LR with human evaluation. The figure replaces both the
earlier global dot plot and the metric-criterion diff heatmap.

Per panel:
- One row per metric, top-N selected by frequency in that task's papers.
- Two paired markers per row: orange = LR of that metric within
  task papers that use LaaJ; blue = LR within task papers that use human eval.
- Filled markers are $G^2$+BH-FDR significant ($q \\le 0.05$); open are not.
- Vertical dashed line at $LR=1$ (independence).
- Metrics are sorted (within each panel) by LaaJ$-$Human LR difference, so
  LaaJ-favored metrics sit at the top of the panel and human-favored at the
  bottom; this gives a quick visual answer to "in this task, which metrics
  do LaaJ and human evaluation agree on, and which do they diverge on?"
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from data_loader import load_data
from association_measures import compute_all, bh_fdr

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_method_task_dotplot.png"
OUT_PDF = OUT_DIR / "metric_method_task_dotplot.pdf"
NORMALIZATION_CSV = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"

TASKS_ORDERED = [
    "Dialogue Generation",
    "Machine Translation",
    "Text Summarization",
    "Question Answering",
]

TOP_METRICS_PER_TASK = 12
SIG_Q = 0.05
LR_FLOOR = 0.05
LR_CEIL = 40.0
MIN_TOTAL_IN_TASK = 5    # metric must appear in at least this many papers of the task


def load_metric_display() -> dict[str, str]:
    out: dict[str, str] = {}
    if not NORMALIZATION_CSV.exists():
        return out
    import csv
    with open(NORMALIZATION_CSV) as f:
        for row in csv.DictReader(f):
            normalized = (row.get("normalized") or "").strip()
            variants = (row.get("variants_with_counts") or "").split(";")
            if not normalized or not variants:
                continue
            top = variants[0].strip().rsplit("(", 1)[0].strip()
            out[normalized.lower()] = top
    return out


def per_task_lrs(task_papers: list, top_metrics: list[str]) -> list[dict]:
    """For each metric, compute LR/p with LaaJ-using and Human-using subsets within the task."""
    laaj_papers = [p for p in task_papers if p.get("laaj_criteria")]
    human_papers = [p for p in task_papers if p.get("human_criteria")]

    def stat_for(metric: str, subset_attr: str) -> dict:
        subset = laaj_papers if subset_attr == "laaj" else human_papers
        non_subset = [p for p in task_papers if p not in subset]
        m_low = metric.lower().strip()
        k_with = sum(1 for p in subset if m_low in {x.lower().strip() for x in (p.get("auto_metrics") or [])})
        k_without = sum(1 for p in non_subset if m_low in {x.lower().strip() for x in (p.get("auto_metrics") or [])})
        n_with = len(subset)
        n_without = len(non_subset)
        if n_with == 0 or n_without == 0:
            return {"lr": None, "p": 1.0, "k": 0}
        stats = compute_all(k_with, n_with - k_with, k_without, n_without - k_without)
        return {"lr": stats.get("lr"), "p": stats.get("p_value", 1.0), "k": int(k_with)}

    rows = []
    laaj_p, human_p = [], []
    for m in top_metrics:
        l = stat_for(m, "laaj")
        h = stat_for(m, "human")
        rows.append({"metric": m, "laaj": l, "human": h})
        laaj_p.append(l["p"])
        human_p.append(h["p"])
    laaj_q, _ = bh_fdr(laaj_p)
    human_q, _ = bh_fdr(human_p)
    for r, lq, hq in zip(rows, laaj_q, human_q):
        r["laaj"]["q"] = float(lq)
        r["human"]["q"] = float(hq)
    return rows


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")

    metric_display_map = load_metric_display()
    def metric_label(m: str) -> str:
        return metric_display_map.get(m, m.title())

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    LAAJ_C = "#ff7f0e"
    HUMAN_C = "#1f77b4"

    panel_summaries = []
    for ax, task_name in zip(axes.flat, TASKS_ORDERED):
        task_low = task_name.lower().strip()
        task_papers = [p for p in papers if any(t.lower().strip() == task_low for t in (p.get("tasks") or []))]
        n_laaj = sum(1 for p in task_papers if p.get("laaj_criteria"))
        n_human = sum(1 for p in task_papers if p.get("human_criteria"))

        # Pick top metrics by frequency in the task subset
        metric_counts: Counter = Counter()
        for p in task_papers:
            for m in (p.get("auto_metrics") or []):
                metric_counts[m.lower().strip()] += 1
        top_metrics = [m for m, c in metric_counts.most_common(TOP_METRICS_PER_TASK)
                       if c >= MIN_TOTAL_IN_TASK]

        rows = per_task_lrs(task_papers, top_metrics)

        # Sort: LaaJ-favored first
        def diff_key(r):
            l = r["laaj"]["lr"] if r["laaj"]["lr"] is not None else 1.0
            h = r["human"]["lr"] if r["human"]["lr"] is not None else 1.0
            return math.log(max(l, LR_FLOOR)) - math.log(max(h, LR_FLOOR))
        rows.sort(key=diff_key, reverse=True)

        n_sig_laaj = sum(1 for r in rows if r["laaj"]["q"] <= SIG_Q)
        n_sig_human = sum(1 for r in rows if r["human"]["q"] <= SIG_Q)
        panel_summaries.append({
            "task": task_name, "n_laaj": n_laaj, "n_human": n_human,
            "n_metrics": len(rows), "n_sig_laaj": n_sig_laaj, "n_sig_human": n_sig_human,
        })

        y = list(range(len(rows)))
        for i, r in enumerate(rows):
            l = max(min(r["laaj"]["lr"] or 1.0, LR_CEIL), LR_FLOOR)
            h = max(min(r["human"]["lr"] or 1.0, LR_CEIL), LR_FLOOR)
            ax.plot([l, h], [i, i], color="#cccccc", lw=0.8, zorder=1)
            l_sig = r["laaj"]["q"] <= SIG_Q
            h_sig = r["human"]["q"] <= SIG_Q
            ax.scatter([l], [i], s=34, color=LAAJ_C if l_sig else "white",
                       edgecolor=LAAJ_C, linewidth=1.1, zorder=3)
            ax.scatter([h], [i], s=34, color=HUMAN_C if h_sig else "white",
                       edgecolor=HUMAN_C, linewidth=1.1, zorder=3)

        ax.axvline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.5, zorder=2)
        ax.axvspan(LR_FLOOR * 0.9, 1.0, alpha=0.04, color=HUMAN_C, zorder=0)
        ax.axvspan(1.0, LR_CEIL * 1.1, alpha=0.04, color=LAAJ_C, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(LR_FLOOR * 0.9, LR_CEIL * 1.1)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.set_yticks(y)
        ax.set_yticklabels([metric_label(r["metric"]) for r in rows])
        ax.invert_yaxis()
        ax.set_title(f"{task_name}  (LaaJ: {n_laaj}, Human: {n_human})", pad=4, fontsize=9.5)

    # Common x-axis label
    for ax in axes[-1, :]:
        ax.set_xlabel("Likelihood Ratio (LR)")

    # Shared legend at top
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LAAJ_C,
               markeredgecolor=LAAJ_C, markersize=7, label="LaaJ, significant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=LAAJ_C, markersize=7, label="LaaJ, n.s."),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HUMAN_C,
               markeredgecolor=HUMAN_C, markersize=7, label="Human, significant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=HUMAN_C, markersize=7, label="Human, n.s."),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print()
    for s in panel_summaries:
        print(f"  {s['task']:25} n_LaaJ={s['n_laaj']:>4}  n_Human={s['n_human']:>4}  "
              f"metrics={s['n_metrics']}  sig(LaaJ)={s['n_sig_laaj']}  sig(Human)={s['n_sig_human']}")


if __name__ == "__main__":
    main()
