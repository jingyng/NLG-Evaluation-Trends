"""
plot_metric_criterion_task_diff.py

§4.2 figure: 2×2 grid of per-task diff heatmaps. Each panel shows, for one
of the top four NLG tasks, the difference $\\log LR_{LaaJ} - \\log LR_{Human}$
per (metric, criterion) cell within that task's papers.

Replaces both the metric-method dot plot and the metric-criterion diff heatmap
in §4.2: combines task-specific resolution with the LaaJ-vs-Human divergence
story, and lets readers see which tasks have orthogonal evaluation paradigms
vs. which are uniformly dominated by a single metric family.

Layout:
  ┌──────────────┬──────────────┐
  │  Dialogue    │  Translation │
  │  Generation  │              │
  ├──────────────┼──────────────┤
  │  Text        │  Question    │
  │  Summarization│ Answering   │
  └──────────────┴──────────────┘
Shared diverging colorbar (orange = LaaJ-favored, blue = human-favored).
Cells are hatched if non-significant ($G^2$+BH-FDR) or low-support.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from data_loader import load_data, short_label
from association_measures import compute_all, bh_fdr

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_criterion_task_diff.png"
OUT_PDF = OUT_DIR / "metric_criterion_task_diff.pdf"
NORMALIZATION_CSV = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"

TASKS_ORDERED = [
    ("Dialogue Generation",   "Dialogue Generation"),
    ("Machine Translation",   "Machine Translation"),
    ("Text Summarization",    "Text Summarization"),
    ("Question Answering",    "Question Answering"),
]

TOP_METRICS_PER_TASK = 12
TOP_CRITERIA_PER_TASK = 8
SIG_Q = 0.05
MIN_PAIR_CO_OCC = 8


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


def select_for_task(papers: list, task_name: str,
                    top_metrics: int, top_criteria: int) -> tuple[list, list, list, list]:
    """Subset to papers including the task; pick top metrics + top criteria for it."""
    task_low = task_name.lower().strip()
    task_papers = [p for p in papers if any(t.lower().strip() == task_low for t in (p.get("tasks") or []))]
    laaj_papers = [p for p in task_papers if p.get("laaj_criteria")]
    human_papers = [p for p in task_papers if p.get("human_criteria")]

    metric_counts: Counter = Counter()
    for p in task_papers:
        for m in (p.get("auto_metrics") or []):
            metric_counts[m.lower().strip()] += 1
    top_m = [m for m, _ in metric_counts.most_common(top_metrics)]

    crit_counts: Counter = Counter()
    for p in task_papers:
        for c in (p.get("laaj_criteria") or []):
            crit_counts[c.strip()] += 1
        for c in (p.get("human_criteria") or []):
            crit_counts[c.strip()] += 1
    top_c = [c for c, _ in crit_counts.most_common(top_criteria)]

    return laaj_papers, human_papers, top_m, top_c


def compute_diff(laaj_papers: list, human_papers: list,
                 top_metrics: list[str], top_criteria: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (diff matrix, mask matrix) for one task."""
    n_m, n_c = len(top_metrics), len(top_criteria)
    lr_l = np.ones((n_m, n_c)); lr_h = np.ones((n_m, n_c))
    p_l = np.ones((n_m, n_c));  p_h = np.ones((n_m, n_c))
    co_l = np.zeros((n_m, n_c)); co_h = np.zeros((n_m, n_c))

    def fill(papers: list, lr: np.ndarray, p: np.ndarray, co: np.ndarray, crit_field: str):
        if not papers:
            return
        metric_sets = [{x.lower().strip() for x in (q.get("auto_metrics") or [])} for q in papers]
        crit_sets = [{x.lower().strip() for x in (q.get(crit_field) or [])} for q in papers]
        for ci, c in enumerate(top_criteria):
            cl = c.lower().strip()
            with_idx = [i for i, s in enumerate(crit_sets) if cl in s]
            without_idx = [i for i, s in enumerate(crit_sets) if cl not in s]
            n_w, n_wo = len(with_idx), len(without_idx)
            if n_w == 0 or n_wo == 0:
                continue
            for mi, m in enumerate(top_metrics):
                ml = m.lower().strip()
                k_w = sum(1 for i in with_idx if ml in metric_sets[i])
                k_wo = sum(1 for i in without_idx if ml in metric_sets[i])
                stats = compute_all(k_w, n_w - k_w, k_wo, n_wo - k_wo)
                lr_val = stats.get("lr"); p_val = stats.get("p_value", 1.0)
                if lr_val is None or not np.isfinite(lr_val):
                    lr_val = 1.0
                lr[mi, ci] = lr_val
                p[mi, ci] = p_val
                co[mi, ci] = k_w

    fill(laaj_papers,  lr_l, p_l, co_l, "laaj_criteria")
    fill(human_papers, lr_h, p_h, co_h, "human_criteria")

    flat_q_l, _ = bh_fdr(p_l.flatten().tolist())
    flat_q_h, _ = bh_fdr(p_h.flatten().tolist())
    q_l = np.array(flat_q_l).reshape(p_l.shape)
    q_h = np.array(flat_q_h).reshape(p_h.shape)

    eps = 1e-3
    diff = np.log(np.maximum(lr_l, eps)) - np.log(np.maximum(lr_h, eps))
    sig = (q_l <= SIG_Q) | (q_h <= SIG_Q)
    support = (co_l + co_h) >= MIN_PAIR_CO_OCC
    mask = sig & support
    return diff, mask


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")

    metric_display_map = load_metric_display()
    def metric_label(m: str) -> str:
        return metric_display_map.get(m, m.title())

    # Build per-task data
    task_data = []
    for task_name, label in TASKS_ORDERED:
        laaj, human, top_m, top_c = select_for_task(
            papers, task_name, TOP_METRICS_PER_TASK, TOP_CRITERIA_PER_TASK
        )
        diff, mask = compute_diff(laaj, human, top_m, top_c)
        task_data.append({
            "task": task_name, "label": label,
            "n_laaj": len(laaj), "n_human": len(human),
            "metrics": top_m, "criteria": top_c,
            "diff": diff, "mask": mask,
        })
        print(f"\n{label}: n_LaaJ={len(laaj)}  n_Human={len(human)}")
        print(f"  metrics: {[metric_label(m) for m in top_m]}")
        print(f"  criteria: {[short_label(c, prefixed=False) for c in top_c]}")
        print(f"  significant cells: {int(mask.sum())} / {mask.size}")

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    cmap = LinearSegmentedColormap.from_list(
        "laaj_human_diff",
        [(0.0, "#1f77b4"), (0.5, "#ffffff"), (1.0, "#ff7f0e")],
        N=256,
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.2))
    vmax = 2.0  # symmetric clip; keeps colorbar legible

    images = []
    for ax, td in zip(axes.flat, task_data):
        diff = td["diff"]; mask = td["mask"]
        diff_display = np.where(mask, diff, 0.0)
        clipped = np.clip(diff_display, -vmax, vmax)
        im = ax.imshow(clipped, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        images.append(im)
        for mi in range(diff.shape[0]):
            for ci in range(diff.shape[1]):
                if not mask[mi, ci]:
                    ax.add_patch(mpatches.Rectangle(
                        (ci - 0.5, mi - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor="#9a9a9a",
                        linewidth=0, alpha=0.55, zorder=2,
                    ))
        ax.set_xticks(range(len(td["criteria"])))
        ax.set_xticklabels([short_label(c, prefixed=False) for c in td["criteria"]],
                           rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(td["metrics"])))
        ax.set_yticklabels([metric_label(m) for m in td["metrics"]])
        n_l, n_h = td["n_laaj"], td["n_human"]
        ax.set_title(f"{td['label']}  (LaaJ {n_l}, Human {n_h})",
                     fontsize=9.5, pad=6)

    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(),
                        fraction=0.025, pad=0.025, shrink=0.85)
    cbar.set_label(r"$\log\,LR_{\rm LaaJ}-\log\,LR_{\rm Human}$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
