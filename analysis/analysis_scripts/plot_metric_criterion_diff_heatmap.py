"""
plot_metric_criterion_diff_heatmap.py

Companion to `plot_metric_method_dotplot.py` for §4.2: a single-panel
column-width heatmap showing for each (metric, criterion) cell whether the
pair appears more strongly with LaaJ (orange) or with human evaluation (blue).

Cell value = log(LR_LaaJ) − log(LR_Human), so:
  positive → LaaJ-favored pair
  near 0   → similar association on both sides (or both ~ chance)
  negative → human-favored pair

Cells where neither LaaJ nor Human survives BH-FDR correction (q > 0.05) are
hatched: there is no evidence of association on either side. We otherwise
display the diff regardless of which side is significant, so the reader can
see one-sided pairings (e.g., BARTScore--Naturalness on LaaJ only).

Reads the same JSON corpus as the dot plot.
"""

from __future__ import annotations

import os
import sys
import math
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from data_loader import load_data, short_label
from association_measures import compute_all, bh_fdr

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_criterion_diff_heatmap.png"
OUT_PDF = OUT_DIR / "metric_criterion_diff_heatmap.pdf"
NORMALIZATION_CSV = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"

TOP_METRICS = 14
TOP_CRITERIA = 9
SIG_Q = 0.05
MIN_PAIR_CO_OCC = 8   # require at least this many co-occurrences in the BIGGER subset
                       # (dampens "ghost" cells that are zero on both sides)


def load_metric_display() -> dict[str, str]:
    """{normalized_lower: most-common-original-form}."""
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


def collect_pairs(papers: list, source: str, top_metrics: list[str], top_criteria: list[str]):
    """Return LR matrix + p matrix for the given subset (source ∈ {'laaj','human'})."""
    crit_field = "laaj_criteria" if source == "laaj" else "human_criteria"
    metric_field = "auto_metrics"

    n = len(papers)
    n_m = len(top_metrics)
    n_c = len(top_criteria)
    lr = np.ones((n_m, n_c), dtype=float)
    pv = np.ones((n_m, n_c), dtype=float)

    # Pre-compute per-paper sets for speed
    metric_sets = [{x.lower().strip() for x in (p.get(metric_field) or [])} for p in papers]
    criterion_sets = [{x.lower().strip() for x in (p.get(crit_field) or [])} for p in papers]

    for ci, c in enumerate(top_criteria):
        c_low = c.lower().strip()
        with_crit = [i for i, s in enumerate(criterion_sets) if c_low in s]
        without_crit = [i for i, s in enumerate(criterion_sets) if c_low not in s]
        n_with = len(with_crit)
        n_without = len(without_crit)
        if n_with == 0 or n_without == 0:
            continue
        for mi, m in enumerate(top_metrics):
            m_low = m.lower().strip()
            count_with = sum(1 for i in with_crit if m_low in metric_sets[i])
            count_without = sum(1 for i in without_crit if m_low in metric_sets[i])
            stats = compute_all(count_with, n_with - count_with,
                                count_without, n_without - count_without)
            lr_val = stats.get("lr")
            if lr_val is None or not np.isfinite(lr_val):
                lr_val = 1.0
            lr[mi, ci] = lr_val
            pv[mi, ci] = stats.get("p_value", 1.0)

    return lr, pv


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")

    # Subset by which evaluation method is used
    laaj_papers = [p for p in papers if p.get("laaj_criteria")]
    human_papers = [p for p in papers if p.get("human_criteria")]
    print(f"  LaaJ-using: {len(laaj_papers)}; Human-using: {len(human_papers)}")

    # Metric selection: union of top by frequency in LaaJ subset + top by
    # frequency in Human subset. This keeps generic high-freq metrics in view
    # and surfaces method-specific ones (Win Rate, MoverScore, etc.) that
    # would be missed by a single global frequency rank.
    laaj_metric_counts: Counter = Counter()
    human_metric_counts: Counter = Counter()
    for p in laaj_papers:
        for m in (p.get("auto_metrics") or []):
            laaj_metric_counts[m.lower().strip()] += 1
    for p in human_papers:
        for m in (p.get("auto_metrics") or []):
            human_metric_counts[m.lower().strip()] += 1
    half = TOP_METRICS // 2 + 2
    selected_metrics: list[str] = []
    for m, _ in laaj_metric_counts.most_common(half):
        if m not in selected_metrics:
            selected_metrics.append(m)
    for m, _ in human_metric_counts.most_common(half):
        if m not in selected_metrics:
            selected_metrics.append(m)
    top_metrics = selected_metrics[:TOP_METRICS]

    # Top-K criteria across both LaaJ and Human (rank by total occurrence)
    criterion_counts: Counter = Counter()
    for p in papers:
        for c in (p.get("laaj_criteria") or []):
            criterion_counts[c.strip()] += 1
        for c in (p.get("human_criteria") or []):
            criterion_counts[c.strip()] += 1
    top_criteria = [c for c, _ in criterion_counts.most_common(TOP_CRITERIA)]

    # Compute LR and p matrices on each subset
    lr_l, p_l = collect_pairs(laaj_papers,  "laaj",  top_metrics, top_criteria)
    lr_h, p_h = collect_pairs(human_papers, "human", top_metrics, top_criteria)

    # BH-FDR within each subset
    flat_q_l, _ = bh_fdr(p_l.flatten().tolist())
    flat_q_h, _ = bh_fdr(p_h.flatten().tolist())
    q_l = np.array(flat_q_l).reshape(p_l.shape)
    q_h = np.array(flat_q_h).reshape(p_h.shape)

    # Diff in log-space
    eps = 1e-3
    diff = np.log(np.maximum(lr_l, eps)) - np.log(np.maximum(lr_h, eps))

    # Mask: cells where neither side is significant OR the support is too thin.
    # Also compute observed co-occurrences per cell so we can require minimum
    # support before displaying a directional claim.
    co_l = np.zeros_like(lr_l)
    co_h = np.zeros_like(lr_h)
    metric_sets_l = [{x.lower().strip() for x in (p.get("auto_metrics") or [])} for p in laaj_papers]
    crit_sets_l   = [{x.lower().strip() for x in (p.get("laaj_criteria") or [])} for p in laaj_papers]
    metric_sets_h = [{x.lower().strip() for x in (p.get("auto_metrics") or [])} for p in human_papers]
    crit_sets_h   = [{x.lower().strip() for x in (p.get("human_criteria") or [])} for p in human_papers]
    for mi, m in enumerate(top_metrics):
        for ci, c in enumerate(top_criteria):
            ml = m.lower().strip(); cl = c.lower().strip()
            co_l[mi, ci] = sum(1 for s_m, s_c in zip(metric_sets_l, crit_sets_l) if ml in s_m and cl in s_c)
            co_h[mi, ci] = sum(1 for s_m, s_c in zip(metric_sets_h, crit_sets_h) if ml in s_m and cl in s_c)

    sig_either = (q_l <= SIG_Q) | (q_h <= SIG_Q)
    enough_support = (co_l + co_h) >= MIN_PAIR_CO_OCC
    show_color = sig_either & enough_support

    # Display setup
    metric_display_map = load_metric_display()
    def metric_label(m: str) -> str:
        return metric_display_map.get(m, m.title())

    # Order metrics by total frequency (already sorted)
    metric_labels = [metric_label(m) for m in top_metrics]
    criterion_labels = [short_label(c, prefixed=False) for c in top_criteria]

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, ax = plt.subplots(figsize=(3.5, 3.7))

    # Symmetric color scale around 0
    vmax = max(1.5, np.nanpercentile(np.abs(diff), 95))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # Note RdBu_r: red=positive (LaaJ-favored), blue=negative (Human-favored).
    # We want orange instead of red; use a custom palette.
    from matplotlib.colors import LinearSegmentedColormap
    custom = LinearSegmentedColormap.from_list(
        "laaj_human_diff",
        [(0.0, "#1f77b4"), (0.5, "#ffffff"), (1.0, "#ff7f0e")],
        N=256,
    )
    im.set_cmap(custom)

    # Hatch cells without enough support OR not significant on either side.
    # Also gray out the diff in those cells so the eye focuses on the meaningful signal.
    diff_display = np.where(show_color, diff, 0.0)
    im.set_array(diff_display)
    for mi in range(diff.shape[0]):
        for ci in range(diff.shape[1]):
            if not show_color[mi, ci]:
                ax.add_patch(mpatches.Rectangle(
                    (ci - 0.5, mi - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="#888888", linewidth=0,
                    alpha=0.6, zorder=2,
                ))

    ax.set_xticks(range(len(criterion_labels)))
    ax.set_xticklabels(criterion_labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label(r"$\log\,LR_{\rm LaaJ}-\log\,LR_{\rm Human}$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Evaluation criterion")
    ax.set_ylabel("Automatic metric")

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"\nMetrics ({len(metric_labels)}): {metric_labels}")
    print(f"Criteria ({len(criterion_labels)}): {criterion_labels}")
    print(f"Cells significant on at least one side: {sig_either.sum()} / {sig_either.size}")
    # Surface the most LaaJ-favored and most Human-favored pairs
    flat_diff = [(metric_labels[mi], criterion_labels[ci], diff[mi, ci],
                  q_l[mi, ci], q_h[mi, ci]) for mi in range(diff.shape[0]) for ci in range(diff.shape[1])]
    flat_diff.sort(key=lambda x: -x[2])
    print("\nMost LaaJ-favored cells:")
    for m, c, d, ql, qh in flat_diff[:8]:
        sig = "*" if (ql <= SIG_Q or qh <= SIG_Q) else " "
        print(f"  {sig} {m:18} × {c:38} diff={d:+.2f}  q_L={ql:.3f}  q_H={qh:.3f}")
    print("\nMost Human-favored cells:")
    for m, c, d, ql, qh in flat_diff[-8:]:
        sig = "*" if (ql <= SIG_Q or qh <= SIG_Q) else " "
        print(f"  {sig} {m:18} × {c:38} diff={d:+.2f}  q_L={ql:.3f}  q_H={qh:.3f}")


if __name__ == "__main__":
    main()
