"""
plot_metric_method_dotplot.py

§4.2 figure: diverging dot plot showing each metric's LR with LaaJ vs.\\ with
human evaluation, with G²/BH-FDR significance flagged. Replaces the older
scatter (`human_vs_llm_scatter_readable.png`) and the metric-criterion heatmap
in the body — the heatmap stays as an appendix figure.

Design:
- One row per metric, top-N by total occurrence.
- Two paired markers: orange (LaaJ LR) and blue (Human LR), x = LR (log-scale).
- Filled markers = G²+BH-FDR significant (adjusted p ≤ 0.05); open markers otherwise.
- Vertical dashed line at LR=1 (independence).
- Metrics sorted by LaaJ_LR - Human_LR so LaaJ-favored sit at the top,
  generic in the middle, human-favored at the bottom.

Reads precomputed JSONs:
  outputs/figures/metrics_vs_llm_eval/metrics_vs_llm_eval_results.json
  outputs/figures/metrics_vs_human_eval/metrics_vs_human_eval_results.json
(each row already has LR, G², p, BH-FDR-adjusted q, k11, total_count.)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
LAAJ_FILE = BASE / "outputs" / "figures" / "metrics_vs_llm_eval" / "metrics_vs_llm_eval_results.json"
HUMAN_FILE = BASE / "outputs" / "figures" / "metrics_vs_human_eval" / "metrics_vs_human_eval_results.json"
NORMALIZATION_CSV = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"
OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_method_dotplot.png"
OUT_PDF = OUT_DIR / "metric_method_dotplot.pdf"


def load_display_name_map() -> dict[str, str]:
    """{normalized_uppercase: most-common-original-form} from the merges CSV."""
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
            top = variants[0].strip()
            # Strip trailing "(N)" count
            display = top.rsplit("(", 1)[0].strip()
            out[normalized.upper()] = display
    # Manual cleanups for verbose names
    out["ATTACK SUCCESS RATE ASR"] = "Attack Success Rate"
    return out

TOP_FREQ = 12             # always-include: top-K most frequent metrics (generic + popular)
DIVERGENCE_LR = 2.0       # significant pairs with LR >= this on either side count as divergent
MIN_TOTAL = 15            # minimum paper-occurrence floor
LR_FLOOR = 0.1            # clip LR to this for plotting (avoids log(0))
LR_CEIL = 40.0            # clip large outliers
SIG_Q = 0.05              # BH-FDR threshold


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    laaj = load(LAAJ_FILE)["metrics"]
    human = load(HUMAN_FILE)["metrics"]
    metrics = sorted(set(laaj) | set(human))
    display_map = load_display_name_map()
    def display_name(m: str) -> str:
        return display_map.get(m.upper(), m.title())

    # Build per-metric record
    rows = []
    for m in metrics:
        l = laaj.get(m, {})
        h = human.get(m, {})
        l_lr = l.get("enrichment")
        h_lr = h.get("enrichment")
        l_q = l.get("q_value")
        h_q = h.get("q_value")
        l_n = l.get("total_count", 0)
        h_n = h.get("total_count", 0)
        # Skip metrics with no usable LR on either side
        if l_lr is None and h_lr is None:
            continue
        if max(l_n, h_n) < MIN_TOTAL:
            continue
        rows.append({
            "metric": m,
            "laaj_lr": l_lr,
            "laaj_q":  l_q,
            "laaj_n":  l_n,
            "human_lr": h_lr,
            "human_q":  h_q,
            "human_n":  h_n,
        })

    # Selection: union of (a) top-K most frequent and (b) significantly
    # divergent pairs. The frequency tier shows the popular generic metrics
    # in the middle band; the divergence tier surfaces method-specific
    # metrics (e.g., Win Rate for LaaJ, Distinct for Human) regardless of
    # how often they appear overall.
    by_freq = sorted(rows, key=lambda r: -max(r["laaj_n"], r["human_n"]))
    high_freq = by_freq[:TOP_FREQ]

    def is_divergent(r) -> bool:
        l_lr, h_lr = r["laaj_lr"], r["human_lr"]
        l_q, h_q = r["laaj_q"], r["human_q"]
        if l_lr is not None and l_q is not None and l_q <= SIG_Q and l_lr >= DIVERGENCE_LR:
            return True
        if h_lr is not None and h_q is not None and h_q <= SIG_Q and h_lr >= DIVERGENCE_LR:
            return True
        return False
    divergent = [r for r in rows if is_divergent(r)]

    # Deduplicate by metric name (preserve order: high-freq first, then divergent)
    selected: dict[str, dict] = {}
    for r in high_freq + divergent:
        selected.setdefault(r["metric"], r)
    rows = list(selected.values())

    # Sort for display: LaaJ_LR - Human_LR descending so LaaJ-favored at top.
    def diff_key(r):
        l = r["laaj_lr"] if r["laaj_lr"] is not None else 1.0
        h = r["human_lr"] if r["human_lr"] is not None else 1.0
        # Use log space so the diff captures multiplicative divergence symmetrically
        return math.log(max(l, LR_FLOOR)) - math.log(max(h, LR_FLOOR))
    rows.sort(key=diff_key, reverse=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n = len(rows)
    fig_h = 0.30 * n + 1.0           # ~0.30 inch per row + room for axis/legend
    fig, ax = plt.subplots(figsize=(3.3, fig_h))

    LAAJ_C = "#ff7f0e"
    HUMAN_C = "#1f77b4"

    y = list(range(n))
    for i, r in enumerate(rows):
        # Connector line between LaaJ and Human marker for the same metric
        l = max(min(r["laaj_lr"] or 1.0, LR_CEIL), LR_FLOOR)
        h = max(min(r["human_lr"] or 1.0, LR_CEIL), LR_FLOOR)
        ax.plot([l, h], [i, i], color="#cccccc", lw=0.8, zorder=1)

        l_sig = (r["laaj_q"] is not None) and (r["laaj_q"] <= SIG_Q)
        h_sig = (r["human_q"] is not None) and (r["human_q"] <= SIG_Q)

        ax.scatter([l], [i], s=42, color=LAAJ_C if l_sig else "white",
                   edgecolor=LAAJ_C, linewidth=1.2, zorder=3,
                   label="LaaJ (significant)" if i == 0 and l_sig else None)
        ax.scatter([h], [i], s=42, color=HUMAN_C if h_sig else "white",
                   edgecolor=HUMAN_C, linewidth=1.2, zorder=3,
                   label="Human (significant)" if i == 0 and h_sig else None)

    # Reference line at LR=1
    ax.axvline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5, zorder=2)

    # Y axis
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(r["metric"]) for r in rows])
    ax.invert_yaxis()                # top metric on top

    # X axis log scale
    ax.set_xscale("log")
    ax.set_xlim(LR_FLOOR * 0.9, LR_CEIL * 1.1)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.set_xlabel("Likelihood Ratio (LR)")

    # Light shaded zones to indicate "Human-favored" vs "LaaJ-favored"
    # (purely visual aid; LR=1 is the independence line)
    ax.axvspan(LR_FLOOR * 0.9, 1.0, alpha=0.04, color=HUMAN_C, zorder=0)
    ax.axvspan(1.0, LR_CEIL * 1.1, alpha=0.04, color=LAAJ_C, zorder=0)

    # Custom legend (4 entries: 2 colors × significant/not)
    from matplotlib.lines import Line2D
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
    # Place the legend in a clear band above the axes so it doesn't fight the x-axis label.
    ax.legend(handles=legend_handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False, ncol=2, handletextpad=0.4,
              borderpad=0.3, columnspacing=1.2, labelspacing=0.25)

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"\nMetrics in plot ({n} rows):")
    for r in rows:
        l = r["laaj_lr"] or 0
        h = r["human_lr"] or 0
        ls = "*" if r["laaj_q"] is not None and r["laaj_q"] <= SIG_Q else " "
        hs = "*" if r["human_q"] is not None and r["human_q"] <= SIG_Q else " "
        print(f"  {r['metric']:25} LR_LaaJ={l:>5.2f}{ls}  LR_Human={h:>5.2f}{hs}  "
              f"n_LaaJ={r['laaj_n']:>4} n_Human={r['human_n']:>4}")


if __name__ == "__main__":
    main()
