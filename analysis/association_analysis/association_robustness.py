#!/usr/bin/env python3
"""
Robustness analysis: compare LR (relative risk) against Dunning G^2
for metric x criterion pairs in both Human and LaaJ evaluation papers.

This script produces the artifacts behind the G^2 + BH-FDR robustness
check referenced in the paper:

  - association_robustness/pairs_human.csv
        One row per (metric, criterion) pair from the human-evaluation
        sub-corpus, with k11/k12/k21/k22, LR, G^2, p-value, q-value.
  - association_robustness/pairs_laaj.csv
        Same for the LaaJ sub-corpus.
  - association_robustness/scatter_lr_vs_g2.pdf
        LR vs. G^2, log-log, color = log(k11), one panel per evaluation type.
  - association_robustness/spearman_by_k11_stratum.csv
        Spearman rho between LR and G^2, stratified by k11.
  - association_robustness/top10_high_lr_low_g2.csv
        Pairs with the largest LR but non-significant G^2 (BH q > 0.05).
  - association_robustness/filter_impact.csv
        Number of pairs surviving each combination of (k11 floor, q threshold).
  - association_robustness/summary.txt
        Human-readable summary, suitable for pasting into the appendix.

Usage
-----
    cd analysis/
    python association_robustness.py
"""

from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from association_measures import compute_all, bh_fdr
from data_loader import load_data, short_label


OUT_DIR = Path(__file__).parent / "association_robustness"

# Filtering thresholds used for the "filter impact" diagnostic.
K11_FLOORS = [1, 3, 5, 10]
Q_THRESHOLDS = [0.05, 0.01, 0.001]

# Minimum frequency thresholds for which terms to even consider as "active".
# We keep these intentionally low for the robustness analysis (we want to
# observe what happens to the long tail), and let the downstream main-text
# rendering apply tighter filters via k11 floor + q threshold.
MIN_METRIC_PAPERS = 10
MIN_CRITERION_PAPERS = 5


def _normalise(items):
    return {x.lower().strip() for x in items if isinstance(x, str) and x.strip()}


def build_pairs_table(papers, criterion_field):
    """Build per-pair contingency counts for a given evaluation type.

    criterion_field: 'human_criteria' or 'laaj_criteria'.
    Returns a pandas DataFrame with columns
      metric, criterion, k11, k12, k21, k22, n_metric, n_criterion, n_total
    restricted to pairs where the metric appears in >= MIN_METRIC_PAPERS papers
    of the relevant sub-corpus and the criterion in >= MIN_CRITERION_PAPERS.
    """
    sub_papers = [p for p in papers if len(p.get(criterion_field, [])) > 0]
    n_total = len(sub_papers)
    if n_total == 0:
        return pd.DataFrame(columns=[
            "metric", "criterion", "k11", "k12", "k21", "k22",
            "n_metric", "n_criterion", "n_total",
        ])

    # Pre-normalise to sets per paper for O(|metrics|+|criteria|) lookups.
    paper_metrics = [_normalise(p.get("auto_metrics", [])) for p in sub_papers]
    paper_criteria = [_normalise(p.get(criterion_field, [])) for p in sub_papers]

    metric_counts = Counter(m for ms in paper_metrics for m in ms)
    criterion_counts = Counter(c for cs in paper_criteria for c in cs)

    metrics = [m for m, c in metric_counts.items() if c >= MIN_METRIC_PAPERS]
    criteria = [c for c, n in criterion_counts.items() if n >= MIN_CRITERION_PAPERS]

    rows = []
    for criterion in criteria:
        n_b = criterion_counts[criterion]
        # Boolean masks per paper: cheaper than re-iterating dicts.
        has_b = np.array([criterion in cs for cs in paper_criteria], dtype=bool)
        for metric in metrics:
            n_a = metric_counts[metric]
            has_a = np.array([metric in ms for ms in paper_metrics], dtype=bool)
            k11 = int(np.sum(has_a & has_b))
            k12 = int(np.sum(has_a & ~has_b))
            k21 = int(np.sum(~has_a & has_b))
            k22 = int(np.sum(~has_a & ~has_b))
            rows.append(
                dict(
                    metric=metric,
                    criterion=criterion,
                    k11=k11,
                    k12=k12,
                    k21=k21,
                    k22=k22,
                    n_metric=n_a,
                    n_criterion=n_b,
                    n_total=n_total,
                )
            )

    return pd.DataFrame(rows)


def annotate_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Add lr, g2, p_value, q_value columns to a pairs table."""
    if df.empty:
        for col in ["lr", "g2", "p_value", "q_value", "reject_q05"]:
            df[col] = []
        return df

    out = compute_all(
        df["k11"].to_numpy(),
        df["k12"].to_numpy(),
        df["k21"].to_numpy(),
        df["k22"].to_numpy(),
        fdr_q=0.05,
    )
    df = df.copy()
    df["lr"] = out["lr"]
    df["g2"] = out["g2"]
    df["p_value"] = out["p_value"]
    df["q_value"] = out["q_value"]
    df["reject_q05"] = out["reject"]
    return df


def spearman_by_stratum(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Spearman rho between LR and G^2, stratified by k11 bucket.

    We exclude pairs with k11 == 0 (LR always equals 1 there) and pairs
    with infinite LR (no support in the denominator) for the correlation.
    """
    if df.empty:
        return pd.DataFrame()
    work = df[(df["k11"] > 0) & np.isfinite(df["lr"])].copy()
    work = work.dropna(subset=["g2"])

    strata = [
        ("all (k11 >= 1)", work),
        ("k11 in [1,2]", work[work["k11"].between(1, 2)]),
        ("k11 in [3,9]", work[work["k11"].between(3, 9)]),
        ("k11 >= 10", work[work["k11"] >= 10]),
    ]

    rows = []
    for name, sub in strata:
        if len(sub) < 5:
            rows.append(dict(eval_type=label, stratum=name, n=len(sub),
                             spearman_lr_g2=np.nan))
            continue
        rho_g2 = sub[["lr", "g2"]].corr(method="spearman").iloc[0, 1]
        rows.append(dict(eval_type=label, stratum=name, n=len(sub),
                         spearman_lr_g2=rho_g2))
    return pd.DataFrame(rows)


def top_high_lr_low_g2(df: pd.DataFrame, label: str, n: int = 10) -> pd.DataFrame:
    """Pairs with the highest LR among those that are NOT significant under G^2.

    'Not significant' = BH q-value > 0.05.
    """
    if df.empty:
        return df
    cand = df[df["q_value"] > 0.05].copy()
    cand = cand[np.isfinite(cand["lr"])]
    cand = cand.sort_values("lr", ascending=False).head(n)
    cand.insert(0, "eval_type", label)
    return cand[[
        "eval_type", "metric", "criterion",
        "k11", "n_metric", "n_criterion", "n_total",
        "lr", "g2", "q_value",
    ]]


def filter_impact(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Number of pairs surviving each (k11 floor, q threshold) combination."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for k_floor in K11_FLOORS:
        for q_t in Q_THRESHOLDS:
            n_pairs = int(((df["k11"] >= k_floor) & (df["q_value"] <= q_t)).sum())
            rows.append(dict(eval_type=label, k11_floor=k_floor,
                             q_threshold=q_t, n_pairs_kept=n_pairs,
                             n_pairs_total=len(df)))
    return pd.DataFrame(rows)


def make_scatter(dfs: dict, x: str, y: str, out_path: Path,
                  log_x: bool = False, log_y: bool = False) -> None:
    """One panel per evaluation type, color = log(k11)."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, dfs.items()):
        sub = df[(df["k11"] > 0) & np.isfinite(df[x]) & np.isfinite(df[y])].dropna(
            subset=[x, y]
        )
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue
        sc = ax.scatter(
            sub[x], sub[y],
            c=sub["k11"].clip(lower=1),
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(sub["k11"].max(), 2)),
            s=14, alpha=0.8, edgecolors="none",
        )
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(x.upper())
        ax.set_ylabel(y.upper())
        ax.set_title(f"{label}  (n={len(sub)} pairs)")
        ax.grid(True, alpha=0.3, linestyle=":")
        # Reference lines.
        if x == "lr":
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.7)
        if y == "g2":
            # chi^2_{1, 0.95} = 3.84; 0.99 = 6.63; 0.999 = 10.83
            for thresh, label_t in [(3.84, "p=.05"), (10.83, "p=.001")]:
                ax.axhline(thresh, color="firebrick", linestyle=":", linewidth=0.7)
                ax.text(ax.get_xlim()[1] * 0.6, thresh, label_t,
                        color="firebrick", fontsize=8, va="bottom")
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("k11 (joint count)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(out_path: Path, summary_pieces: dict) -> None:
    """Render a plain-text summary suitable for pasting into the appendix."""
    lines = []
    lines.append("Robustness of associations to choice of measure")
    lines.append("=" * 60)
    lines.append("")
    for label, df in summary_pieces["pairs"].items():
        lines.append(f"[{label}] {len(df)} (metric, criterion) pairs after "
                     f"frequency floor "
                     f"(metric>={MIN_METRIC_PAPERS}, criterion>={MIN_CRITERION_PAPERS}).")
    lines.append("")
    lines.append("Spearman rho between LR and G^2:")
    lines.append(summary_pieces["spearman"].to_string(index=False, float_format="%.3f"))
    lines.append("")
    lines.append("Filter impact (number of pairs surviving k11 floor and BH q threshold):")
    lines.append(summary_pieces["filter"].to_string(index=False))
    lines.append("")
    lines.append("Top-10 'high LR, non-significant G^2' pairs (BH q > 0.05):")
    lines.append(summary_pieces["top10"].to_string(index=False, float_format="%.3g"))
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading papers...")
    papers = load_data()
    print(f"Loaded {len(papers)} papers.")

    results = {}
    for label, field in [("Human", "human_criteria"), ("LaaJ", "laaj_criteria")]:
        print(f"Building pairs table for {label} ({field})...")
        df = build_pairs_table(papers, field)
        print(f"  {len(df)} candidate pairs.")
        df = annotate_measures(df)
        # Add a short-label column for figure/appendix consumption.
        if not df.empty:
            df.insert(df.columns.get_loc("criterion") + 1,
                      "criterion_short",
                      df["criterion"].map(short_label))
        out_path = OUT_DIR / f"pairs_{label.lower()}.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")
        results[label] = df

    spearman_rows = []
    filter_rows = []
    top10_rows = []
    for label, df in results.items():
        spearman_rows.append(spearman_by_stratum(df, label))
        filter_rows.append(filter_impact(df, label))
        top10_rows.append(top_high_lr_low_g2(df, label))

    spearman_df = pd.concat(spearman_rows, ignore_index=True)
    filter_df = pd.concat(filter_rows, ignore_index=True)
    top10_df = pd.concat(top10_rows, ignore_index=True)

    if not top10_df.empty and "criterion" in top10_df.columns:
        top10_df.insert(top10_df.columns.get_loc("criterion") + 1,
                        "criterion_short",
                        top10_df["criterion"].map(short_label))

    spearman_df.to_csv(OUT_DIR / "spearman_by_k11_stratum.csv", index=False)
    filter_df.to_csv(OUT_DIR / "filter_impact.csv", index=False)
    top10_df.to_csv(OUT_DIR / "top10_high_lr_low_g2.csv", index=False)

    print("Drawing scatter plots...")
    # LR vs G^2: both highly skewed, plot in log-log.
    make_scatter(results, x="lr", y="g2",
                 out_path=OUT_DIR / "scatter_lr_vs_g2.pdf",
                 log_x=True, log_y=True)

    write_summary_text(
        OUT_DIR / "summary.txt",
        {"pairs": results, "spearman": spearman_df,
         "filter": filter_df, "top10": top10_df},
    )

    print(f"\nAll outputs written to {OUT_DIR}")
    print("Recommended next step: read summary.txt and copy the spearman/filter")
    print("tables into Appendix B (Robustness of associations).")


if __name__ == "__main__":
    main()
