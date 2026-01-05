#!/usr/bin/env python
"""Build a consolidated coverage summary table across tasks.

Reads the per-task paper coverage CSVs and writes:
  - analysis/coverage_summary.csv
  - analysis/coverage_summary.md

Run from project root:
  python nlg-eval-llm/analysis/build_coverage_summary.py
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _summarize(df: pd.DataFrame) -> Dict[str, float]:
    total = int(df["paper_id"].nunique()) if not df.empty else 0
    denom = max(total, 1)

    auto = int(df["has_auto_metrics"].sum()) if not df.empty else 0
    llm = int(df["has_llm_judge"].sum()) if not df.empty else 0
    human = int(df["has_human_eval"].sum()) if not df.empty else 0

    return {
        "total_papers": total,
        "papers_with_auto_metrics": auto,
        "papers_with_auto_metrics_pct": auto / denom,
        "papers_with_llm_judge": llm,
        "papers_with_llm_judge_pct": llm / denom,
        "papers_with_human_eval": human,
        "papers_with_human_eval_pct": human / denom,
    }


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    base_dir = os.path.dirname(__file__)

    inputs: List[Dict[str, str]] = [
        {
            "task": "dialogue",
            "path": os.path.join(
                base_dir, "dialogue_analysis_data", "dialogue_paper_coverage.csv"
            ),
        },
        {
            "task": "mt",
            "path": os.path.join(base_dir, "mt_analysis_data", "mt_paper_coverage.csv"),
        },
        {
            "task": "summarization",
            "path": os.path.join(
                base_dir,
                "summarization_analysis_data",
                "summarization_paper_coverage.csv",
            ),
        },
        {
            "task": "qa",
            "path": os.path.join(base_dir, "qa_analysis_data", "qa_paper_coverage.csv"),
        },
    ]

    rows = []
    coverage_frames = []
    for item in inputs:
        path = item["path"]
        if not os.path.exists(path):
            print(f"Missing input: {path} (skipping)")
            continue
        df = pd.read_csv(path)
        df = df.copy()
        df["task"] = item["task"]
        coverage_frames.append(df)
        stats = _summarize(df)
        stats["task"] = item["task"]
        rows.append(stats)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No inputs found; nothing to write.")
        return

    out_df = out_df[
        [
            "task",
            "total_papers",
            "papers_with_auto_metrics",
            "papers_with_auto_metrics_pct",
            "papers_with_llm_judge",
            "papers_with_llm_judge_pct",
            "papers_with_human_eval",
            "papers_with_human_eval_pct",
        ]
    ]

    task_order = ["dialogue", "mt", "summarization", "qa"]
    out_df["task"] = pd.Categorical(out_df["task"], categories=task_order, ordered=True)
    out_df = out_df.sort_values("task")

    csv_out = os.path.join(base_dir, "coverage_summary.csv")
    out_df.to_csv(csv_out, index=False)
    print(f"Wrote: {csv_out}")

    # Markdown version with percentage formatting
    md_df = out_df.copy()
    for col in [
        "papers_with_auto_metrics_pct",
        "papers_with_llm_judge_pct",
        "papers_with_human_eval_pct",
    ]:
        md_df[col] = md_df[col].map(_fmt_pct)

    md_out = os.path.join(base_dir, "coverage_summary.md")
    with open(md_out, "w", encoding="utf-8") as f:
        headers = list(md_df.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in md_df.iterrows():
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
    print(f"Wrote: {md_out}")

    # Plot: counts by year (y-axis = number of papers)
    if not coverage_frames:
        print("No per-paper coverage inputs found; skipping plot.")
        return

    cov = pd.concat(coverage_frames, ignore_index=True)
    cov = cov.dropna(subset=["paper_id"]).copy()

    cov["year"] = pd.to_numeric(cov["year"], errors="coerce")
    cov = cov.dropna(subset=["year"])
    cov["year"] = cov["year"].astype(int)

    cov["task"] = pd.Categorical(cov["task"], categories=task_order, ordered=True)
    cov = cov.sort_values(["task", "year"])

    yearly = (
        cov.groupby(["task", "year"], observed=True)
        .agg(
            total_papers=("paper_id", "nunique"),
            papers_with_auto_metrics=("has_auto_metrics", "sum"),
            papers_with_llm_judge=("has_llm_judge", "sum"),
            papers_with_human_eval=("has_human_eval", "sum"),
        )
        .reset_index()
    )

    years = sorted(yearly["year"].unique().tolist())
    tasks = task_order

    # Preferred layout: one subplot per signal, with multiple task lines in each.
    task_colors = {
        "dialogue": "#4C72B0",
        "mt": "#55A868",
        "summarization": "#C44E52",
        "qa": "#8172B3",
    }
    task_labels = {
        "dialogue": "Dialogue Generation",
        "mt": "Machine Translation",
        "summarization": "Text Summarization",
        "qa": "Question Answering",
    }

    panels = [
        ("Total papers", "total_papers"),
        ("Papers with automatic metrics", "papers_with_auto_metrics"),
        ("Papers with LaaJ", "papers_with_llm_judge"),
        ("Papers with human eval", "papers_with_human_eval"),
    ]
    pct_for = {
        "total_papers": "total_papers_pct",  # Percentage across all tasks per year
        "papers_with_auto_metrics": "papers_with_auto_metrics_pct",
        "papers_with_llm_judge": "papers_with_llm_judge_pct",
        "papers_with_human_eval": "papers_with_human_eval_pct",
    }

    # Split into two plots: (2,3) and (1,4)
    plot1_panels = [panels[1], panels[2]]  # Auto metrics, LLM-judge
    plot2_panels = [panels[0], panels[3]]  # Total papers, Human eval

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 15,
            "legend.title_fontsize": 14,
        }
    )

    def plot_panels(fig, axes, panels_to_plot):
        """Helper function to plot a set of panels."""
        for ax, (title, col) in zip(axes, panels_to_plot):
            ax_pct = None
            pct_col = pct_for.get(col)
            if pct_col is not None:
                ax_pct = ax.twinx()
                ax_pct.set_ylim(0, 120)
                ax_pct.set_ylabel("% of total", fontsize=12, color='gray')
                ax_pct.set_yticks([0, 25, 50, 75, 100])
                ax_pct.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]], color='gray', fontsize=10)
                for spine in ax_pct.spines.values():
                    spine.set_visible(False)

            max_y = 0
            for task in tasks:
                task_df = yearly[yearly["task"] == task]
                ys = []
                ps = []
                for year in years:
                    match = task_df[task_df["year"] == year]
                    v = int(match[col].iloc[0]) if not match.empty else 0
                    ys.append(v)
                    if pct_col is not None:
                        if col == "total_papers":
                            # For total papers, show percentage across all tasks in that year
                            year_total = yearly[yearly["year"] == year]["total_papers"].sum()
                            ps.append((v / year_total * 100) if year_total else 0.0)
                        else:
                            # For other metrics, show percentage within task
                            denom = int(match["total_papers"].iloc[0]) if not match.empty else 0
                            ps.append((v / denom * 100) if denom else 0.0)
                max_y = max(max_y, max(ys) if ys else 0)

                color = task_colors.get(task, None)
                ax.plot(
                    years,
                    ys,
                    marker="o",
                    linewidth=3.5,
                    markersize=8,
                    color=color,
                    label=task_labels.get(task, task),
                    markeredgewidth=2.0,
                    markeredgecolor='white',
                )
                if ax_pct is not None:
                    ax_pct.plot(
                        years,
                        ps,
                        linestyle="--",
                        linewidth=1.5,
                        color=color,
                        alpha=0.35,
                    )

            ax.set_title(title, fontweight='bold', pad=12)
            ax.set_ylabel("# papers", fontsize=12, fontweight='bold')
            ax.set_xlim(min(years) - 0.2, max(years) + 0.2)
            ax.set_ylim(0, max(5, int(max_y * 1.25)))
            ax.grid(axis="y", alpha=0.35, linewidth=0.8, linestyle='-', color='lightgray')
            ax.grid(axis="x", alpha=0.2, linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ax_pct is not None:
                if col == "total_papers":
                    label_text = "━━ Count\n- - % across tasks"
                else:
                    label_text = "━━ Count\n- - % of total"
                ax.text(
                    0.02,
                    0.98,
                    label_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9),
                )

            # No overall title (user preference)
            ax.set_xticks(years)
            ax.tick_params(axis="x", labelrotation=0)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles and labels:
            fig.legend(
                handles,
                labels,
                # title="Task",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=len(tasks),
                frameon=False,
            )

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    # Plot 1: Auto metrics and LLM-judge
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    axes1 = axes1.flatten().tolist() if isinstance(axes1, np.ndarray) else [axes1]
    fig1.set_dpi(150)
    plot_panels(fig1, axes1, plot1_panels)
    png_out1 = os.path.join(base_dir, "coverage_summary_plot1.png")
    fig1.savefig(png_out1, dpi=300)
    plt.close(fig1)
    print(f"Wrote: {png_out1}")

    # Plot 2: Total papers and Human eval
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    axes2 = axes2.flatten().tolist() if isinstance(axes2, np.ndarray) else [axes2]
    fig2.set_dpi(150)
    plot_panels(fig2, axes2, plot2_panels)
    png_out2 = os.path.join(base_dir, "coverage_summary_plot2.png")
    fig2.savefig(png_out2, dpi=300)
    plt.close(fig2)
    print(f"Wrote: {png_out2}")

    # Plot 3: 2x2 grid layout
    # Top left: Total papers & Auto metrics (merged with dual y-axis)
    # Top right: Papers with LLM-judge
    # Bottom left: Papers with Human eval
    # Bottom right: Percentage adoption (LLM-judge and Human eval)

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig3.set_dpi(150)
    axes3_flat = axes3.flatten()

    # Panel 0: Total papers (count only, no dual y-axis)
    ax0 = axes3_flat[0]
    max_y0 = 0
    for task in tasks:
        task_df = yearly[yearly["task"] == task]

        # Total papers
        ys_total = []
        for year in years:
            match = task_df[task_df["year"] == year]
            v_total = int(match["total_papers"].iloc[0]) if not match.empty else 0
            ys_total.append(v_total)

        max_y0 = max(max_y0, max(ys_total) if ys_total else 0)

        color = task_colors.get(task, None)

        # Plot total papers (solid line with circles)
        ax0.plot(
            years,
            ys_total,
            marker="o",
            linewidth=3.5,
            markersize=8,
            color=color,
            label=task_labels.get(task, task),
            markeredgewidth=2.0,
            markeredgecolor='white',
        )

    ax0.set_title("(a) Total papers", fontweight='bold', pad=15)
    ax0.set_ylabel("# papers", fontsize=15, fontweight='bold')
    ax0.set_xlim(min(years) - 0.2, max(years) + 0.2)
    ax0.set_ylim(0, max(5, int(max_y0 * 1.25)))
    ax0.grid(axis="y", alpha=0.35, linewidth=0.8, linestyle='-', color='lightgray')
    ax0.grid(axis="x", alpha=0.2, linewidth=0.6)
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set_xticks(years)
    ax0.tick_params(axis="x", labelrotation=0)

    # Panel 1: Papers with LLM-judge
    ax1 = axes3_flat[1]
    max_y1 = 0
    for task in tasks:
        task_df = yearly[yearly["task"] == task]
        ys = []
        for year in years:
            match = task_df[task_df["year"] == year]
            v = int(match["papers_with_llm_judge"].iloc[0]) if not match.empty else 0
            ys.append(v)
        max_y1 = max(max_y1, max(ys) if ys else 0)

        color = task_colors.get(task, None)
        ax1.plot(
            years,
            ys,
            marker="o",
            linewidth=3.5,
            markersize=8,
            color=color,
            label=task_labels.get(task, task),
            markeredgewidth=2.0,
            markeredgecolor='white',
        )

    ax1.set_title("(b) Papers with LaaJ", fontweight='bold', pad=15)
    ax1.set_ylabel("# papers", fontsize=15, fontweight='bold')
    ax1.set_xlim(min(years) - 0.2, max(years) + 0.2)
    ax1.set_ylim(0, max(5, int(max_y1 * 1.25)))
    ax1.grid(axis="y", alpha=0.35, linewidth=0.8, linestyle='-', color='lightgray')
    ax1.grid(axis="x", alpha=0.2, linewidth=0.6)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks(years)
    ax1.tick_params(axis="x", labelrotation=0)

    # Panel 2: Papers with Human eval
    ax2 = axes3_flat[2]
    max_y2 = 0
    for task in tasks:
        task_df = yearly[yearly["task"] == task]
        ys = []
        for year in years:
            match = task_df[task_df["year"] == year]
            v = int(match["papers_with_human_eval"].iloc[0]) if not match.empty else 0
            ys.append(v)
        max_y2 = max(max_y2, max(ys) if ys else 0)

        color = task_colors.get(task, None)
        ax2.plot(
            years,
            ys,
            marker="o",
            linewidth=3.5,
            markersize=8,
            color=color,
            label=task_labels.get(task, task),
            markeredgewidth=2.0,
            markeredgecolor='white',
        )

    ax2.set_title("(c) Papers with human eval", fontweight='bold', pad=15)
    ax2.set_ylabel("# papers", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Year", fontsize=15, fontweight='bold')
    ax2.set_xlim(min(years) - 0.2, max(years) + 0.2)
    ax2.set_ylim(0, max(5, int(max_y2 * 1.25)))
    ax2.grid(axis="y", alpha=0.35, linewidth=0.8, linestyle='-', color='lightgray')
    ax2.grid(axis="x", alpha=0.2, linewidth=0.6)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks(years)
    ax2.tick_params(axis="x", labelrotation=0)

    # Panel 3: Percentage adoption - Emphasize METHODS with consistent line styles
    ax3 = axes3_flat[3]

    for task in tasks:
        task_df = yearly[yearly["task"] == task]
        ps_auto = []
        ps_llm = []
        ps_human = []
        for year in years:
            match = task_df[task_df["year"] == year]
            v_total = int(match["total_papers"].iloc[0]) if not match.empty else 0
            v_auto = int(match["papers_with_auto_metrics"].iloc[0]) if not match.empty else 0
            v_llm = int(match["papers_with_llm_judge"].iloc[0]) if not match.empty else 0
            v_human = int(match["papers_with_human_eval"].iloc[0]) if not match.empty else 0
            ps_auto.append((v_auto / v_total * 100) if v_total else 0.0)
            ps_llm.append((v_llm / v_total * 100) if v_total else 0.0)
            ps_human.append((v_human / v_total * 100) if v_total else 0.0)

        color = task_colors.get(task, None)

        # LLM-judge: SOLID BOLD lines (the new standard - upward trend)
        ax3.plot(
            years,
            ps_llm,
            linestyle="-",
            linewidth=4.0,
            marker="o",
            markersize=8,
            color=color,
            label=task_labels.get(task, task),
            markeredgewidth=2.0,
            markeredgecolor='white',
            alpha=0.9,
        )

        # Human eval: DASHED lines (the declining standard)
        ax3.plot(
            years,
            ps_human,
            linestyle="--",
            linewidth=3.5,
            marker="s",
            markersize=7,
            color=color,
            alpha=0.75,
            markeredgewidth=2.0,
            markeredgecolor='white',
        )

        # Auto metrics: DOTTED lines (the background constant)
        ax3.plot(
            years,
            ps_auto,
            linestyle=":",
            linewidth=3.0,
            marker=".",
            markersize=10,
            color=color,
            alpha=0.5,
        )

    ax3.set_title("(d) Evaluation method adoption", fontweight='bold', pad=15)
    ax3.set_ylabel("% of papers", fontsize=15, fontweight='bold')
    ax3.set_xlabel("Year", fontsize=15, fontweight='bold')
    ax3.set_xlim(min(years) - 0.2, max(years) + 0.2)
    ax3.set_ylim(0, 105)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]])
    ax3.set_xticks(years)
    ax3.tick_params(axis="x", labelrotation=0)
    ax3.grid(axis="y", alpha=0.35, linewidth=0.8, linestyle='-', color='lightgray')
    ax3.grid(axis="x", alpha=0.2, linewidth=0.6)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # Add legend box emphasizing line styles for methods
    label_text = "━━ LaaJ\n- - Human eval\n··· Auto metrics"
    ax3.text(
        0.02,
        0.98,
        label_text,
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='lightgray', alpha=0.9),
    )

    # Single legend for all subplots
    handles, labels = axes3_flat[0].get_legend_handles_labels()
    if handles and labels:
        fig3.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(tasks),
            frameon=False,
            fontsize=15,
        )

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    png_out3 = os.path.join(base_dir, "coverage_summary.png")
    fig3.savefig(png_out3, dpi=300)
    plt.close(fig3)
    print(f"Wrote: {png_out3}")




if __name__ == "__main__":
    main()
