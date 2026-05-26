#!/usr/bin/env python3
"""
Generate Figure laaj_human (validation_single_grouped_boxplot) from the
90-paper HUMAN-ANNOTATED validation subset.

Reference: figure 6 of the paper. Produces analysis/figures/validation_single_grouped_boxplot.png
(which uses LLM-extracted JSONs across the full 433-paper dual-method set).
This script reads the human-annotation Excel, applies the same normalization
and filtering, and renders a figure in the same shape for direct comparability.

Pipeline:
  1. Read 'LaaJ against Human Validation.xlsx' (sheet: yes-no-LaaJ&Human).
  2. Restrict to (paper, annotator) cells with Answer_yes/no == 'Yes'.
  3. Stack all rows from those cells (union across annotators).
  4. Apply value cleaning, metric normalisation, and QCET criterion mapping
     via the existing ~7,000-row catalogue
     (paper_code/05_criteria_normalization/outputs/stage4_classifications_simple_with_overrides.csv).
  5. Dedupe within (paper_id, norm_metric, norm_criterion, round(value, 2)).
  6. Filter to Correlation + Agreement metric families.
  7. Render a grouped boxplot to imgs/validation_single_grouped_boxplot.{png,pdf}.

Intermediate audit data in paper_code/07_figures/validation_analysis_data/:
  validation_human_stack.csv      pre-dedupe rows (audit data for Sec 4.3)
  validation_human_deduped.csv    rows fed to the figure
  validation_human_summary.json   counts/stats for Sec 4.3 and Sec 5.3 text
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----- Paths (HERE-relative, mirroring the convention in this directory) -----

HERE = Path(__file__).resolve().parent  # analysis/analysis_scripts/
REPO_ROOT = HERE.parent.parent           # acl2026-nlg-eval/ (repo root)

XLSX = REPO_ROOT / 'human_annotation' / 'LaaJ against Human Validation.xlsx'
QCET_MAPPING_CSV = (REPO_ROOT / 'analysis' / 'intermediate_results' / 'qcet'
                    / 'stage4_classifications_simple_with_overrides.csv')

OUT_IMGS_DIR = REPO_ROOT / 'analysis' / 'figures'
OUT_PNG = OUT_IMGS_DIR / 'validation_single_grouped_boxplot.png'
OUT_PDF = OUT_IMGS_DIR / 'validation_single_grouped_boxplot.pdf'

# Audit / intermediate data lives next to this script.
DATA_DIR = HERE / 'validation_analysis_data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMGS_DIR.mkdir(parents=True, exist_ok=True)


# ----- Thresholds -----
# n>10 (i.e., n>=11) keeps the long tail of validated constructs visible.
MIN_N_PER_CRITERION = 11
MIN_N_PER_METRIC = 15


# ----- Helpers copied from paper_visualization_code/scripts/plot_validation_combined_boxplot_strip.py
# (inlined to keep this script self-contained within paper_code/). -----

def normalize_metric_name(metric: str) -> str:
    ml = str(metric).lower().strip()
    if 'pearson' in ml: return 'Pearson r'
    if 'spearman' in ml: return 'Spearman ρ'
    if 'kendall' in ml: return "Kendall's τ"
    if 'fleiss' in ml and 'kappa' in ml: return "Fleiss' κ"
    if 'krippendorff' in ml: return "Krippendorff's α"
    if 'cohen' in ml and 'kappa' in ml: return "Cohen's κ"
    if 'kappa' in ml: return "Cohen's κ"
    if ml in ('accuracy', 'acc'): return 'Accuracy'
    if 'agreement' in ml and 'percentage' in ml: return 'Agreement %'
    if 'agreement with expert majority' in ml: return 'Agree w/ experts'
    return str(metric).strip()


def categorize_metric(metric: str) -> str:
    ml = str(metric).lower()
    if any(x in ml for x in ('pearson', 'spearman', 'kendall', 'correlation')):
        return 'Correlation'
    if any(x in ml for x in ('kappa', 'krippendorff', 'agreement', 'iaa')):
        return 'Agreement'
    return 'Other'


def clean_value(value, metric):
    """Mirror the value-cleaning logic in the LLM-side load_validation_data."""
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        return None
    metric_lower = str(metric).lower()
    is_correlation = any(x in metric_lower for x in ['pearson', 'spearman', 'kendall', 'correlation'])
    is_agreement_metric = (
        any(x in metric_lower for x in ['kappa', 'krippendorff', 'alpha'])
        and 'percentage' not in metric_lower
    )
    is_agreement_pct = 'agreement' in metric_lower and 'percentage' in metric_lower
    is_accuracy = 'accuracy' in metric_lower or metric_lower == 'acc'
    is_classification = any(x in metric_lower for x in ['f1', 'precision', 'recall'])

    if abs(val_float) > 1.0:
        val_float = val_float / 100.0

    if is_correlation or is_agreement_metric:
        val_float = max(-1.0, min(1.0, val_float))
    elif is_agreement_pct or is_accuracy or is_classification:
        val_float = max(0.0, min(1.0, val_float))
    else:
        if val_float < 0:
            val_float = max(-1.0, min(1.0, val_float))
        else:
            val_float = max(0.0, min(1.0, val_float))
    return val_float


# ----- QCET lookup (reuses the existing ~7,000-row mapping built for the LLM corpus) -----

_QCET_LOOKUP = None
_TYPO_FIXES = {
    'faithfullness': 'faithfulness',
    'descreptiveness': 'descriptiveness',
    'understaning': 'understanding',
}


def _load_qcet_lookup():
    """Load the existing raw_string -> chosen_name (QCET) mapping. Lowercase-keyed,
    with same-key duplicates resolved by higher occurrence count."""
    df = pd.read_csv(QCET_MAPPING_CSV)
    df['raw_lc'] = df['raw_string'].astype(str).str.strip().str.lower()
    df['occ_total'] = (df['occurrences_llm'].fillna(0)
                       + df['occurrences_human'].fillna(0))
    df = df.sort_values('occ_total', ascending=False).drop_duplicates('raw_lc', keep='first')
    return df.set_index('raw_lc')['chosen_name'].to_dict()


def _apply_typo_fixes(s):
    for k, v in _TYPO_FIXES.items():
        s = s.replace(k, v)
    return s


def _strip_parentheticals(s):
    return re.sub(r'\s*\([^)]*\)', '', s).strip()


def _replace_dashes_slashes(s):
    return s.replace('-', ' ').replace('/', ' ')


def qcet_map_criterion(raw):
    """Map a raw criterion string to its QCET canonical name via the existing
    catalogue. Returns None if the string cannot be resolved through the
    lowercase / parenthetical-strip / dash-or-slash-to-space / typo-fix
    transformations.
    """
    global _QCET_LOOKUP
    if _QCET_LOOKUP is None:
        _QCET_LOOKUP = _load_qcet_lookup()
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    candidates = [
        s,
        _apply_typo_fixes(s),
        _strip_parentheticals(s),
        _apply_typo_fixes(_strip_parentheticals(s)),
        _replace_dashes_slashes(s),
        _apply_typo_fixes(_replace_dashes_slashes(_strip_parentheticals(s))),
    ]
    seen = set()
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if c in _QCET_LOOKUP:
            return _QCET_LOOKUP[c]
    return None


# ----- Matrix building -----

def compute_matrix_qcet(scores):
    """Build the (metric -> criterion -> [values]) matrix. The criterion is
    already QCET-mapped upstream, so no further normalisation here.
    """
    matrix = defaultdict(lambda: defaultdict(list))
    for s in scores:
        norm_metric = normalize_metric_name(s['metric'])
        crit = s['criterion']
        if crit is None:
            continue
        if categorize_metric(s['metric']) in ('Correlation', 'Agreement'):
            matrix[norm_metric][crit].append(s['value'])
    return matrix


# ----- Plotting -----

# Short forms mirror Figure 4 / Figure 5's LABEL_ABBREV (defined in
# plot_task_dashboard_rich_v2.py and plot_metric_criterion_split_heatmap.py)
# so the same criterion gets the same short name across figures.
LABEL_ABBREV = {
    'Correctness of Outputs (outputs as a whole)': 'Correctness of Outputs',
    'Internal Consistency of Outputs': 'Internal Consistency',
    'Overall Quality / Preference': 'Overall Quality',
    'Consistency with Input': 'Input Consistency',
    'Usefulness for Task/Information Need': 'Task Usefulness',
    'Absence of Omissions (relative to input)': 'Absence of Omissions',
    'Control over Style': 'Style Control',
    'Nonredundancy (form)': 'Nonredundancy',
    'Absence of Toxic / Harmful Content': 'Non-Toxicity',
    'Absence of Bias / Stereotypes': 'Non-Bias',
    'Translation Accuracy': 'Translation Acc.',
    'Empathy / Emotional Appropriateness': 'Empathy',
    'Emppathy / Emotional Appropriateness': 'Empathy',
}


def plot_grouped_boxplot_thresholded(matrix, png_path, pdf_path, min_n_criterion,
                                     min_n_metric):
    metric_totals = defaultdict(int)
    criterion_totals = defaultdict(int)
    for metric, criteria in matrix.items():
        for criterion, score_list in criteria.items():
            metric_totals[metric] += len(score_list)
            criterion_totals[criterion] += len(score_list)

    valid_metrics = sorted(
        [m for m, count in metric_totals.items() if count >= min_n_metric],
        key=lambda x: -metric_totals[x],
    )
    valid_criteria = sorted(
        [c for c, count in criterion_totals.items()
         if count >= min_n_criterion and c != 'Other / Unclassifiable'],
        key=lambda x: -criterion_totals[x],
    )
    n_metrics = len(valid_metrics)
    n_criteria = len(valid_criteria)

    print(f'\nFigure: {n_metrics} metrics x {n_criteria} criteria '
          f'(thresholds n>={min_n_metric}, n>={min_n_criterion})')
    print('  Metrics kept:', [(m, metric_totals[m]) for m in valid_metrics])
    print('  Criteria kept:', [(c, criterion_totals[c]) for c in valid_criteria])

    fig, ax = plt.subplots(figsize=(15, 4.0))
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
              '#66a61e', '#e6ab02', '#a6761d', '#666666']
    metric_colors = {m: colors[i % len(colors)] for i, m in enumerate(valid_metrics)}

    group_width = 0.8
    box_width = group_width / max(1, n_metrics)
    positions_all, data_all, colors_all = [], [], []
    for i, criterion in enumerate(valid_criteria):
        for j, metric in enumerate(valid_metrics):
            score_list = matrix[metric].get(criterion, [])
            if score_list:
                offset = (j - n_metrics / 2 + 0.5) * box_width
                positions_all.append(i + offset)
                data_all.append(score_list)
                colors_all.append(metric_colors[metric])

    bp = ax.boxplot(
        data_all, positions=positions_all, widths=box_width * 0.8,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='black',
                       markeredgecolor='white', markersize=5, zorder=3),
        boxprops=dict(linewidth=1, edgecolor='#333'),
        whiskerprops=dict(linewidth=1, color='#666'),
        capprops=dict(linewidth=1, color='#666'),
        medianprops=dict(linewidth=1.5, color='white', zorder=2),
        flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.4,
                        markersize=3, markeredgecolor='none'),
    )
    for patch, color in zip(bp['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for pos, data in zip(positions_all, data_all):
        ax.scatter([pos] * len(data), data, alpha=0.3, s=12,
                   color='black', edgecolor='none', zorder=10)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlim(-0.5, max(0.5, n_criteria - 0.5))

    paren_re = re.compile(r'\s*\([^)]*\)')

    def _abbrev(label):
        if label in LABEL_ABBREV:
            return LABEL_ABBREV[label]
        return paren_re.sub('', label).replace('/', ' / ').strip()

    display_labels = [
        f"{_abbrev(c)} (n={criterion_totals[c]})"
        for c in valid_criteria
    ]
    ax.set_xticks(range(n_criteria))
    ax.set_xticklabels(display_labels, fontsize=10, fontweight='medium',
                       rotation=15, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('none')
    ax.axhline(y=0, color='#999', linewidth=0.8, linestyle='-', alpha=0.5)
    for i in range(1, n_criteria):
        ax.axvline(x=i - 0.5, color='#666', linewidth=1.5, linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    legend_elements = [
        mpatches.Patch(facecolor=metric_colors[m], edgecolor='#333',
                       label=f'{m} (n={metric_totals[m]})', alpha=0.8)
        for m in valid_metrics
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.32), frameon=True, fancybox=True,
              shadow=False, ncol=4, fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(png_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(pdf_path, bbox_inches='tight', transparent=True)
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')
    plt.close()


# ----- xlsx loading -----

def extract_pid(url):
    if not isinstance(url, str):
        return None
    m = re.search(r'aclanthology\.org/([^/]+)\.pdf', url)
    return m.group(1) if m else None


def load_xlsx():
    df = pd.read_excel(XLSX, sheet_name='yes-no-LaaJ&Human', header=None)
    df.columns = df.iloc[2]
    df = df.iloc[3:].reset_index(drop=True)
    df['paper_id'] = df['Paper_link'].apply(extract_pid)
    df = df[df['paper_id'].notna()].copy()
    df['Annotator'] = df['Annotator'].astype(str).str.strip()
    return df


# ----- main -----

def main():
    df = load_xlsx()
    n_unique_papers = df['paper_id'].nunique()

    pa = (df.dropna(subset=['Answer_yes/no'])
            .drop_duplicates(['paper_id', 'Annotator'])
            [['paper_id', 'Annotator', 'Answer_yes/no']])
    pa['is_yes'] = pa['Answer_yes/no'].astype(str).str.strip().str.lower() == 'yes'
    yes_pa_keys = set(map(tuple, pa.loc[pa['is_yes'], ['paper_id', 'Annotator']].values))
    yes_papers = sorted(pa.loc[pa['is_yes'], 'paper_id'].unique().tolist())

    annotators_per_paper = pa.groupby('paper_id').size()
    single_annotator_papers = sorted(
        annotators_per_paper[annotators_per_paper == 1].index.tolist()
    )

    mask = df.apply(lambda r: (r['paper_id'], r['Annotator']) in yes_pa_keys, axis=1)
    rows = df.loc[mask].copy()
    rows = rows[rows[['Metric', 'Value', 'Criterion']].notna().any(axis=1)].copy()
    stacked_total = len(rows)

    rows['metric_norm'] = rows['Metric'].apply(
        lambda m: normalize_metric_name(m) if pd.notna(m) else None
    )
    rows['criterion_norm'] = rows['Criterion'].apply(qcet_map_criterion)
    rows['value_clean'] = [clean_value(v, m) for v, m in zip(rows['Value'], rows['Metric'])]

    qcet_total = rows['criterion_norm'].notna().sum()
    qcet_unmapped = rows['criterion_norm'].isna().sum()
    print(f'QCET mapping: {qcet_total} mapped / {qcet_unmapped} unmapped '
          f'({100 * qcet_total / max(1, qcet_total + qcet_unmapped):.1f}% coverage)')

    rows_clean = rows.dropna(subset=['metric_norm', 'criterion_norm', 'value_clean']).copy()

    stack_cols = ['paper_id', 'Annotator', 'Metric', 'metric_norm', 'Value', 'value_clean',
                  'Criterion', 'criterion_norm', 'LaaJ_Model']
    rows_clean[stack_cols].to_csv(DATA_DIR / 'validation_human_stack.csv', index=False)

    rows_clean['value_r2'] = rows_clean['value_clean'].round(2)
    dedup = rows_clean.drop_duplicates(
        subset=['paper_id', 'metric_norm', 'criterion_norm', 'value_r2'],
        keep='first',
    ).copy()
    dedup[stack_cols + ['value_r2']].to_csv(DATA_DIR / 'validation_human_deduped.csv', index=False)

    scores = [{
        'paper_id': r['paper_id'],
        'metric': r['Metric'],
        'value': r['value_clean'],
        'criterion': r['criterion_norm'],
    } for _, r in dedup.iterrows()]

    matrix = compute_matrix_qcet(scores)
    plot_grouped_boxplot_thresholded(
        matrix, OUT_PNG, OUT_PDF,
        min_n_criterion=MIN_N_PER_CRITERION,
        min_n_metric=MIN_N_PER_METRIC,
    )

    rows_in_figure = sum(len(v) for cdict in matrix.values() for v in cdict.values())
    per_criterion = defaultdict(int)
    per_metric = defaultdict(int)
    per_cell_means = defaultdict(lambda: defaultdict(float))
    for m, cdict in matrix.items():
        for c, vals in cdict.items():
            per_metric[m] += len(vals)
            per_criterion[c] += len(vals)
            per_cell_means[m][c] = float(np.mean(vals)) if vals else None

    summary = {
        'source_xlsx': str(XLSX),
        'unique_papers_total': int(n_unique_papers),
        'unique_yes_papers': len(yes_papers),
        'single_annotator_papers_count': len(single_annotator_papers),
        'single_annotator_papers': single_annotator_papers,
        'rows_stacked_total_yes': int(stacked_total),
        'rows_clean_after_normalization': int(len(rows_clean)),
        'rows_after_dedupe': int(len(dedup)),
        'rows_in_figure_after_family_filter': int(rows_in_figure),
        'avg_stacked_rows_per_yes_paper': round(stacked_total / max(1, len(yes_papers)), 2),
        'avg_clean_rows_per_yes_paper': round(len(rows_clean) / max(1, len(yes_papers)), 2),
        'avg_dedupe_rows_per_yes_paper': round(len(dedup) / max(1, len(yes_papers)), 2),
        'rows_per_metric_in_matrix': dict(sorted(per_metric.items(), key=lambda x: -x[1])),
        'rows_per_criterion_in_matrix': dict(sorted(per_criterion.items(), key=lambda x: -x[1])),
        'cell_means': {m: dict(c) for m, c in per_cell_means.items()},
    }
    with open(DATA_DIR / 'validation_human_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(
        {k: v for k, v in summary.items() if k not in {'cell_means', 'single_annotator_papers'}},
        indent=2,
    ))


if __name__ == '__main__':
    main()
