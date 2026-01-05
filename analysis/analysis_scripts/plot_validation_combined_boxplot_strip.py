#!/usr/bin/env python3
"""
Combined visualization: Boxplots + Strip Plot in each cell.
Shows both individual data points (scatter) and summary statistics (boxplot).
"""

import json
import os
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# DATA LOADING (reuse from balloon heatmap)
# ============================================================================

def load_validation_data(data_dir):
    """Load all validation results from normalized JSON files."""
    all_scores = []
    
    json_files = glob.glob(os.path.join(data_dir, "**", "*_validation_normalized.json"), recursive=True)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paper_id = data.get('paper_id', os.path.basename(json_file))
            
            explicit_val = data.get('explicit_validation', {})
            if explicit_val.get('answer', '').lower() != 'yes':
                continue
            
            val_results = data.get('validation_results')
            if val_results is None:
                continue
            
            quant_scores = val_results.get('quantitative_scores', [])
            if not quant_scores:
                continue
            
            for score_entry in quant_scores:
                metric = score_entry.get('metric', 'Unknown')
                value = score_entry.get('value')
                criterion = score_entry.get('criterion', 'Overall')
                
                if value is None:
                    continue
                
                try:
                    val_float = float(value)
                except (ValueError, TypeError):
                    continue
                
                metric_lower = metric.lower()
                is_correlation = any(x in metric_lower for x in ['pearson', 'spearman', 'kendall', 'correlation'])
                is_agreement_metric = any(x in metric_lower for x in ['kappa', 'krippendorff', 'alpha']) and 'percentage' not in metric_lower
                is_agreement_pct = 'agreement' in metric_lower and 'percentage' in metric_lower
                is_accuracy = 'accuracy' in metric_lower or metric_lower == 'acc'
                is_classification = any(x in metric_lower for x in ['f1', 'precision', 'recall'])
                
                # Normalize percentages
                original_value = val_float
                if abs(val_float) > 1.0 and abs(val_float) <= 100:
                    val_float = val_float / 100.0
                elif abs(val_float) > 100:
                    val_float = val_float / 100.0
                
                # Clamp to valid ranges
                if is_correlation or is_agreement_metric:
                    if val_float < -1.0 or val_float > 1.0:
                        print(f"⚠️  WARNING: {paper_id} - {metric} for '{criterion}': "
                              f"value {original_value} → {val_float:.4f} clamped to valid range [-1, 1]")
                    val_float = max(-1.0, min(1.0, val_float))
                elif is_agreement_pct or is_accuracy or is_classification:
                    if val_float < 0.0 or val_float > 1.0:
                        print(f"⚠️  WARNING: {paper_id} - {metric} for '{criterion}': "
                              f"value {original_value} → {val_float:.4f} clamped to valid range [0, 1]")
                    val_float = max(0.0, min(1.0, val_float))
                else:
                    if val_float < 0:
                        if val_float < -1.0 or val_float > 1.0:
                            print(f"⚠️  WARNING: {paper_id} - {metric} for '{criterion}': "
                                  f"value {original_value} → {val_float:.4f} clamped to valid range [-1, 1]")
                        val_float = max(-1.0, min(1.0, val_float))
                    else:
                        if val_float > 1.0:
                            print(f"⚠️  WARNING: {paper_id} - {metric} for '{criterion}': "
                                  f"value {original_value} → {val_float:.4f} clamped to valid range [0, 1]")
                        val_float = max(0.0, min(1.0, val_float))
                
                all_scores.append({
                    'paper_id': paper_id,
                    'metric': metric,
                    'value': val_float,
                    'criterion': criterion if criterion else 'Overall',
                })
        except Exception as e:
            pass
    
    return all_scores


def normalize_metric_name(metric):
    metric_lower = metric.lower().strip()
    if 'pearson' in metric_lower: return 'Pearson r'
    if 'spearman' in metric_lower: return 'Spearman ρ'
    if 'kendall' in metric_lower: return "Kendall's τ"
    if 'fleiss' in metric_lower and 'kappa' in metric_lower: return "Fleiss' κ"
    if 'krippendorff' in metric_lower: return "Krippendorff's α"
    if 'cohen' in metric_lower and 'kappa' in metric_lower: return "Cohen's κ"
    if 'kappa' in metric_lower: return "Cohen's κ"
    if metric_lower in ['accuracy', 'acc']: return 'Accuracy'
    if 'agreement' in metric_lower and 'percentage' in metric_lower: return 'Agreement %'
    if 'agreement with expert majority' in metric_lower: return 'Agree w/ experts'
    return metric.strip()


def normalize_criterion_name(criterion):
    if not criterion: return 'Overall'
    crit_lower = criterion.lower().strip()
    mappings = {
        'overall': 'Overall', 'overall score': 'Overall', 'overall quality': 'Overall',
        'overall aspects': 'Overall',  # Merge Overall Aspects into Overall
        'quality': 'Overall',  # Merge Quality into Overall
        'fluency': 'Fluency', 'coherence': 'Coherence', 'relevance': 'Relevance',
        'correctness': 'Correctness', 'accuracy': 'Accuracy', 'faithfulness': 'Faithfulness',
        'factuality': 'Factuality', 'consistency': 'Consistency',
        'informativeness': 'Informativeness', 'helpfulness': 'Helpfulness',
    }
    return mappings.get(crit_lower, criterion.title())


def categorize_metric(metric):
    metric_lower = metric.lower()
    if any(x in metric_lower for x in ['pearson', 'spearman', 'kendall', 'correlation']): return 'Correlation'
    if any(x in metric_lower for x in ['kappa', 'krippendorff', 'agreement', 'iaa']): return 'Agreement'
    return 'Other'


def compute_matrix(scores):
    """Compute matrix with mean, n, and std for each cell."""
    matrix = defaultdict(lambda: defaultdict(list))
    for s in scores:
        norm_metric = normalize_metric_name(s['metric'])
        norm_crit = normalize_criterion_name(s['criterion'])
        category = categorize_metric(s['metric'])
        if category in ['Correlation', 'Agreement']:
            matrix[norm_metric][norm_crit].append(s['value'])
    return matrix


def create_custom_cmap():
    """Create a custom colormap: red -> yellow -> green."""
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    return LinearSegmentedColormap.from_list('custom_ryg', colors)


# ============================================================================
# COMBINED VISUALIZATION
# ============================================================================


def plot_single_grouped_boxplot(matrix, scores, output_dir):
    """
    Single plot with all criteria grouped by metric.
    All criteria share the same x-axis (metrics) and y-axis (score).
    """
    # Count total data points for filtering
    metric_totals = defaultdict(int)
    criterion_totals = defaultdict(int)

    for metric, criteria in matrix.items():
        for criterion, score_list in criteria.items():
            metric_totals[metric] += len(score_list)
            criterion_totals[criterion] += len(score_list)

    # Filter metrics and criteria with more than 10 total data points
    valid_metrics = [m for m, count in metric_totals.items() if count > 10]
    valid_criteria = [c for c, count in criterion_totals.items() if count > 10]

    # Sort by total count (descending)
    valid_metrics = sorted(valid_metrics, key=lambda x: metric_totals[x], reverse=True)
    valid_criteria = sorted(valid_criteria, key=lambda x: criterion_totals[x], reverse=True)

    # Remove Faithfulness
    valid_criteria = [c for c in valid_criteria if c != 'Faithfulness']

    n_metrics = len(valid_metrics)
    n_criteria = len(valid_criteria)

    # Create a single figure - reduced height
    fig, ax = plt.subplots(figsize=(12, 3.5))

    # Color palette for different metrics
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
    metric_colors = {metric: colors[i % len(colors)] for i, metric in enumerate(valid_metrics)}

    # Prepare grouped positions - now x-axis is criteria, grouped by metrics
    group_width = 0.8
    box_width = group_width / n_metrics
    positions_all = []
    data_all = []
    colors_all = []
    labels_all = []

    for i, criterion in enumerate(valid_criteria):
        for j, metric in enumerate(valid_metrics):
            score_list = matrix[metric].get(criterion, [])
            if len(score_list) > 0:
                # Position: criterion index + offset for metric
                offset = (j - n_metrics / 2 + 0.5) * box_width
                position = i + offset
                positions_all.append(position)
                data_all.append(score_list)
                colors_all.append(metric_colors[metric])
                labels_all.append(f"{metric}\n(n={len(score_list)})")

    # Create boxplots
    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.8,
                   patch_artist=True, showmeans=True,
                   meanprops=dict(marker='o', markerfacecolor='black',
                                 markeredgecolor='white', markersize=5, zorder=3),
                   boxprops=dict(linewidth=1, edgecolor='#333'),
                   whiskerprops=dict(linewidth=1, color='#666'),
                   capprops=dict(linewidth=1, color='#666'),
                   medianprops=dict(linewidth=1.5, color='white', zorder=2),
                   flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.4,
                                  markersize=3, markeredgecolor='none'))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Add individual data points (strip plot) - no jitter, subtle appearance
    for i, (position, data, color) in enumerate(zip(positions_all, data_all, colors_all)):
        if len(data) > 0:
            # No jitter - all points at same x-position
            x_positions = [position] * len(data)

            # Plot individual points - subtle appearance
            ax.scatter(x_positions, data, alpha=0.3, s=12,
                      color='black', edgecolor='none', zorder=10)

    # Styling - reduced font sizes
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.15, 1.05)  # Extended lower range for negative values
    ax.set_xlim(-0.5, n_criteria - 0.5)
    ax.set_xticks(range(n_criteria))
    ax.set_xticklabels(valid_criteria, fontsize=10, fontweight='medium')
    ax.tick_params(axis='y', labelsize=10)  # y-axis tick label size
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('none')

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='#999', linewidth=0.8, linestyle='-', alpha=0.5)

    # Add vertical separator lines between criteria groups
    for i in range(1, n_criteria):
        ax.axvline(x=i - 0.5, color='#666', linewidth=1.5, linestyle='-', alpha=0.3)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Create legend with metrics - placed outside at bottom in two rows
    legend_elements = [mpatches.Patch(facecolor=metric_colors[m],
                                      edgecolor='#333', label=f'{m} (n={metric_totals[m]})', alpha=0.8)
                      for m in valid_metrics]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.10),
             frameon=True, fancybox=True, shadow=False, ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    plt.savefig(os.path.join(output_dir, 'validation_single_grouped_boxplot.png'),
                dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(output_dir, 'validation_single_grouped_boxplot.pdf'),
                bbox_inches='tight', transparent=True)
    print(f"✓ Saved: validation_single_grouped_boxplot.png/pdf")
    plt.close()


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "laaj_human_validation_results_normalized")
    OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "figures", "validation_analysis")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading validation data...")
    scores = load_validation_data(DATA_DIR)
    print(f"Loaded {len(scores)} score entries")

    print("\nComputing matrix...")
    matrix = compute_matrix(scores)

    print("\nGenerating single grouped boxplot...")
    plot_single_grouped_boxplot(matrix, scores, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

