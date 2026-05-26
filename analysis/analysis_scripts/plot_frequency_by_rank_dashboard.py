import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
import os
import csv
from data_loader import load_data, short_label
from association_measures import compute_all, bh_fdr
from pathlib import Path

# Visual encoding: bars for items that are statistically task-distinctive
# (G²+BH-FDR q ≤ 0.05, LR > 1) render at full alpha; non-significant
# items render at low alpha so the eye picks out "frequent AND
# task-specific" vs "frequent but not distinctive". The "All Tasks"
# column is treated as having no target task, so all its bars render at
# full alpha (no significance dimming there).
SIG_ALPHA = 0.85
NONSIG_ALPHA = 0.25
SIG_Q = 0.05

def load_normalization_mappings():
    """Load normalization mappings from CSV files"""
    base_dir = Path(__file__).parent.parent / 'metadata_unique_counts'

    mappings = {}
    display_names = {}  # Maps normalized names to their most frequent variant for display

    # Load automatic metrics normalization
    metrics_file = base_dir / 'automatic_metrics' / 'automatic_metrics_normalization_mapping.csv'
    mappings['auto_metrics'] = {}
    with open(metrics_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['auto_metrics'][row['original'].lower().strip()] = row['normalized']

    # Load automatic metrics merges to get most frequent variant for display
    metrics_merges_file = base_dir / 'automatic_metrics' / 'automatic_metrics_normalization_merges.csv'
    display_names['auto_metrics'] = {}
    with open(metrics_merges_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = row['normalized']
            variants_str = row['variants_with_counts']
            # Extract first variant (most frequent): "BLEU (2380); ..." -> "BLEU"
            first_variant = variants_str.split(';')[0].strip()
            # Remove count: "BLEU (2380)" -> "BLEU"
            variant_name = first_variant.rsplit('(', 1)[0].strip()
            display_names['auto_metrics'][normalized] = variant_name

    # Load datasets normalization
    datasets_file = base_dir / 'datasets' / 'datasets_normalization_mapping.csv'
    mappings['datasets'] = {}
    with open(datasets_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['datasets'][row['original'].lower().strip()] = row['normalized']

    # Load models normalization
    models_file = base_dir / 'models' / 'models_normalization_mapping.csv'
    mappings['models'] = {}
    with open(models_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['models'][row['original'].lower().strip()] = row['normalized']

    # Load models merges to get most frequent variant for display
    models_merges_file = base_dir / 'models' / 'models_normalization_merges.csv'
    display_names['models'] = {}
    with open(models_merges_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = row['normalized']
            variants_str = row['variants_with_counts']
            # Extract first variant (most frequent)
            first_variant = variants_str.split(';')[0].strip()
            variant_name = first_variant.rsplit('(', 1)[0].strip()
            display_names['models'][normalized] = variant_name

    # Use same models mapping for laaj_models
    mappings['laaj_models'] = mappings['models']
    display_names['laaj_models'] = display_names['models']

    # Load languages normalization
    languages_file = base_dir / 'languages' / 'languages_normalization_mapping.csv'
    mappings['languages'] = {}
    with open(languages_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['languages'][row['original'].lower().strip()] = row['normalized']

    # Load human criteria normalization
    human_file = base_dir / 'criteria' / 'human_criteria_normalization_mapping.csv'
    mappings['human_criteria'] = {}
    with open(human_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['human_criteria'][row['original'].lower().strip()] = row['normalized']

    # Load LLM criteria normalization
    llm_file = base_dir / 'criteria' / 'llm_criteria_normalization_mapping.csv'
    mappings['laaj_criteria'] = {}
    with open(llm_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['laaj_criteria'][row['original'].lower().strip()] = row['normalized']

    return mappings, display_names

def create_frequency_rank_dashboard(papers):
    """
    Create a dashboard showing frequency by rank for different categories across tasks.
    Rows: datasets, models, languages, metrics, laaj criteria, laaj models, human criteria
    Columns: dialogue generation, machine translation, text summarization, question answering, all tasks
    """
    print("Generating Frequency by Rank Dashboard...")

    # Load normalization mappings
    print("Loading normalization mappings...")
    norm_mappings, display_mappings = load_normalization_mappings()
    print(f"Loaded normalization mappings for {len(norm_mappings)} categories")

    # Define top tasks (including 'all tasks' as a special case)
    top_tasks = ['dialogue generation', 'machine translation', 'text summarization', 'question answering', 'all tasks']

    # Define categories to analyze
    categories = [
        ('datasets', 'Datasets'),
        ('models', 'Models'),
        ('languages', 'Languages'),
        ('auto_metrics', 'Automatic Metrics'),
        ('laaj_models', 'LLM-as-a-Judge Models'),
        ('laaj_criteria', 'LLM-as-a-Judge Criteria'),
        ('human_criteria', 'Human Evaluation Criteria')
    ]

    def get_task_papers_pure(task_name):
        """Filter for papers that ONLY discuss this task (or all papers for 'all tasks')"""
        if task_name == 'all tasks':
            # Return all papers with at least one task
            return [p for p in papers if len(p['tasks']) >= 1]
        else:
            # Return papers that ONLY discuss this specific task
            return [p for p in papers
                    if len(p['tasks']) == 1
                    and task_name == p['tasks'][0].lower().strip()]

    def get_top_items_by_frequency(task_name, field_key, top_n=10):
        """Get top N items by frequency for a given task and field.

        Also flags each item as LR-significant (G²+BH-FDR q ≤ SIG_Q and
        LR > 1) so the caller can dim non-significant bars. The "All
        Tasks" column has no target task, so all items there are flagged
        as significant=True (no dimming)."""
        task_papers = get_task_papers_pure(task_name)

        if not task_papers:
            return []

        # Get normalization mapping for this field
        field_mapping = norm_mappings.get(field_key, {})
        # Get display name mapping for this field
        display_mapping = display_mappings.get(field_key, {})

        # Count items with normalization (per-paper deduplicated)
        counter = Counter()
        for paper in task_papers:
            items = set()
            for item in (paper.get(field_key) or []):
                item_lower = item.lower().strip()
                normalized = field_mapping.get(item_lower, item)
                items.add(normalized)
            for it in items:
                counter[it] += 1

        # Get top N
        top_items = counter.most_common(top_n)

        # Compute LR + significance for the top items (skip for "all tasks")
        if task_name == "all tasks":
            sig_flags = [True] * len(top_items)
            lrs_for_items = [None] * len(top_items)
        else:
            other_papers = [p for p in papers if p not in task_papers]
            n_task = len(task_papers)
            n_other = len(other_papers)
            other_counter = Counter()
            for paper in other_papers:
                items = set()
                for item in (paper.get(field_key) or []):
                    item_lower = item.lower().strip()
                    normalized = field_mapping.get(item_lower, item)
                    items.add(normalized)
                for it in items:
                    other_counter[it] += 1

            lrs, pvals = [], []
            for internal, k_w in top_items:
                k_o = other_counter.get(internal, 0)
                stats = compute_all(k_w, n_task - k_w, k_o, n_other - k_o)
                lr = stats.get("lr")
                p = stats.get("p_value", 1.0)
                lrs.append(lr if lr is not None and np.isfinite(lr) else 0.0)
                pvals.append(p)
            qvals, _ = bh_fdr(pvals)
            sig_flags = [q <= SIG_Q and lr > 1.0 for q, lr in zip(qvals, lrs)]
            lrs_for_items = lrs

        # Convert to display names if available
        result = []
        for (internal, count), sig, lr in zip(top_items, sig_flags, lrs_for_items):
            display_name = display_mapping.get(internal, internal)
            if field_key in ('laaj_criteria', 'human_criteria'):
                display_name = short_label(display_name)
            result.append((display_name, count, sig, lr))

        return result

    # Setup plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(7, 5, figsize=(30, 35))

    # Color palettes for each category
    category_colors = {
        'datasets': '#4e79a7',
        'models': '#f28e2b',
        'languages': '#e15759',
        'auto_metrics': '#76b7b2',
        'laaj_models': '#b07aa1',
        'laaj_criteria': '#59a14f',
        'human_criteria': '#edc948'
    }

    # Collect all frequencies for statistics
    all_frequencies = []

    # Process each category (row)
    for row_idx, (field_key, category_name) in enumerate(categories):

        # Process each task (column)
        for col_idx, task in enumerate(top_tasks):
            ax = axes[row_idx, col_idx]

            # Get top items by frequency, each annotated with an LR
            # significance flag and the LR value itself.
            top_items = get_top_items_by_frequency(task, field_key, top_n=10)

            # Collect frequencies for statistics
            if top_items:
                all_frequencies.extend([count for _, count, _, _ in top_items])

            if not top_items:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            else:
                # Prepare data for horizontal bar chart
                items = [item for item, _, _, _ in top_items]
                counts = [count for _, count, _, _ in top_items]
                sigs = [sig for _, _, sig, _ in top_items]
                lrs = [lr for _, _, _, lr in top_items]

                # Reverse order so most frequent is at top
                items = items[::-1]
                counts = counts[::-1]
                sigs = sigs[::-1]
                lrs = lrs[::-1]

                # Create horizontal bar chart with per-bar alpha
                # (full for LR-significant, dimmed for non-significant)
                y_pos = np.arange(len(items))
                bars = ax.barh(y_pos, counts,
                               color=category_colors[field_key])
                for bar, sig in zip(bars, sigs):
                    bar.set_alpha(SIG_ALPHA if sig else NONSIG_ALPHA)

                # Label format: "count (LR=X.X)" for task-specific
                # columns; "count" only for the "All Tasks" column where
                # LR is undefined (lr is None there). Label opacity
                # matches the bar opacity so non-significant rows visually
                # recede.
                for i, (count, sig, lr) in enumerate(zip(counts, sigs, lrs)):
                    if lr is None:
                        label = str(count)
                    elif lr >= 100:
                        label = f"{count}  (LR={lr:.0f})"
                    else:
                        label = f"{count}  (LR={lr:.1f})"
                    ax.text(count + max(counts) * 0.02, i, label,
                            va='center', fontsize=10, fontweight='bold',
                            alpha=1.0 if sig else 0.45)

                # Use normalized item names as-is (already properly formatted)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(items, fontsize=11)
                ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Set x-axis limit with some padding
                # Widened x-margin to fit "count (LR=X.X)" text labels.
                ax.set_xlim(0, max(counts) * 1.35)

            # Add title for first row (task names)
            if row_idx == 0:
                n_papers = len(get_task_papers_pure(task))
                task_display = "All Tasks" if task == 'all tasks' else task.title()
                ax.set_title(f"{task_display}\n(N={n_papers})",
                           fontsize=16, fontweight='bold', pad=15)

            # Add y-axis label for first column (category names)
            if col_idx == 0:
                ax.set_ylabel(category_name, fontsize=14, fontweight='bold',
                            rotation=90, labelpad=15)

    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Leave space at bottom for legend

    # Bottom legend: explain the bar-opacity encoding (full = LR-significant,
    # dim = not significant). The "All Tasks" column has no target task and
    # thus no significance test, so all bars there render at full opacity.
    legend_ax = fig.add_axes([0.10, 0.01, 0.80, 0.025])
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis('off')
    # Full-alpha sample bar + label
    legend_ax.barh(0.5, 0.07, height=0.55, left=0.02,
                   color='#4e79a7', alpha=SIG_ALPHA,
                   edgecolor='black', linewidth=0.5)
    legend_ax.text(0.10, 0.5,
                   "Full opacity = LR-significant for this task "
                   "(G²+BH-FDR q≤0.05, LR>1).",
                   ha='left', va='center', fontsize=11)
    # Dim sample bar + label
    legend_ax.barh(0.5, 0.07, height=0.55, left=0.55,
                   color='#4e79a7', alpha=NONSIG_ALPHA,
                   edgecolor='black', linewidth=0.5)
    legend_ax.text(0.63, 0.5,
                   "Dimmed = top-ranked by frequency but not "
                   "task-distinctive.",
                   ha='left', va='center', fontsize=11, alpha=0.55)

    # Calculate statistics (for the printed summary below)
    if all_frequencies:
        min_freq = min(all_frequencies)
        mean_freq = np.mean(all_frequencies)
        max_freq = max(all_frequencies)

        print(f"\nFrequency Statistics:")
        print(f"  Min:  {min_freq:.0f}")
        print(f"  Mean: {mean_freq:.1f}")
        print(f"  Max:  {max_freq:.0f}")

    # Save figure
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures',
                           'frequency_dashboard', 'frequency_by_rank_dashboard.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    papers = load_data()
    print(f"Loaded {len(papers)} papers")
    create_frequency_rank_dashboard(papers)
