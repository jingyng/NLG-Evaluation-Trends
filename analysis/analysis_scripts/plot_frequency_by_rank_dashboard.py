import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
import os
import csv
from data_loader import load_data
from pathlib import Path

def load_normalization_mappings():
    """Load normalization mappings from CSV files"""
    base_dir = Path(__file__).parent.parent / 'metadata_unique_counts'

    mappings = {}
    display_names = {}  # Maps normalized names to their most frequent variant for display

    # Load automatic metrics normalization
    metrics_file = base_dir / 'automatic_metrics_normalization_mapping.csv'
    mappings['auto_metrics'] = {}
    with open(metrics_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['auto_metrics'][row['original'].lower().strip()] = row['normalized']

    # Load automatic metrics merges to get most frequent variant for display
    metrics_merges_file = base_dir / 'automatic_metrics_normalization_merges.csv'
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
    datasets_file = base_dir / 'datasets_normalization_mapping.csv'
    mappings['datasets'] = {}
    with open(datasets_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['datasets'][row['original'].lower().strip()] = row['normalized']

    # Load models normalization
    models_file = base_dir / 'models_normalization_mapping.csv'
    mappings['models'] = {}
    with open(models_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['models'][row['original'].lower().strip()] = row['normalized']

    # Load models merges to get most frequent variant for display
    models_merges_file = base_dir / 'models_normalization_merges.csv'
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
    languages_file = base_dir / 'languages_normalization_mapping.csv'
    mappings['languages'] = {}
    with open(languages_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['languages'][row['original'].lower().strip()] = row['normalized']

    # Load human criteria normalization
    human_file = base_dir / 'human_criteria_normalization_mapping.csv'
    mappings['human_criteria'] = {}
    with open(human_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings['human_criteria'][row['original'].lower().strip()] = row['normalized']

    # Load LLM criteria normalization
    llm_file = base_dir / 'llm_criteria_normalization_mapping.csv'
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
        """Get top N items by frequency for a given task and field"""
        task_papers = get_task_papers_pure(task_name)

        if not task_papers:
            return []

        # Get normalization mapping for this field
        field_mapping = norm_mappings.get(field_key, {})
        # Get display name mapping for this field
        display_mapping = display_mappings.get(field_key, {})

        # Count items with normalization
        counter = Counter()
        for paper in task_papers:
            items = paper.get(field_key, [])
            for item in items:
                item_lower = item.lower().strip()
                # Use normalized name if available, otherwise use original
                normalized = field_mapping.get(item_lower, item)
                counter[normalized] += 1

        # Get top N
        top_items = counter.most_common(top_n)

        # Convert to display names if available
        result = []
        for item, count in top_items:
            # Use display name if available (for metrics, models, laaj_models)
            display_name = display_mapping.get(item, item)
            result.append((display_name, count))

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

            # Get top items by frequency
            top_items = get_top_items_by_frequency(task, field_key, top_n=10)

            # Collect frequencies for statistics
            if top_items:
                all_frequencies.extend([count for _, count in top_items])

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
                items = [item for item, count in top_items]
                counts = [count for item, count in top_items]

                # Reverse order so most frequent is at top
                items = items[::-1]
                counts = counts[::-1]

                # Create horizontal bar chart
                y_pos = np.arange(len(items))
                bars = ax.barh(y_pos, counts, color=category_colors[field_key], alpha=0.8)

                # Add count labels at the end of each bar
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.text(count + max(counts) * 0.02, i, str(count),
                           va='center', fontsize=10, fontweight='bold')

                # Use normalized item names as-is (already properly formatted)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(items, fontsize=11)
                ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Set x-axis limit with some padding
                ax.set_xlim(0, max(counts) * 1.15)

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

    # Calculate statistics
    if all_frequencies:
        min_freq = min(all_frequencies)
        mean_freq = np.mean(all_frequencies)
        max_freq = max(all_frequencies)

        # Create a legend axes at the bottom showing visual bars
        legend_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis('off')

        # Draw example bars for min, mean, max (scaled to max_freq for visualization)
        bar_height = 0.6
        bar_spacing = 0.33

        # Min bar
        min_width = (min_freq / max_freq) * 0.25
        legend_ax.barh(0.5, min_width, height=bar_height, left=0.02,
                      color='#888888', alpha=0.7, edgecolor='black', linewidth=0.5)
        legend_ax.text(0.02 + min_width/2, 0.5, f'Min={min_freq:.0f}',
                      ha='center', va='center', fontsize=11, fontweight='bold')

        # Mean bar
        mean_width = (mean_freq / max_freq) * 0.25
        legend_ax.barh(0.5, mean_width, height=bar_height, left=0.02 + bar_spacing,
                      color='#555555', alpha=0.7, edgecolor='black', linewidth=0.5)
        legend_ax.text(0.02 + bar_spacing + mean_width/2, 0.5, f'Mean={mean_freq:.1f}',
                      ha='center', va='center', fontsize=11, fontweight='bold')

        # Max bar
        max_width = 0.25  # Full width for max
        legend_ax.barh(0.5, max_width, height=bar_height, left=0.02 + 2*bar_spacing,
                      color='#222222', alpha=0.7, edgecolor='black', linewidth=0.5)
        legend_ax.text(0.02 + 2*bar_spacing + max_width/2, 0.5, f'Max={max_freq:.0f}',
                      ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        print(f"\nFrequency Statistics:")
        print(f"  Min:  {min_freq:.0f}")
        print(f"  Mean: {mean_freq:.1f}")
        print(f"  Max:  {max_freq:.0f}")

    # Save figure
    out_path = os.path.join(os.path.dirname(__file__), 'figures',
                           'frequency_by_rank_dashboard.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    papers = load_data()
    print(f"Loaded {len(papers)} papers")
    create_frequency_rank_dashboard(papers)
