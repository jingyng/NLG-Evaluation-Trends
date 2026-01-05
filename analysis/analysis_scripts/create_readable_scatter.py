#!/usr/bin/env python3
"""
Create an improved, more readable scatter plot comparing human vs LLM evaluation enrichment.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).parent.parent
HUMAN_FILE = BASE / "analysis" / "figures" / "metrics_vs_human_eval" / "metrics_vs_human_eval_results.json"
LLM_FILE = BASE / "analysis" / "figures" / "metrics_vs_llm_eval" / "metrics_vs_llm_eval_results.json"
OUTPUT_DIR = BASE / "analysis" / "figures" / "human_vs_llm_comparison"
NORMALIZATION_CSV = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"


def load_normalization_mapping():
    """Load mapping from normalized metric names to their most common variant."""
    mapping = {}

    with open(NORMALIZATION_CSV, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                normalized = parts[0]
                variants_with_counts = parts[3]

                # Extract first variant (most common)
                # Format: "BLEU (2380); BLEU-C (1); ..."
                first_variant = variants_with_counts.split(';')[0].strip()
                # Remove count in parentheses: "BLEU (2380)" -> "BLEU"
                variant_name = first_variant.rsplit('(', 1)[0].strip()

                # Map normalized name (uppercase) to most common variant
                mapping[normalized.upper()] = variant_name

    return mapping


def load_data():
    """Load both human and LLM evaluation results."""
    with open(HUMAN_FILE) as f:
        human_data = json.load(f)
    with open(LLM_FILE) as f:
        llm_data = json.load(f)
    return human_data, llm_data


def create_comparison_dataframe(human_data, llm_data, normalization_map, min_total=10):
    """Create a comparison DataFrame."""
    all_metrics = set(human_data["metrics"].keys()) | set(llm_data["metrics"].keys())

    rows = []
    for metric in all_metrics:
        human_stats = human_data["metrics"].get(metric, {})
        llm_stats = llm_data["metrics"].get(metric, {})

        total = max(
            human_stats.get("total_count", 0),
            llm_stats.get("total_count", 0)
        )

        if total > min_total:
            # Check if metric exists in each dataset
            metric_in_human = metric in human_data["metrics"]
            metric_in_llm = metric in llm_data["metrics"]
            
            # Get enrichment values
            human_enrich = human_stats.get("enrichment")
            llm_enrich = llm_stats.get("enrichment")
            
            # Check if enrichment is None due to 100% usage (count_without = 0)
            # This means the metric is used in ALL papers of that type, indicating very strong association
            human_pct_without = human_stats.get("pct_without_human", 1.0)
            llm_pct_without = llm_stats.get("pct_without_llm", 1.0)
            
            # Handle metrics that are only in one dataset
            # For log scale, use 0.3 as placeholder (visible but indicates missing)
            if not metric_in_human:
                # Metric only in LLM - set human enrichment to 0.3 (visible but indicates missing)
                if llm_enrich is None or llm_enrich == float('inf'):
                    continue
                # Allow 0.0 values - they indicate no enrichment, which is valid
                if llm_enrich == 0 or llm_enrich == 0.0:
                    llm_enrich = 0.3  # Use 0.3 instead of 0.0 for log scale
                human_enrich = 0.3  # Placeholder at axis minimum
            elif not metric_in_llm:
                # Metric only in Human - set LLM enrichment to 0.3 (visible but indicates missing)
                if human_enrich is None or human_enrich == float('inf'):
                    continue
                # Allow 0.0 values - they indicate no enrichment, which is valid
                if human_enrich == 0 or human_enrich == 0.0:
                    human_enrich = 0.3  # Use 0.3 instead of 0.0 for log scale
                llm_enrich = 0.3  # Placeholder at axis minimum
            else:
                # Metric in both - check for valid values
                # If enrichment is None, check if it's due to 100% usage (very strong association)
                # or if it's truly missing
                if human_enrich is None:
                    if human_pct_without == 0.0:
                        # Used in 100% of Human papers - very strong association, use high value
                        human_enrich = 20.0  # High value to indicate 100% usage
                    else:
                        # Truly missing, use placeholder
                        human_enrich = 0.3
                if llm_enrich is None:
                    if llm_pct_without == 0.0:
                        # Used in 100% of LLM papers - very strong association, use high value
                        llm_enrich = 20.0  # High value to indicate 100% usage
                    else:
                        # Truly missing, use placeholder
                        llm_enrich = 0.3
                
                if human_enrich == float('inf') or llm_enrich == float('inf'):
                    continue
                # Replace 0.0 with 0.3 for log scale compatibility
                if human_enrich == 0 or human_enrich == 0.0:
                    human_enrich = 0.3
                if llm_enrich == 0 or llm_enrich == 0.0:
                    llm_enrich = 0.3
            
            # Use the most common variant instead of normalized name
            display_name = normalization_map.get(metric.upper(), metric)

            rows.append({
                "Metric": display_name,
                "Human Enrichment": human_enrich,
                "LLM Enrichment": llm_enrich,
                "Total": total,
            })

    df = pd.DataFrame(rows)
    return df


def categorize_metrics_simplified(df):
    """Categorize metrics with simplified categories for better visualization."""
    categories = {}

    for _, row in df.iterrows():
        metric = row["Metric"]
        human_e = row["Human Enrichment"]
        llm_e = row["LLM Enrichment"]

        # 4-quadrant categorization (2x2 grid: Low/High x Low/High)
        # Threshold at 2.0
        human_high = human_e >= 2.0
        llm_high = llm_e >= 2.0

        # 4 combinations
        if human_high and llm_high:
            category = "High-High"
        elif human_high and not llm_high:
            category = "High-Low"
        elif not human_high and llm_high:
            category = "Low-High"
        else:
            category = "Low-Low"

        categories[metric] = category

    df["Category"] = df["Metric"].map(categories)
    return df


def create_improved_scatter(df, output_dir):
    """Create an improved, readable scatter plot."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Cap extreme values for plotting (only upper bound, let lower values show naturally)
    df_plot = df.copy()
    df_plot["Human Plot"] = df_plot["Human Enrichment"].apply(lambda x: min(x, 22))
    df_plot["LLM Plot"] = df_plot["LLM Enrichment"].apply(lambda x: min(x, 22))
    
    # Calculate correlation statistics
    # NOTE: We keep the filter here because correlation requires finite numbers
    # Filter out extreme values (>= 900) which are placeholders for infinite or invalid values
    valid_mask = (df_plot["Human Enrichment"] < 900) & (df_plot["LLM Enrichment"] < 900)
    
    print(f"\nCorrelation calculation details:")
    print(f"  Total metrics in plot: {len(df_plot)}")
    print(f"  Metrics with both enrichments < 900: {valid_mask.sum()}")
    print(f"  Metrics excluded (>= 900): {(~valid_mask).sum()}")
    
    if valid_mask.sum() > 2:
        pearson_r, pearson_p = pearsonr(
            df_plot.loc[valid_mask, "Human Enrichment"], 
            df_plot.loc[valid_mask, "LLM Enrichment"]
        )
        spearman_r, spearman_p = spearmanr(
            df_plot.loc[valid_mask, "Human Enrichment"], 
            df_plot.loc[valid_mask, "LLM Enrichment"]
        )
        print(f"  Final n for correlation: {valid_mask.sum()}")
    else:
        pearson_r, spearman_r = 0, 0
        print(f"  Not enough valid data points for correlation (n={valid_mask.sum()})")

    # Set up the plot with publication-friendly size (optimized for single-column)
    # Set professional font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Nature Style: Clean white background with no colored zones
    # Zones will be created implicitly with dashed grid lines

    # Define 4 quadrant categories
    category_order = [
        "High-High", "High-Low",
        "Low-High", "Low-Low"
    ]

    # Define key metrics to highlight (The Protagonists) - moved here for earlier access
    # First, identify human-preferred metrics (below diagonal, Human > 5)
    human_preferred = df_plot[(df_plot["Human Plot"] > df_plot["LLM Plot"]) &
                              (df_plot["Human Plot"] >= 5.0)]

    print(f"\nHuman-preferred metrics (below diagonal, Human association > 5): {len(human_preferred)}")
    if len(human_preferred) > 0:
        print("  Metric Name           | Human Assoc | LLM Assoc")
        print("  " + "-"*55)
        for _, row in human_preferred.sort_values('Human Plot', ascending=False).iterrows():
            print(f"  {row['Metric']:20s} | {row['Human Plot']:11.2f} | {row['LLM Plot']:9.2f}")

    # Select representative human-preferred metrics - include most frequent + diverse levels
    human_preferred_to_label = set()
    if len(human_preferred) > 0:
        sorted_hp = human_preferred.sort_values('Human Plot', ascending=False)

        # First, add top 3 most frequent
        top_frequent = human_preferred.nlargest(3, 'Total')
        selected = top_frequent['Metric'].tolist()

        print(f"\nTop 3 most frequent human-preferred metrics:")
        for _, row in top_frequent.iterrows():
            print(f"  {row['Metric']:25s} | H:{row['Human Plot']:6.2f} L:{row['LLM Plot']:6.2f} | n={row['Total']:3.0f}")

        # Then add metrics at different H levels for diversity (if not already included)
        targets = [
            (20, 'Kendall\'s Tau'),  # H=20, but higher LLM (6.84) - more visible
            (10, 'Spearman\'s ρ'),    # H=10.58 - mid range
        ]

        for h_level, target_name in targets:
            if target_name not in selected:
                matches = sorted_hp[sorted_hp['Metric'] == target_name]
                if len(matches) > 0:
                    selected.append(target_name)

        human_preferred_to_label.update(selected)

        print(f"\nAll selected human-preferred metrics for labeling:")
        for m in selected:
            row = sorted_hp[sorted_hp['Metric'] == m].iloc[0]
            print(f"  {m:25s} | H:{row['Human Plot']:6.2f} L:{row['LLM Plot']:6.2f} | n={row['Total']:3.0f}")

    key_metrics = {
        # The Old Guard - reduced
        'BLEU', 'ROUGE', 'METEOR',
        # The New Wave - reduced
        'G-Eval', 'BERTScore',
        # The Outliers
        'Win Rate', 'ATTACK SUCCESS RATE ASR',
        # Additional important ones for context - reduced
        'F1', 'Accuracy',
    }

    # Add human-preferred metrics - these are prioritized
    key_metrics.update(human_preferred_to_label)
    print(f"\nAdded {len(human_preferred_to_label)} human-preferred metrics to labels: {human_preferred_to_label}")
    print(f"Total metrics to label: {len(key_metrics)}")

    # Nature Style: Simplified 3-color scheme
    # Determine if metric is in generic/inertia zone (0.5-2.0 on both axes)
    generic_mask = (df_plot["Human Plot"] >= 0.5) & (df_plot["Human Plot"] <= 2.0) & \
                   (df_plot["LLM Plot"] >= 0.5) & (df_plot["LLM Plot"] <= 2.0)

    # Simplified coloring:
    # 1. Grey: Generic (both Human AND LLM association < 5)
    # 2. Red: Above diagonal (LaaJ-favored)
    # 3. Blue: Below diagonal (Human-favored)
    import numpy as np

    generic_color_mask = (df_plot["Human Plot"] < 5.0) & (df_plot["LLM Plot"] < 5.0)

    # For non-generic metrics, determine if above or below diagonal
    above_diagonal = df_plot["LLM Plot"] > df_plot["Human Plot"]
    below_diagonal = df_plot["LLM Plot"] < df_plot["Human Plot"]

    # Colors
    red_color = '#D62728'  # Red for above diagonal (LaaJ-favored)
    blue_color = '#1F77B4'  # Blue for below diagonal (Human-favored)
    grey_color = '#7F7F7F'  # Grey for Generic

    # Plot UNLABELED points first - all filled
    for color_name, color_mask, color in [
        ('grey', generic_color_mask, grey_color),
        ('red', (~generic_color_mask) & above_diagonal, red_color),
        ('blue', (~generic_color_mask) & below_diagonal, blue_color)
    ]:
        mask = (~df_plot["Metric"].isin(key_metrics)) & color_mask
        if mask.sum() > 0:
            ax.scatter(
                df_plot.loc[mask, "Human Plot"],
                df_plot.loc[mask, "LLM Plot"],
                alpha=0.4,
                s=df_plot.loc[mask, "Total"] * 2.0,  # Reduced for single-column layout
                color=color,
                marker='o',
                edgecolors='none',
                zorder=1
            )

    # Plot LABELED points on top - all filled
    for color_name, color_mask, color in [
        ('grey', generic_color_mask, grey_color),
        ('red', (~generic_color_mask) & above_diagonal, red_color),
        ('blue', (~generic_color_mask) & below_diagonal, blue_color)
    ]:
        mask = (df_plot["Metric"].isin(key_metrics)) & color_mask
        if mask.sum() > 0:
            ax.scatter(
                df_plot.loc[mask, "Human Plot"],
                df_plot.loc[mask, "LLM Plot"],
                alpha=0.8,
                s=df_plot.loc[mask, "Total"] * 3.0,  # Reduced for single-column layout
                color=color,
                marker='o',
                edgecolors='#333333',
                linewidth=1.0,  # Reduced for single-column layout
                zorder=3
            )

    # Nature Style: Diagonal line for equality - simple black dashed
    ax.plot([0.3, 30], [0.3, 30], 'k--', alpha=0.4, linewidth=1.0, zorder=2)

    # Nature Style: Dashed grid lines at key thresholds
    # LR=1 (Neutral) - lighter dashed lines
    ax.axvline(x=1.0, color='#AAAAAA', linestyle='--', alpha=0.5, linewidth=0.8, zorder=1)
    ax.axhline(y=1.0, color='#AAAAAA', linestyle='--', alpha=0.5, linewidth=0.8, zorder=1)
    # LR=5 (Strong) - darker dashed lines
    ax.axvline(x=5.0, color='#666666', linestyle='--', alpha=0.6, linewidth=1.0, zorder=1)
    ax.axhline(y=5.0, color='#666666', linestyle='--', alpha=0.6, linewidth=1.0, zorder=1)

    # De-cluttered labeling: Only show key metrics
    important_metrics = []
    labeled_metrics = set()

    # Debug: Check which human-preferred metrics are actually in df_plot
    print(f"\nDEBUG: Checking human-preferred metrics in df_plot:")
    for hpm in human_preferred_to_label:
        if hpm in df_plot["Metric"].values:
            row = df_plot[df_plot["Metric"] == hpm].iloc[0]
            print(f"  ✓ {hpm:25s} | H:{row['Human Plot']:6.2f} L:{row['LLM Plot']:6.2f}")
        else:
            print(f"  ✗ {hpm:25s} | NOT FOUND IN DF_PLOT")

    # Only label metrics in the key set (key_metrics defined earlier)
    for _, row in df_plot.iterrows():
        metric = row["Metric"]
        if metric in key_metrics:
            x = row["Human Plot"]
            y = row["LLM Plot"]
            important_metrics.append((metric, x, y))
            labeled_metrics.add(metric)

    print(f"\nDEBUG: Total metrics to be labeled: {len(important_metrics)}")
    print(f"DEBUG: Human-preferred metrics in label list: {len([m for m in labeled_metrics if m in human_preferred_to_label])}")

    # Use adjustText for better label placement if available, otherwise smart manual placement
    try:
        from adjustText import adjust_text
        import matplotlib.patheffects as pe

        texts = []
        for metric, x, y in important_metrics:
            # For points on the right edge, position label to the left
            ha = 'right' if x > 5 else 'center'
            text = ax.text(x, y, metric, fontsize=10, fontweight='600',  # Reduced for single-column
                          ha=ha, color='#222222', zorder=10)
            # Add white stroke/halo for readability
            text.set_path_effects([
                pe.withStroke(linewidth=2.5, foreground='white', alpha=0.9),  # Reduced for single-column
                pe.Normal()
            ])
            texts.append(text)

        # More aggressive parameters to keep labels within plot bounds
        # Add explicit axis limits to prevent labels from moving outside
        adjust_text(texts,
                   ax=ax,
                   arrowprops=dict(arrowstyle='->', color='#666666', lw=0.4, alpha=0.6),
                   expand_points=(1.8, 1.8), expand_text=(1.5, 1.5),
                   lim=2000, force_text=(0.8, 0.8), force_points=(0.5, 0.5),
                   only_move={'points':'', 'text':'xy', 'objects':'xy'},
                   # Force labels to stay within axis limits
                   ensure_inside_axes=True)
    except ImportError:
        # Smart manual labeling with position-based offsets
        import numpy as np
        for i, (metric, x, y) in enumerate(important_metrics):
            angle = (i * 45) % 360  # Rotate through different angles
            # Larger radius for edge points to ensure visibility
            if x > 6 or y > 8 or x < 1 or y < 1:
                radius = 22  # Larger offset for edge points
            else:
                radius = 18  # Standard offset for center points

            angle_rad = np.radians(angle)
            dx = radius * np.cos(angle_rad)
            dy = radius * np.sin(angle_rad)

            # More aggressive left placement for right-side points
            if x > 5:  # Right side - FORCE LEFT placement
                dx = -abs(dx) - 40  # Extra offset to the left
            elif x < 1:  # Left side - prefer RIGHT (toward center)
                dx = abs(dx)

            if y > 8:  # Top - prefer DOWN (toward center)
                dy = -abs(dy)
            elif y < 1:  # Bottom - prefer UP (toward center)
                dy = abs(dy)

            import matplotlib.patheffects as pe
            text = ax.annotate(metric, (x, y), fontsize=9, fontweight='600',
                              xytext=(dx, dy), textcoords='offset points',
                              color='#222222', zorder=10,
                              arrowprops=dict(arrowstyle='->', color='#666666', lw=0.4, alpha=0.6))
            # Add white stroke/halo
            text.set_path_effects([
                pe.withStroke(linewidth=2, foreground='white', alpha=0.9),
                pe.Normal()
            ])

    # Nature Style: Minimal zone labels - very subtle, small, grey
    # Only add subtle threshold labels next to grid lines
    ax.text(1.0, 0.32, 'LR=1', fontsize=8, alpha=0.5, ha='center', color='#666666')  # Reduced for single-column
    ax.text(5.0, 0.32, 'LR=5', fontsize=8, alpha=0.5, ha='center', color='#666666')
    ax.text(0.32, 1.0, 'LR=1', fontsize=8, alpha=0.5, va='center', rotation=90, color='#666666')
    ax.text(0.32, 5.0, 'LR=5', fontsize=8, alpha=0.5, va='center', rotation=90, color='#666666')

    # Styling - improved axis labels
    ax.set_xlabel('Association with Human Eval (>5 = Highly Specific)',
                 fontsize=11, fontweight='bold')  # Reduced for single-column
    ax.set_ylabel('Association with LaaJ (>5 = Highly Specific)',
                 fontsize=11, fontweight='bold')  # Reduced for single-column


    # No legend needed since all points use the same color
    # The panel background colors provide the visual categorization

    # Use log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4, which='both')

    # Set limits (log scale friendly) - start at 0.3, same range for both axes
    ax.set_xlim(0.3, 30)
    ax.set_ylim(0.3, 30)  # Same range as x-axis

    # Add more tick values for better readability
    # Create more granular ticks for log scale
    x_ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    y_ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=9)  # Reduced for single-column
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=9)  # Reduced for single-column

    # Add axis labels for ticks
    ax.tick_params(labelsize=9)  # Reduced for single-column

    # Quadrant labels removed per user request

    plt.tight_layout()

    # Save high resolution
    output_file = output_dir / "human_vs_llm_scatter_readable.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved readable scatter plot to {output_file}")

    # Also save as PDF for papers
    output_pdf = output_dir / "human_vs_llm_scatter_readable.pdf"

    # Recreate for PDF with same improvements
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Nature Style: Clean white background (same as PNG)
    # Plot bubbles with simplified 3-color scheme (same as PNG)

    # Plot UNLABELED points first - all filled
    for color_name, color_mask, color in [
        ('grey', generic_color_mask, grey_color),
        ('red', (~generic_color_mask) & above_diagonal, red_color),
        ('blue', (~generic_color_mask) & below_diagonal, blue_color)
    ]:
        mask = (~df_plot["Metric"].isin(key_metrics)) & color_mask
        if mask.sum() > 0:
            ax.scatter(
                df_plot.loc[mask, "Human Plot"],
                df_plot.loc[mask, "LLM Plot"],
                alpha=0.4,
                s=df_plot.loc[mask, "Total"] * 2.0,  # Reduced for single-column layout
                color=color,
                marker='o',
                edgecolors='none',
                zorder=1
            )

    # Plot LABELED points on top - all filled
    for color_name, color_mask, color in [
        ('grey', generic_color_mask, grey_color),
        ('red', (~generic_color_mask) & above_diagonal, red_color),
        ('blue', (~generic_color_mask) & below_diagonal, blue_color)
    ]:
        mask = (df_plot["Metric"].isin(key_metrics)) & color_mask
        if mask.sum() > 0:
            ax.scatter(
                df_plot.loc[mask, "Human Plot"],
                df_plot.loc[mask, "LLM Plot"],
                alpha=0.8,
                s=df_plot.loc[mask, "Total"] * 3.0,  # Reduced for single-column layout
                color=color,
                marker='o',
                edgecolors='#333333',
                linewidth=1.0,  # Reduced for single-column layout
                zorder=3
            )

    # Nature Style: Diagonal line and grid lines (same as PNG)
    ax.plot([0.3, 30], [0.3, 30], 'k--', alpha=0.4, linewidth=1.0, zorder=2)

    # Dashed grid lines at LR=1 and LR=5
    ax.axvline(x=1.0, color='#AAAAAA', linestyle='--', alpha=0.5, linewidth=0.8, zorder=1)
    ax.axhline(y=1.0, color='#AAAAAA', linestyle='--', alpha=0.5, linewidth=0.8, zorder=1)
    ax.axvline(x=5.0, color='#666666', linestyle='--', alpha=0.6, linewidth=1.0, zorder=1)
    ax.axhline(y=5.0, color='#666666', linestyle='--', alpha=0.6, linewidth=1.0, zorder=1)

    try:
        from adjustText import adjust_text
        import matplotlib.patheffects as pe

        texts = []
        for metric, x, y in important_metrics:
            # For points on the right edge, position label to the left
            ha = 'right' if x > 5 else 'center'
            text = ax.text(x, y, metric, fontsize=10, fontweight='600',  # Reduced for single-column
                          ha=ha, color='#222222', zorder=10)
            # Add white stroke/halo for readability
            text.set_path_effects([
                pe.withStroke(linewidth=2.5, foreground='white', alpha=0.9),  # Reduced for single-column
                pe.Normal()
            ])
            texts.append(text)

        # More aggressive parameters to keep labels within plot bounds
        # Add explicit axis limits to prevent labels from moving outside
        adjust_text(texts,
                   ax=ax,
                   arrowprops=dict(arrowstyle='->', color='#666666', lw=0.4, alpha=0.6),
                   expand_points=(1.8, 1.8), expand_text=(1.5, 1.5),
                   lim=2000, force_text=(0.8, 0.8), force_points=(0.5, 0.5),
                   only_move={'points':'', 'text':'xy', 'objects':'xy'},
                   # Force labels to stay within axis limits
                   ensure_inside_axes=True)
    except ImportError:
        for i, (metric, x, y) in enumerate(important_metrics):
            angle = (i * 45) % 360
            if x > 6 or y > 8 or x < 1 or y < 1:
                radius = 22
            else:
                radius = 18

            angle_rad = np.radians(angle)
            dx = radius * np.cos(angle_rad)
            dy = radius * np.sin(angle_rad)

            # More aggressive left placement for right-side points
            if x > 5:  # Right side - FORCE LEFT placement
                dx = -abs(dx) - 40  # Extra offset to the left
            elif x < 1:  # Left side - prefer RIGHT (toward center)
                dx = abs(dx)

            if y > 8:  # Top - prefer DOWN (toward center)
                dy = -abs(dy)
            elif y < 1:  # Bottom - prefer UP (toward center)
                dy = abs(dy)

            import matplotlib.patheffects as pe
            text = ax.annotate(metric, (x, y), fontsize=9, fontweight='600',
                              xytext=(dx, dy), textcoords='offset points',
                              color='#222222', zorder=10,
                              arrowprops=dict(arrowstyle='->', color='#666666', lw=0.4, alpha=0.6))
            # Add white stroke/halo
            text.set_path_effects([
                pe.withStroke(linewidth=2, foreground='white', alpha=0.9),
                pe.Normal()
            ])

    # Nature Style: Minimal zone labels (same as PNG)
    ax.text(1.0, 0.32, 'LR=1', fontsize=8, alpha=0.5, ha='center', color='#666666')  # Reduced for single-column
    ax.text(5.0, 0.32, 'LR=5', fontsize=8, alpha=0.5, ha='center', color='#666666')
    ax.text(0.32, 1.0, 'LR=1', fontsize=8, alpha=0.5, va='center', rotation=90, color='#666666')
    ax.text(0.32, 5.0, 'LR=5', fontsize=8, alpha=0.5, va='center', rotation=90, color='#666666')

    ax.set_xlabel('Association with Human Judges (>5 = Highly Specific)',
                 fontsize=11, fontweight='bold')  # Reduced for single-column
    ax.set_ylabel('Association with LLM Judges (LaaJ)\n(>5 = Highly Specific)',
                 fontsize=11, fontweight='bold')  # Reduced for single-column


    # No legend needed since all points use the same color
    # The panel background colors provide the visual categorization
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4, which='both')
    # Same range for both axes
    ax.set_xlim(0.3, 30)
    ax.set_ylim(0.3, 30)  # Same range as x-axis

    # Add more tick values for better readability (same as PNG version)
    x_ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    y_ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=9)  # Reduced for single-column
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=9)  # Reduced for single-column

    ax.tick_params(labelsize=9)  # Reduced for single-column

    # Quadrant labels removed per user request

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved PDF version to {output_pdf}")


def main():
    print("Loading data...")
    human_data, llm_data = load_data()

    print("Loading normalization mapping...")
    normalization_map = load_normalization_mapping()
    print(f"Loaded {len(normalization_map)} metric name mappings")

    print("Creating comparison dataframe...")
    df = create_comparison_dataframe(human_data, llm_data, normalization_map, min_total=10)
    
    # Check for single-side metrics and show their actual association ratios
    # Note: 0.3 can mean either:
    # 1. Metric doesn't appear in that evaluation type (true single-side)
    # 2. Metric appears but enrichment is None (couldn't be calculated)
    # Human-only: appears in Human but not in LLM, OR Human enrichment available but LLM is None/0.3
    human_only = df[(df["LLM Enrichment"] == 0.3) & (df["Human Enrichment"] != 0.3)]
    # LLM-only: appears in LLM but not in Human, OR LLM enrichment available but Human is None/0.3
    llm_only = df[(df["Human Enrichment"] == 0.3) & (df["LLM Enrichment"] != 0.3)]
    
    print("\n" + "="*80)
    print("SINGLE-SIDE METRICS ANALYSIS")
    print("="*80)
    print(f"\nHuman-only metrics (appear only in Human evaluation, NOT in LLM): {len(human_only)}")
    if len(human_only) > 0:
        print("\n  Metric Name      | Human Assoc | LLM Assoc (0.3=placeholder) | Category    | Total Usage")
        print("  " + "-"*85)
        for _, row in human_only.iterrows():
            # Determine category based on Human enrichment
            h_e = row['Human Enrichment']
            if h_e >= 2.0:
                cat = "High"
            else:
                cat = "Low"
            category = f"{cat}-Low"
            print(f"  {row['Metric']:17s} | {h_e:10.2f} | {row['LLM Enrichment']:28.2f} | {category:11s} | {row['Total']:11.0f}")
    
    print(f"\nLLM-only metrics (appear only in LLM evaluation, NOT in Human): {len(llm_only)}")
    if len(llm_only) > 0:
        print("\n  Metric Name      | Human Assoc (0.3=placeholder) | LLM Assoc | Category    | Total Usage")
        print("  " + "-"*85)
        for _, row in llm_only.iterrows():
            # Determine category based on LLM enrichment
            l_e = row['LLM Enrichment']
            if l_e >= 2.0:
                cat = "High"
            else:
                cat = "Low"
            category = f"Low-{cat}"
            print(f"  {row['Metric']:17s} | {row['Human Enrichment']:28.2f} | {l_e:9.2f} | {category:11s} | {row['Total']:11.0f}")
    
    print("\n" + "="*80)

    print("Categorizing metrics...")
    df = categorize_metrics_simplified(df)

    print("Creating improved scatter plot...")
    create_improved_scatter(df, OUTPUT_DIR)

    print("\n" + "="*80)
    print("Improved scatter plot created!")
    print("="*80)


if __name__ == "__main__":
    main()