import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
"""
Clustered Metric-Criteria Alignment Visualization
Combines the strengths of network graphs (clustering/structure) 
and heatmaps (comprehensive view with exact values).

Features:
- Hierarchical clustering of metrics and criteria (shows structure like network graphs)
- Shows all relationships with exact values (like heatmaps)
- Displays both association ratio (color) and coverage (transparency + text)
- Side-by-side comparison of Human vs LLM
- Dendrograms show clustering structure
- AUTOMATIC ALIGNMENT: Ensures heatmap cells are physically identical in size
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
import os
from data_loader import load_data, short_label
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from pathlib import Path
from association_measures import compute_all, bh_fdr

def load_metric_normalization_mapping():
    """Load mapping from normalized metric names to their most common variant."""
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"

    mapping = {}
    with open(csv_path, 'r') as f:
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

                # Map normalized name (lowercase) to most common variant
                mapping[normalized.lower()] = variant_name

    return mapping

def calculate_composite_score(coverage, association, coverage_weight=0.4, association_weight=0.6):
    """
    Calculate composite score combining coverage and association.
    
    Args:
        coverage: Normalized coverage (0-1)
        association: Max association value (can be > 1)
        coverage_weight: Weight for coverage component
        association_weight: Weight for association component
    
    Returns:
        Composite score (higher is better)
    """
    # Normalize association: log scale to handle large values, cap at 10
    norm_assoc = min(np.log1p(association) / np.log1p(10), 1.0) if association > 0 else 0
    
    # Composite score
    score = coverage_weight * coverage + association_weight * norm_assoc
    return score

def select_ideal_items(items_dict, item_counts, n_select=20, 
                      min_coverage_threshold=0.01, min_assoc_threshold=1.5):
    """
    Select items using ideal strategy: prioritize items high on BOTH dimensions.
    
    Args:
        items_dict: {item: {'max': max_assoc, 'mean': mean_assoc, 'frequency': freq}}
        item_counts: Counter of item frequencies
        n_select: Number of items to select
        min_coverage_threshold: Minimum coverage to consider
        min_assoc_threshold: Minimum association to consider
    
    Returns:
        List of selected items
    """
    # Normalize frequencies for coverage score
    max_freq = max(item_counts.values()) if item_counts else 1
    
    scored_items = []
    for item, data in items_dict.items():
        freq = data.get('frequency', item_counts.get(item, 0))
        max_assoc = data.get('max', 0)
        mean_assoc = data.get('mean', 0)
        
        # Normalized coverage (0-1)
        norm_coverage = freq / max_freq if max_freq > 0 else 0
        
        # Use max association
        association = max_assoc
        
        # Calculate composite score
        composite = calculate_composite_score(norm_coverage, association)
        
        # Tier classification
        high_coverage = norm_coverage > 0.3  # Top 30% by frequency
        high_association = association > min_assoc_threshold
        
        if high_coverage and high_association:
            tier = 1  # Best: high on both
        elif high_coverage or high_association:
            tier = 2  # Good: high on one
        else:
            tier = 3  # Moderate
        
        scored_items.append({
            'item': item,
            'composite': composite,
            'coverage': norm_coverage,
            'association': association,
            'tier': tier,
            'frequency': freq
        })
    
    # Sort by tier first, then composite score
    scored_items.sort(key=lambda x: (x['tier'], -x['composite']))
    
    # Select: prioritize Tier 1, then Tier 2, then Tier 3
    selected = []
    for tier in [1, 2, 3]:
        tier_items = [x for x in scored_items if x['tier'] == tier]
        remaining_slots = n_select - len(selected)
        if remaining_slots > 0:
            selected.extend(tier_items[:remaining_slots])
    
    return [x['item'] for x in selected[:n_select]]

def align_heatmap_scales(fig):
    """
    Resizes axes in the figure so that heatmap cells are the same physical size 
    across subplots, regardless of how many columns/rows each has.
    """
    # 1. Identify the main heatmap axes (ignore colorbars)
    target_titles = ['Human Evaluation', 'LLM-as-a-Judge']
    main_axes = []
    
    # Sort by x-position to ensure Left is [0] and Right is [1]
    sorted_axes = sorted(fig.axes, key=lambda x: x.get_position().x0)
    
    for ax in sorted_axes:
        if any(t in ax.get_title() for t in target_titles):
            main_axes.append(ax)
            
    if len(main_axes) < 2:
        return # Need two plots to align
        
    ax1, ax2 = main_axes[0], main_axes[1]
    
    # 2. Get the data dimensions (number of columns and rows)
    cols1 = ax1.get_xlim()[1] - ax1.get_xlim()[0]
    rows1 = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    cols2 = ax2.get_xlim()[1] - ax2.get_xlim()[0]
    rows2 = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    
    # 3. Get current screen positions
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    
    # 4. Calculate cell size for each plot
    # cell_width = plot_width / num_cols
    # cell_height = plot_height / num_rows
    
    # Increase the gap between plots for better separation
    left_edge = pos1.x0
    current_gap = pos2.x0 - (pos1.x0 + pos1.width)
    # Increase gap by 50%
    increased_gap = current_gap * 1.5

    # Total available width for both plots (reduce to accommodate larger gap)
    total_available_width = pos1.width + pos2.width - (increased_gap - current_gap)
    
    # Since both plots should have square cells and same number of rows/cols (20x20),
    # they should get equal widths. But let's calculate proportionally anyway.
    total_cols = cols1 + cols2
    
    # Calculate new widths proportional to column counts
    new_w1 = total_available_width * (cols1 / total_cols)
    new_w2 = total_available_width * (cols2 / total_cols)
    
    # To ensure square cells, height should equal width (since both are 20x20)
    # Use the average height to maintain aspect
    avg_height = (pos1.height + pos2.height) / 2
    
    # Use the same y0 for both plots to align them horizontally
    avg_y0 = (pos1.y0 + pos2.y0) / 2
    
    # 5. Apply new positions with equal heights and same vertical position
    ax1.set_position([left_edge, avg_y0, new_w1, avg_height])
    ax2.set_position([left_edge + new_w1 + increased_gap, avg_y0, new_w2, avg_height])
    
    # 6. Re-enforce aspect ratio after positioning
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

def create_clustered_metric_criteria_alignment(papers):
    print("Generating Clustered Metric-Criteria Alignment Plot...")

    # Load metric normalization mapping for displaying common variants
    metric_mapping = load_metric_normalization_mapping()
    print(f"Loaded {len(metric_mapping)} metric name mappings")

    # Use all papers (no filtering for single-task)
    print(f"Total papers: {len(papers)}")
    
    if len(papers) < 10:
        print("Insufficient data")
        return
    
    # Process all papers
    task_papers = papers
    
    # Count papers using each evaluation type
    human_papers = [p for p in task_papers if len(p['human_criteria']) > 0]
    llm_papers = [p for p in task_papers if len(p['laaj_criteria']) > 0]
    
    n_human = len(human_papers)
    n_llm = len(llm_papers)
    n_total = len(task_papers)
    
    print(f"Human papers: {n_human}, LLM papers: {n_llm}, Total: {n_total}")
    
    # Get top metrics and criteria across all papers
    all_metrics = [m.lower().strip() for p in task_papers for m in p['auto_metrics']]
    all_human_criteria = [c.lower().strip() for p in human_papers for c in p['human_criteria']]
    all_llm_criteria = [c.lower().strip() for p in llm_papers for c in p['laaj_criteria']]
    
    metric_counts = Counter(all_metrics)
    human_criteria_counts = Counter(all_human_criteria)
    llm_criteria_counts = Counter(all_llm_criteria)
    
    # We'll select metrics after calculating associations using ideal strategy
    top_metrics_by_freq = [m for m, _ in metric_counts.most_common(30)]  # Candidate pool
    
    # Create figure with 1 row and 2 columns (Human + LLM, no dendrograms)
    # Increased size to accommodate more items (20 metrics × 20 criteria)
    # Make width larger to ensure square cells when aspect='equal'
    fig = plt.figure(figsize=(36, 18))
    
    # Initialize top_metrics variable (will be set in Human section, reused for LLM)
    top_metrics = None
    
    # Process Human Evaluation
    if n_human >= 5:
        # First, get all unique human criteria
        all_unique_human_criteria = list(set(all_human_criteria))
        
        # Calculate associations for ALL criteria and ALL metrics to find high-association ones
        print(f"Calculating associations for {len(all_unique_human_criteria)} human criteria...")
        criterion_associations = {}
        metric_associations = {}
        
        # First, calculate metric associations across all criteria
        for metric in top_metrics_by_freq:
            max_assoc = 0
            associations = []
            
            for criterion in all_unique_human_criteria:
                papers_with_criterion = [p for p in human_papers 
                                       if criterion in [x.lower().strip() for x in p['human_criteria']]]
                papers_without_criterion = [p for p in human_papers 
                                         if criterion not in [x.lower().strip() for x in p['human_criteria']]]
                
                n_with_crit = len(papers_with_criterion)
                n_without_crit = len(papers_without_criterion)
                
                if n_with_crit > 0:
                    count_with = sum(1 for p in papers_with_criterion
                                   if metric in [x.lower().strip() for x in p['auto_metrics']])
                    count_without = sum(1 for p in papers_without_criterion
                                      if metric in [x.lower().strip() for x in p['auto_metrics']])
                    
                    pct_with = (count_with / n_with_crit * 100) if n_with_crit > 0 else 0
                    pct_without = (count_without / n_without_crit * 100) if n_without_crit > 0 else 0
                    
                    if pct_without > 0:
                        assoc = pct_with / pct_without
                    elif pct_with > 0:
                        assoc = float('inf')
                    else:
                        assoc = 1.0
                    
                    assoc_for_comparison = min(assoc, 100) if assoc != float('inf') else 100
                    associations.append(assoc_for_comparison)
                    max_assoc = max(max_assoc, assoc_for_comparison)
            
            metric_associations[metric] = {
                'max': max_assoc,
                'frequency': metric_counts[metric]
            }
        
        # Calculate criterion associations
        for criterion in all_unique_human_criteria:
            max_assoc = 0
            mean_assoc = 0
            associations = []
            
            papers_with_criterion = [p for p in human_papers 
                                   if criterion in [x.lower().strip() for x in p['human_criteria']]]
            papers_without_criterion = [p for p in human_papers 
                                     if criterion not in [x.lower().strip() for x in p['human_criteria']]]
            
            n_with_crit = len(papers_with_criterion)
            n_without_crit = len(papers_without_criterion)
            
            if n_with_crit > 0:
                for metric in top_metrics_by_freq:
                    count_with = sum(1 for p in papers_with_criterion
                                   if metric in [x.lower().strip() for x in p['auto_metrics']])
                    count_without = sum(1 for p in papers_without_criterion
                                      if metric in [x.lower().strip() for x in p['auto_metrics']])
                    
                    pct_with = (count_with / n_with_crit * 100) if n_with_crit > 0 else 0
                    pct_without = (count_without / n_without_crit * 100) if n_without_crit > 0 else 0
                    
                    if pct_without > 0:
                        assoc = pct_with / pct_without
                    elif pct_with > 0:
                        assoc = float('inf')
                    else:
                        assoc = 1.0
                    
                    assoc_for_comparison = min(assoc, 100) if assoc != float('inf') else 100
                    associations.append(assoc_for_comparison)
                    max_assoc = max(max_assoc, assoc_for_comparison)
            
            if associations:
                mean_assoc = np.mean(associations)
            
            criterion_associations[criterion] = {
                'max': max_assoc,
                'mean': mean_assoc,
                'frequency': human_criteria_counts[criterion]
            }
        
        # Select metrics using ideal tiered strategy
        top_metrics = select_ideal_items(metric_associations, metric_counts, n_select=20,
                                        min_assoc_threshold=1.5)

        # Transform metric names to use most common variants
        top_metrics_display = [metric_mapping.get(m, m) for m in top_metrics]

        # Count tiers for reporting
        tier_counts = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        for m in top_metrics:
            data = metric_associations[m]
            norm_cov = metric_counts[m] / max(metric_counts.values())
            if norm_cov > 0.3 and data['max'] > 1.5:
                tier_counts['tier1'] += 1
            elif norm_cov > 0.3 or data['max'] > 1.5:
                tier_counts['tier2'] += 1
            else:
                tier_counts['tier3'] += 1
        
        print(f"Selected {len(top_metrics)} metrics (Tier 1: {tier_counts['tier1']}, Tier 2: {tier_counts['tier2']}, Tier 3: {tier_counts['tier3']})")
        
        # Select criteria using ideal tiered strategy
        top_human_criteria = select_ideal_items(criterion_associations, human_criteria_counts,
                                               n_select=20, min_assoc_threshold=1.5)

        # Map QCET full names to short labels for figures (case-insensitive lookup).
        top_human_criteria_display = [short_label(c) for c in top_human_criteria]

        # Count tiers for reporting
        tier_counts_crit = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        for c in top_human_criteria:
            data = criterion_associations[c]
            norm_cov = human_criteria_counts[c] / max(human_criteria_counts.values())
            if norm_cov > 0.3 and data['max'] > 1.5:
                tier_counts_crit['tier1'] += 1
            elif norm_cov > 0.3 or data['max'] > 1.5:
                tier_counts_crit['tier2'] += 1
            else:
                tier_counts_crit['tier3'] += 1
        
        print(f"Selected {len(top_human_criteria)} human criteria (Tier 1: {tier_counts_crit['tier1']}, Tier 2: {tier_counts_crit['tier2']}, Tier 3: {tier_counts_crit['tier3']})")
        
        # Calculate matrices
        human_association_matrix = np.zeros((len(top_metrics), len(top_human_criteria)))
        human_coverage_matrix = np.zeros((len(top_metrics), len(top_human_criteria)))
        human_p_matrix = np.ones((len(top_metrics), len(top_human_criteria)))
        
        for m_idx, metric in enumerate(top_metrics):
            for c_idx, criterion in enumerate(top_human_criteria):
                # Get papers with and without this criterion
                papers_with_criterion = [p for p in human_papers 
                                       if criterion in [x.lower().strip() for x in p['human_criteria']]]
                papers_without_criterion = [p for p in human_papers 
                                          if criterion not in [x.lower().strip() for x in p['human_criteria']]]
                
                n_with_crit = len(papers_with_criterion)
                n_without_crit = len(papers_without_criterion)
                
                # Count papers WITH criterion that use metric
                count_with = sum(1 for p in papers_with_criterion
                               if metric in [x.lower().strip() for x in p['auto_metrics']])
                # Count papers WITHOUT criterion that use metric
                count_without = sum(1 for p in papers_without_criterion
                                  if metric in [x.lower().strip() for x in p['auto_metrics']])
                
                # Coverage: co-occurrence normalized by total human papers
                co_occur = sum(1 for p in human_papers
                             if (criterion in [x.lower().strip() for x in p['human_criteria']] and
                                 metric in [x.lower().strip() for x in p['auto_metrics']]))
                human_coverage_matrix[m_idx, c_idx] = co_occur / n_human

                # G² test for significance
                _s = compute_all(count_with, n_with_crit - count_with,
                                 count_without, n_without_crit - count_without)
                human_p_matrix[m_idx, c_idx] = _s["p_value"]

                # Enrichment: P(metric | criterion) / P(metric | not criterion)
                pct_with = (count_with / n_with_crit * 100) if n_with_crit > 0 else 0
                pct_without = (count_without / n_without_crit * 100) if n_without_crit > 0 else 0

                if pct_without > 0:
                    human_association_matrix[m_idx, c_idx] = pct_with / pct_without
                elif pct_with > 0:
                    human_association_matrix[m_idx, c_idx] = float('inf')
                else:
                    human_association_matrix[m_idx, c_idx] = 1.0
        
        # Create DataFrame with display names
        df_human = pd.DataFrame(human_association_matrix,
                               index=top_metrics_display,
                               columns=top_human_criteria_display)
        
        # Replace infinite values with a capped value for visualization
        df_human = df_human.replace([float('inf'), np.inf], 100)
        df_human = df_human.fillna(1.0)

        # BH-FDR across all human (metric × criterion) pairs
        flat_p = human_p_matrix.flatten().tolist()
        flat_q, _ = bh_fdr(flat_p)
        human_q_matrix = np.array(flat_q).reshape(human_p_matrix.shape)

        # Create clustered heatmap (on the right, col_idx=1)
        plot_clustered_heatmap(fig, df_human, human_coverage_matrix,
                              'Human Evaluation', n_human, n_total, 1,
                              q_matrix=human_q_matrix)
    else:
        ax = fig.add_subplot(1, 2, 2)
        ax.text(0.5, 0.5, "Insufficient\nHuman Data", ha='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title('Human Evaluation', fontsize=20, fontweight='bold')
    
    # Process LLM Evaluation
    if n_llm >= 5:
        # Use the same metrics as Human for consistency and easier comparison
        if top_metrics is None:
            # Fallback if Human section didn't run
            top_metrics = [m for m, _ in metric_counts.most_common(20)]
            top_metrics_display = [metric_mapping.get(m, m) for m in top_metrics]
        
        # First, get all unique LLM criteria
        all_unique_llm_criteria = list(set(all_llm_criteria))
        
        # Calculate associations for ALL criteria and metrics
        print(f"Calculating associations for {len(all_unique_llm_criteria)} LLM criteria...")
        criterion_associations = {}
        
        # Calculate criterion associations (using already-selected metrics from Human section)
        for criterion in all_unique_llm_criteria:
            max_assoc = 0
            mean_assoc = 0
            associations = []
            
            papers_with_criterion = [p for p in llm_papers 
                                   if criterion in [x.lower().strip() for x in p['laaj_criteria']]]
            papers_without_criterion = [p for p in llm_papers 
                                      if criterion not in [x.lower().strip() for x in p['laaj_criteria']]]
            
            n_with_crit = len(papers_with_criterion)
            n_without_crit = len(papers_without_criterion)
            
            if n_with_crit > 0:
                for metric in top_metrics_by_freq:
                    count_with = sum(1 for p in papers_with_criterion
                                   if metric in [x.lower().strip() for x in p['auto_metrics']])
                    count_without = sum(1 for p in papers_without_criterion
                                      if metric in [x.lower().strip() for x in p['auto_metrics']])
                    
                    pct_with = (count_with / n_with_crit * 100) if n_with_crit > 0 else 0
                    pct_without = (count_without / n_without_crit * 100) if n_without_crit > 0 else 0
                    
                    if pct_without > 0:
                        assoc = pct_with / pct_without
                    elif pct_with > 0:
                        assoc = float('inf')
                    else:
                        assoc = 1.0
                    
                    assoc_for_comparison = min(assoc, 100) if assoc != float('inf') else 100
                    associations.append(assoc_for_comparison)
                    max_assoc = max(max_assoc, assoc_for_comparison)
            
            if associations:
                mean_assoc = np.mean(associations)
            
            criterion_associations[criterion] = {
                'max': max_assoc,
                'mean': mean_assoc,
                'frequency': llm_criteria_counts[criterion]
            }
        
        # Select criteria using ideal tiered strategy
        top_llm_criteria = select_ideal_items(criterion_associations, llm_criteria_counts,
                                              n_select=20, min_assoc_threshold=1.5)

        # Map QCET full names to short labels for figures (case-insensitive).
        top_llm_criteria_display = [short_label(c) for c in top_llm_criteria]

        # Count tiers for reporting
        tier_counts_llm = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        for c in top_llm_criteria:
            data = criterion_associations[c]
            norm_cov = llm_criteria_counts[c] / max(llm_criteria_counts.values()) if llm_criteria_counts else 0
            if norm_cov > 0.3 and data['max'] > 1.5:
                tier_counts_llm['tier1'] += 1
            elif norm_cov > 0.3 or data['max'] > 1.5:
                tier_counts_llm['tier2'] += 1
            else:
                tier_counts_llm['tier3'] += 1
        
        print(f"Selected {len(top_llm_criteria)} LLM criteria (Tier 1: {tier_counts_llm['tier1']}, Tier 2: {tier_counts_llm['tier2']}, Tier 3: {tier_counts_llm['tier3']})")
        
        # Calculate matrices
        llm_association_matrix = np.zeros((len(top_metrics), len(top_llm_criteria)))
        llm_coverage_matrix = np.zeros((len(top_metrics), len(top_llm_criteria)))
        llm_p_matrix = np.ones((len(top_metrics), len(top_llm_criteria)))
        
        for m_idx, metric in enumerate(top_metrics):
            for c_idx, criterion in enumerate(top_llm_criteria):
                # Get papers with and without this criterion
                papers_with_criterion = [p for p in llm_papers 
                                       if criterion in [x.lower().strip() for x in p['laaj_criteria']]]
                papers_without_criterion = [p for p in llm_papers 
                                          if criterion not in [x.lower().strip() for x in p['laaj_criteria']]]
                
                n_with_crit = len(papers_with_criterion)
                n_without_crit = len(papers_without_criterion)
                
                # Count papers WITH criterion that use metric
                count_with = sum(1 for p in papers_with_criterion
                               if metric in [x.lower().strip() for x in p['auto_metrics']])
                # Count papers WITHOUT criterion that use metric
                count_without = sum(1 for p in papers_without_criterion
                                  if metric in [x.lower().strip() for x in p['auto_metrics']])
                
                # Coverage: co-occurrence normalized by total LLM papers
                co_occur = sum(1 for p in llm_papers
                             if (criterion in [x.lower().strip() for x in p['laaj_criteria']] and
                                 metric in [x.lower().strip() for x in p['auto_metrics']]))
                llm_coverage_matrix[m_idx, c_idx] = co_occur / n_llm

                # G² test for significance
                _s = compute_all(count_with, n_with_crit - count_with,
                                 count_without, n_without_crit - count_without)
                llm_p_matrix[m_idx, c_idx] = _s["p_value"]

                # Enrichment: P(metric | criterion) / P(metric | not criterion)
                pct_with = (count_with / n_with_crit * 100) if n_with_crit > 0 else 0
                pct_without = (count_without / n_without_crit * 100) if n_without_crit > 0 else 0

                if pct_without > 0:
                    llm_association_matrix[m_idx, c_idx] = pct_with / pct_without
                elif pct_with > 0:
                    llm_association_matrix[m_idx, c_idx] = float('inf')
                else:
                    llm_association_matrix[m_idx, c_idx] = 1.0
        
        # Create DataFrame with display names
        df_llm = pd.DataFrame(llm_association_matrix,
                             index=top_metrics_display,
                             columns=top_llm_criteria_display)
        
        # Replace infinite values with a capped value for visualization
        df_llm = df_llm.replace([float('inf'), np.inf], 100)
        df_llm = df_llm.fillna(1.0)

        # BH-FDR across all LLM (metric × criterion) pairs
        flat_p = llm_p_matrix.flatten().tolist()
        flat_q, _ = bh_fdr(flat_p)
        llm_q_matrix = np.array(flat_q).reshape(llm_p_matrix.shape)

        # Create clustered heatmap (on the left, col_idx=0)
        plot_clustered_heatmap(fig, df_llm, llm_coverage_matrix,
                              'LLM-as-a-Judge', n_llm, n_total, 0,
                              q_matrix=llm_q_matrix)
    else:
        ax = fig.add_subplot(1, 2, 1)
        ax.text(0.5, 0.5, "Insufficient\nLLM Data", ha='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title('LLM-as-a-Judge', fontsize=20, fontweight='bold')
    
    # --- ALIGN SCALES BEFORE SAVING ---
    # This aligns the physical size of heatmap cells across both plots
    align_heatmap_scales(fig)
    # ----------------------------------
    
    # --- ADD SHARED COLORBAR ---
    # Create a single colorbar for both subplots
    if hasattr(fig, '_heatmap_cmap'):
        # Get the two main heatmap axes
        target_titles = ['Human Evaluation', 'LLM-as-a-Judge']
        main_axes = [ax for ax in fig.axes if any(t in ax.get_title() for t in target_titles)]
        
        if len(main_axes) == 2:
            # Create a ScalarMappable for the colorbar
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(cmap=fig._heatmap_cmap, norm=fig._heatmap_norm)
            sm.set_array([])  # Required for the colorbar
            
            # Create colorbar spanning both axes
            cbar = fig.colorbar(sm, ax=main_axes, orientation='vertical',
                               fraction=0.02, pad=0.02, aspect=30)
            cbar.set_label('Association Ratio', fontsize=18, fontweight='bold')
            cbar.ax.tick_params(labelsize=16)
            ticks = [0, 1, 2, 5, 10]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(['0', '1', '2', '5', '10+'])
    # ----------------------------------

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures', 'metric_criteria_alignment', 'metric_criteria_alignment_clustered_final.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {out_path}")

def plot_clustered_heatmap(fig, df, coverage_matrix, title, n_eval, n_total, col_idx, q_matrix=None):
    """Plot clustered heatmap with dendrograms."""
    
    # Perform hierarchical clustering
    # Cluster metrics (rows)
    if len(df) > 1:
        metric_linkage = linkage(df.values, method='ward', metric='euclidean')
        metric_order = leaves_list(metric_linkage)
    else:
        metric_order = [0]
        metric_linkage = None
    
    # Cluster criteria (columns)
    if len(df.columns) > 1:
        criteria_linkage = linkage(df.T.values, method='ward', metric='euclidean')
        criteria_order = leaves_list(criteria_linkage)
    else:
        criteria_order = [0]
        criteria_linkage = None
    
    # Reorder data according to clustering (but don't show dendrograms)
    df_clustered = df.iloc[metric_order, criteria_order]
    coverage_clustered = coverage_matrix[np.ix_(metric_order, criteria_order)]
    q_clustered = q_matrix[np.ix_(metric_order, criteria_order)] if q_matrix is not None else None
    
    # Create heatmap subplot (no dendrograms)
    ax_heatmap = fig.add_subplot(1, 2, col_idx + 1)
    
    # Set equal aspect ratio FIRST to ensure square cells
    # Use 'box' to adjust data limits, allowing resize by our helper
    ax_heatmap.set_aspect('equal', adjustable='box')
    
    # Set title with larger font
    ax_heatmap.set_title(f'{title}\n({n_eval} papers with eval out of {n_total} total)',
                        fontsize=20, fontweight='bold', pad=15)
    
    # Colormap - handle values up to 100 (for inf values)
    colors = ['#f0f0f0', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b']
    cmap = LinearSegmentedColormap.from_list('sequential', colors, N=6)
    # Cap values at 10 for color mapping, but allow display of higher values
    df_for_plot = df_clustered.copy()
    df_for_plot = df_for_plot.clip(upper=10)  # Cap at 10 for color mapping
    norm = Normalize(vmin=0, vmax=10)
    
    # Use coverage for circle size inside each cell
    max_coverage = np.max(coverage_clustered) if np.max(coverage_clustered) > 0 else 1.0
    normalized_coverage = coverage_clustered / max_coverage
    
    # Circle size: 0.1 (low) to 0.4 (high coverage) as fraction of cell size
    min_circle_size = 0.1
    max_circle_size = 0.4
    circle_sizes = min_circle_size + normalized_coverage * (max_circle_size - min_circle_size)
    
    # Plot heatmap with fixed borders
    for m_idx in range(len(df_clustered)):
        for c_idx in range(len(df_clustered.columns)):
            assoc = df_clustered.iloc[m_idx, c_idx]
            assoc_for_color = min(assoc, 10)  # Cap at 10 for color
            cov = coverage_clustered[m_idx, c_idx]
            norm_cov = normalized_coverage[m_idx, c_idx]
            
            # Cell rectangle with fixed border
            rect = Rectangle((c_idx - 0.5, m_idx - 0.5), 1, 1,
                           facecolor=cmap(norm(assoc_for_color)),
                           alpha=0.9,
                           edgecolor='gray', 
                           linewidth=0.5, 
                           zorder=1)
            ax_heatmap.add_patch(rect)
            
            # Hatch non-significant cells (BH-FDR q > 0.05)
            if q_clustered is not None and q_clustered[m_idx, c_idx] > 0.05:
                hatch_rect = Rectangle((c_idx - 0.5, m_idx - 0.5), 1, 1,
                                       facecolor='none', hatch='/////',
                                       edgecolor='#888888', linewidth=0,
                                       alpha=0.45, zorder=2)
                ax_heatmap.add_patch(hatch_rect)

            # Add circle inside cell to indicate coverage frequency
            if cov > 0.01:  # Only show circle if there's some coverage
                circle = Circle((c_idx, m_idx),
                               radius=circle_sizes[m_idx, c_idx],
                               facecolor='white',
                               edgecolor='black',
                               linewidth=1.5,
                               alpha=0.7,
                               zorder=3)
                ax_heatmap.add_patch(circle)
    
    # No text annotations - circles and colors provide the information
    
    # Set labels with larger font sizes
    ax_heatmap.set_xticks(np.arange(len(df_clustered.columns)))
    ax_heatmap.set_xticklabels(df_clustered.columns, fontsize=20, rotation=45, ha='right')
    ax_heatmap.set_yticks(np.arange(len(df_clustered)))
    ax_heatmap.set_yticklabels(df_clustered.index, fontsize=20)
    
    # Set limits
    ax_heatmap.set_xlim(-0.5, len(df_clustered.columns) - 0.5)
    ax_heatmap.set_ylim(-0.5, len(df_clustered) - 0.5)
    
    # Ensure cells remain square (aspect already set earlier, but reinforce it)
    ax_heatmap.set_aspect('equal', adjustable='box')
    
    # Add grid
    ax_heatmap.set_xticks(np.arange(len(df_clustered.columns)) - 0.5, minor=True)
    ax_heatmap.set_yticks(np.arange(len(df_clustered)) - 0.5, minor=True)
    ax_heatmap.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Invert y-axis to match typical heatmap convention
    ax_heatmap.invert_yaxis()
    
    # Store the colormap and norm for shared colorbar (will be created later)
    # We'll create it after both subplots are drawn
    if not hasattr(fig, '_heatmap_cmap'):
        fig._heatmap_cmap = cmap
        fig._heatmap_norm = norm
        fig._heatmap_cmap = cmap
    
    # Add bubble size legend in upper right corner (using axes coordinates)
    max_coverage = np.max(coverage_clustered) if np.max(coverage_clustered) > 0 else 1.0
    min_circle_size = 0.1
    max_circle_size = 0.4
    
    # Example coverage values to show in legend (as fractions of max)
    example_coverages = [0.1, 0.3, 0.5, 0.7, 1.0]
    # Convert to actual coverage values (multiply by max_coverage)
    actual_coverages = [cov * max_coverage for cov in example_coverages]
    # Calculate circle sizes for these coverages (normalized)
    example_circle_sizes = [min_circle_size + cov * (max_circle_size - min_circle_size) 
                           for cov in example_coverages]
    
    # Position legend in upper right using axes coordinates (0-1)
    legend_x_start_ax = 0.98
    legend_y_start_ax = 0.98
    legend_spacing_ax = 0.08
    
    # Convert circle sizes to axes coordinates (approximate, using a reference)
    # We'll use a fixed reference size based on the plot dimensions
    num_cols = len(df_clustered.columns)
    num_rows = len(df_clustered)
    # Approximate cell size in axes coordinates
    cell_size_ax = 0.8 / max(num_cols, num_rows)  # Rough estimate
    
    # Draw example circles with labels
    for i, (cov_frac, actual_cov, circle_size) in enumerate(zip(example_coverages, actual_coverages, example_circle_sizes)):
        x_pos_ax = legend_x_start_ax - i * legend_spacing_ax
        y_pos_ax = legend_y_start_ax
        
        # Convert circle radius to axes coordinates
        circle_radius_ax = circle_size * cell_size_ax * 0.5  # Scale appropriately
        
        # Draw circle using axes coordinates
        circle = Circle((x_pos_ax, y_pos_ax), 
                         radius=circle_radius_ax,
                         facecolor='white',
                         edgecolor='black',
                         linewidth=1.5,
                         alpha=0.7,
                         zorder=10,
                         transform=ax_heatmap.transAxes)
        ax_heatmap.add_patch(circle)
        
        # Add label showing coverage percentage
        coverage_pct = actual_cov * 100
        ax_heatmap.text(x_pos_ax, y_pos_ax - circle_radius_ax - 0.02, f'{coverage_pct:.1f}%',
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       transform=ax_heatmap.transAxes)
    
    # Add legend title
    ax_heatmap.text(legend_x_start_ax - (len(example_coverages) - 1) * legend_spacing_ax / 2, 
                   legend_y_start_ax + 0.05, 'Coverage',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   transform=ax_heatmap.transAxes)

if __name__ == "__main__":
    papers = load_data()
    create_clustered_metric_criteria_alignment(papers)