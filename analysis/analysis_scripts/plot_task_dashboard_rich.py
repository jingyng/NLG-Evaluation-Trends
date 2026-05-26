import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
import os
import hashlib
from data_loader import load_data, short_label
import matplotlib.colors as mcolors
from pathlib import Path

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

def enhance_color(color_tuple):
    """
    Make a color more saturated and darker while preserving hue.
    Converts RGB to HSV, increases saturation and decreases value.
    """
    # Extract RGB (ignore alpha if present)
    if len(color_tuple) == 4:
        r, g, b, a = color_tuple
    else:
        r, g, b = color_tuple
        a = 1.0

    # Convert to HSV
    h, s, v = mcolors.rgb_to_hsv([r, g, b])

    # Increase saturation (cap at 1.0)
    s = min(s * 1.5, 1.0)

    # Decrease value/brightness to make darker (multiply by 0.7)
    v = v * 0.7

    # Convert back to RGB
    r_new, g_new, b_new = mcolors.hsv_to_rgb([h, s, v])

    return (r_new, g_new, b_new, a)

def create_rich_task_dashboard(papers):
    print("Generating Rich Task Dashboard...")

    # Load metric normalization mapping for displaying common variants
    metric_mapping = load_metric_normalization_mapping()
    print(f"Loaded {len(metric_mapping)} metric name mappings")

    # 0. Pre-calculate Global Task Specificity
    # Formula: 1 - (Num_Tasks_Using_Term / Total_Tasks)
    all_tasks_set = set()
    term_task_map = defaultdict(set) # (type, term) -> {tasks}
    
    for p in papers:
        # We use the first task as the primary task for "used in task" count? 
        # Or all tasks? The user said "tasks_using_A".
        # Assuming p['tasks'] lists all relevant tasks.
        tasks = [t.lower().strip() for t in p['tasks']]
        all_tasks_set.update(tasks)
        
        for t_type in ['auto_metrics', 'human_criteria', 'laaj_criteria']:
            for term in p.get(t_type, []):
                term_clean = term.lower().strip()
                for t in tasks:
                    term_task_map[(t_type, term_clean)].add(t)

    total_unique_tasks = len(all_tasks_set)
    print(f"Total Unique Tasks for Specificity Calc: {total_unique_tasks}")
    
    def get_specificity(term, t_type):
        used_in = term_task_map.get((t_type, term), set())
        if not used_in: return 0
        return 1.0 - (len(used_in) / total_unique_tasks)

    # 1. Setup Data
    top_tasks = ['dialogue generation', 'machine translation', 'text summarization', 'question answering']
    
    # Helper to calculate stats
    def get_task_papers_pure(task_name):
        # Filter for papers that ONLY discuss this task
        return [p for p in papers 
                if len(p['tasks']) == 1 
                and task_name == p['tasks'][0].lower().strip()]

    def get_task_stats(task_name, field_key, min_count=5):
        task_papers = get_task_papers_pure(task_name)
        not_task_papers = [p for p in papers if task_name not in [t.lower() for t in p['tasks']]]
        
        n_task = len(task_papers)
        n_not_task = len(not_task_papers)
        
        if n_task == 0: return []
        
        # Trend Calc Prep
        early_papers = [p for p in task_papers if p['year'] in [2020, 2021, 2022]]
        late_papers = [p for p in task_papers if p['year'] in [2024, 2025]]
        n_early = max(len(early_papers), 1)
        n_late = max(len(late_papers), 1)
        
        c_early = Counter([m.lower().strip() for p in early_papers for m in p[field_key]])
        c_late = Counter([m.lower().strip() for p in late_papers for m in p[field_key]])

        c_task = Counter([m.lower().strip() for p in task_papers for m in p[field_key]])
        c_not_task = Counter([m.lower().strip() for p in not_task_papers for m in p[field_key]])
        
        stats = []
        for term, count in c_task.items():
            if count < min_count: continue
            
            p_task = count / n_task
            p_not_task = max(c_not_task.get(term, 0) / n_not_task, 0.001)
            
            assoc = p_task / p_not_task
            spec = get_specificity(term, field_key)
            
            # Trend
            p_e = c_early.get(term, 0) / n_early
            p_l = c_late.get(term, 0) / n_late
            if p_e == 0: p_e = 0.005 # Smoothing
            trend = p_l / p_e
            
            stats.append({
                'term': term,
                'coverage': p_task,
                'association': assoc,
                'specificity': spec,
                'trend': trend,
                'count': count
            })
        return stats

    def get_trend_stats(task_name, field_key):
        task_papers = get_task_papers_pure(task_name)
        early = [p for p in task_papers if p['year'] in [2020, 2021, 2022]]
        late = [p for p in task_papers if p['year'] in [2024, 2025]]
        
        if len(early) < 5 or len(late) < 5: return {}
        
        c_early = Counter([m.lower().strip() for p in early for m in p[field_key]])
        c_late = Counter([m.lower().strip() for p in late for m in p[field_key]])
        
        trends = {}
        all_terms = set(c_early.keys()) | set(c_late.keys())
        
        for term in all_terms:
            if c_late.get(term, 0) < 3: continue # Filter rare in late
            
            p_early = max(c_early.get(term, 0) / len(early), 0.005)
            p_late = c_late.get(term, 0) / len(late)
            
            trends[term] = p_late / p_early
            
        return trends

    def get_human_llm_comparison(task_name):
        task_papers = get_task_papers_pure(task_name)
        n_task = len(task_papers)
        if n_task == 0: return []
        
        h_counts = Counter([c.lower().strip() for p in task_papers for c in p['human_criteria']])
        l_counts = Counter([c.lower().strip() for p in task_papers for c in p['laaj_criteria']])
        
        shared = set(h_counts.keys()) & set(l_counts.keys())
        data = []
        for term in shared:
            if h_counts[term] < 3 or l_counts[term] < 3: continue
            data.append({
                'term': term,
                'human_cov': h_counts[term] / n_task,
                'llm_cov': l_counts[term] / n_task,
                'diff': (l_counts[term] / n_task) - (h_counts[term] / n_task)
            })
        return sorted(data, key=lambda x: abs(x['diff']), reverse=True)[:6]

    # 2. Build global color mapping for consistent colors across all plots
    # Collect all unique terms with their frequencies across all tasks
    term_frequencies = Counter()
    term_data_map = {}  # Store term data for frequency calculation
    
    for task in top_tasks:
        for field_key in ['auto_metrics', 'human_criteria', 'laaj_criteria']:
            data = get_task_stats(task, field_key, min_count=2 if field_key != 'auto_metrics' else 5)
            for item in data:
                term = item['term']
                # Count frequency: sum of counts across all tasks
                term_frequencies[term] += item['count']
                # Also consider coverage and association for ranking
                if term not in term_data_map:
                    term_data_map[term] = {
                        'total_count': 0,
                        'avg_coverage': 0,
                        'avg_association': 0,
                        'num_tasks': 0
                    }
                term_data_map[term]['total_count'] += item['count']
                term_data_map[term]['avg_coverage'] += item['coverage']
                term_data_map[term]['avg_association'] += item['association']
                term_data_map[term]['num_tasks'] += 1
    
    # Calculate composite score: prioritize by frequency, coverage, and number of tasks
    term_scores = {}
    for term, freq in term_frequencies.items():
        if term in term_data_map:
            data = term_data_map[term]
            # Composite score: weighted combination
            # More weight to total count, but also consider coverage and number of tasks
            score = (data['total_count'] * 1.0 + 
                    data['avg_coverage'] * 100 + 
                    data['num_tasks'] * 10)
            term_scores[term] = score
    
    # Sort terms by score (most frequent first)
    all_terms_sorted = sorted(term_scores.keys(), key=lambda x: term_scores[x], reverse=True)
    
    # Use distinct colors for top N most frequent terms, less distinct colors for others
    num_distinct_colors = 12  # Use top 12 most distinct colors
    
    # Create a custom palette with highly distinct colors for top terms
    # Using tab10 (10 colors) + 2 from Set2 for maximum distinctness
    distinct_colors = []
    tab10_map = plt.get_cmap('tab10')
    for i in range(10):
        distinct_colors.append(tab10_map(i))
    # Add 2 more from Set2
    set2_map = plt.get_cmap('Set2')
    distinct_colors.append(set2_map(0))
    distinct_colors.append(set2_map(1))
    
    # For less frequent terms, use a different set of colors to ensure uniqueness
    # Use multiple colormaps to generate enough distinct colors
    set2_map = plt.get_cmap('Set2')
    tab20_map = plt.get_cmap('tab20')
    set3_map = plt.get_cmap('Set3')
    pastel2_map = plt.get_cmap('Pastel2')

    # Create a large palette by combining multiple colormaps
    extended_palette = []
    # Add colors from Set2 (8 colors)
    for i in range(8):
        color = set2_map(i)
        if len(color) == 4:
            extended_palette.append((*color[:3], 1.0))
        else:
            extended_palette.append(color)
    # Add colors from tab20 (20 colors)
    for i in range(20):
        color = tab20_map(i)
        if len(color) == 4:
            extended_palette.append((*color[:3], 1.0))
        else:
            extended_palette.append(color)
    # Add colors from Set3 (12 colors) - enhanced for better visibility
    for i in range(12):
        color = set3_map(i)
        if len(color) == 4:
            color = (*color[:3], 1.0)
        # Enhance the color to make it more saturated and darker
        enhanced_color = enhance_color(color)
        extended_palette.append(enhanced_color)
    # Add colors from Pastel2 (8 colors) - enhanced for better visibility
    for i in range(8):
        color = pastel2_map(i)
        if len(color) == 4:
            color = (*color[:3], 1.0)
        # Enhance the color to make it more saturated and darker
        enhanced_color = enhance_color(color)
        extended_palette.append(enhanced_color)
    
    global_term_colors = {}
    
    for idx, term in enumerate(all_terms_sorted):
        if idx < num_distinct_colors:
            # Top N: use most distinct colors (ensure fully opaque)
            color = distinct_colors[idx]
            if len(color) == 4 and color[3] < 1.0:
                global_term_colors[term] = (*color[:3], 1.0)
            else:
                global_term_colors[term] = color
        else:
            # Less frequent: use hash-based assignment to ensure each term gets unique color
            # Hash the term name to get a consistent but unique index
            term_hash = int(hashlib.md5(term.encode()).hexdigest(), 16)
            palette_idx = term_hash % len(extended_palette)
            global_term_colors[term] = extended_palette[palette_idx]

    # 3. Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    # fig.suptitle('NLG Evaluation Methods Dashboard (2020-2025)\nBump Charts: Rank by Association, Size = Coverage', 
    #              fontsize=18, fontweight='bold', y=0.99)

    # Helper for Bump Chart (Rank Evolution)
    def plot_bump_chart(ax, data, task_name, field_key, color, col_idx=0, metric_mapping=None):
        if not data:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            return

        # 1. Select Top Terms by total count
        top_items = sorted(data, key=lambda x: x['count'], reverse=True)[:10]
        terms = [x['term'] for x in top_items]
        
        # 2. Fetch Yearly Data + Calculate Association and Prevalence per Year
        task_papers = get_task_papers_pure(task_name)
        not_task_papers = [p for p in papers if task_name not in [t.lower() for t in p['tasks']]]
        
        years = [2020, 2021, 2022, 2023, 2024, 2025]
        
        # Collect data per year, then rank
        term_trajectory = defaultdict(lambda: {'years': [], 'ranks': [], 'prevalences': []})
        
        for year in years:
            yp = [p for p in task_papers if p['year'] == year]
            yp_not = [p for p in not_task_papers if p['year'] == year]
            n_yp = len(yp)
            n_yp_not = len(yp_not)
            if n_yp < 3: continue 
            
            c_y = Counter([m.lower().strip() for p in yp for m in p[field_key]])
            c_y_not = Counter([m.lower().strip() for p in yp_not for m in p[field_key]])
            
            year_data = []
            for term in terms:
                count = c_y.get(term, 0)
                if count < 2: continue  # Skip very rare occurrences
                
                prevalence = count / n_yp
                
                # Calculate association for this year
                if n_yp_not > 0:
                    p_not_task = max(c_y_not.get(term, 0) / n_yp_not, 0.001)
                    association = prevalence / p_not_task
                else:
                    association = prevalence / 0.001
                
                year_data.append({
                    'term': term,
                    'association': association,
                    'prevalence': prevalence
                })
            
            # Rank by association within this year
            year_data.sort(key=lambda x: x['association'], reverse=True)
            for rank, item in enumerate(year_data[:10], start=1):  # Top 10 per year
                term_trajectory[item['term']]['years'].append(year)
                term_trajectory[item['term']]['ranks'].append(rank)
                term_trajectory[item['term']]['prevalences'].append(item['prevalence'])

        if not term_trajectory:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', transform=ax.transAxes)
            return
        
        # 3. Plot lines for each metric/criterion
        # Use global color mapping for consistent colors across all plots
        for term, traj in sorted(term_trajectory.items(), 
                                 key=lambda x: min(x[1]['ranks'])):
            if len(traj['years']) < 2: continue  # Need at least 2 points for a line
            
            # Always use global color mapping (most important for consistency)
            line_color = global_term_colors.get(term, '#808080')
            
            # Determine if this is a less frequent term (not in top 12 globally)
            term_rank = all_terms_sorted.index(term) if term in all_terms_sorted else len(all_terms_sorted)
            is_less_frequent = term_rank >= num_distinct_colors

            # Use higher alpha for less frequent terms to make them more visible
            line_alpha = 0.8 if is_less_frequent else 0.6
            scatter_alpha = 0.9 if is_less_frequent else 0.7
            
            ax.plot(traj['years'], traj['ranks'], linewidth=2,
                   color=line_color, alpha=line_alpha, zorder=1)

            # Plot bubbles sized by coverage/prevalence
            sizes = [p * 2000 + 50 for p in traj['prevalences']]
            ax.scatter(traj['years'], traj['ranks'], s=sizes,
                      color=line_color, alpha=scatter_alpha, edgecolors='white',
                      linewidth=1.5, zorder=3)
            
            # Add label below the last bubble
            if traj['years']:
                last_year = traj['years'][-1]
                last_rank = traj['ranks'][-1]

                # Use most common variant for auto_metrics; for criteria use the
                # QCET short label.
                display_name = term
                if field_key == 'auto_metrics' and metric_mapping:
                    display_name = metric_mapping.get(term, term)
                else:
                    # For criteria: map QCET full name → figure short label.
                    display_name = short_label(term)

                ax.text(last_year, last_rank + 0.25, display_name,
                       fontsize=14, ha='center', va='top', color=line_color,
                       fontweight='bold')
        
        # 4. Styling
        ax.set_xlim(2019.5, 2025.5)
        ax.set_ylim(0.5, 10.5)
        ax.invert_yaxis()  # Rank 1 at top
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], fontsize=14)
        ax.set_yticks(range(1, 11))
        # Only set ylabel for the first column, with row-specific text
        if col_idx == 0:
            # Map field_key to label text
            label_map = {
                'auto_metrics': 'Rank by Task-Metric Association',
                'human_criteria': 'Rank by Task-Human Criteria Association',
                'laaj_criteria': 'Rank by Task-LaaJ Criteria Association'
            }
            ylabel_text = label_map.get(field_key, 'Rank by Association')
            ax.set_ylabel(ylabel_text, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linewidth=1)
        ax.grid(True, alpha=0.15, axis='x', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

    for col_idx, task in enumerate(top_tasks):
        print(f"Processing {task}...")
        n_papers = len(get_task_papers_pure(task))
        axes[0, col_idx].set_title(f"{task.title()}\n(N={n_papers})", fontsize=20, fontweight='bold', pad=20)
        
        # --- Row 1: Auto Metrics (Bump Chart) ---
        ax = axes[0, col_idx]
        data = get_task_stats(task, 'auto_metrics')
        plot_bump_chart(ax, data, task, 'auto_metrics', '#4e79a7', col_idx, metric_mapping)

        # --- Row 2: Human Criteria (Bump Chart) ---
        ax = axes[1, col_idx]
        data = get_task_stats(task, 'human_criteria', min_count=2)
        plot_bump_chart(ax, data, task, 'human_criteria', '#f28e2b', col_idx, metric_mapping)

        # --- Row 3: LLM Criteria (Bump Chart) ---
        ax = axes[2, col_idx]
        data = get_task_stats(task, 'laaj_criteria', min_count=2)
        plot_bump_chart(ax, data, task, 'laaj_criteria', '#e15759', col_idx, metric_mapping)

    plt.tight_layout()
    
    # Reduce left margin to minimize empty space while keeping y-axis titles visible
    plt.subplots_adjust(left=0.05)
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures', 'task_dashboard', 'task_dashboard_rich.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    papers = load_data()
    create_rich_task_dashboard(papers)
