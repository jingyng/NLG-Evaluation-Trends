#!/usr/bin/env python3
"""
Create a heatmap showing task-year frequency from the merged results.
"""

import json
import os
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'llm-merged-results-top30-tasks')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'figures', 'task_dashboard')

def load_task_year_data():
    """Load all papers and extract task-year pairs."""
    task_year_counts = defaultdict(int)
    year_total_papers = defaultdict(int)  # Total papers per year
    all_tasks = set()
    all_years = set()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract Year
                    year = None
                    if 'paper_id' in data:
                        parts = data['paper_id'].split('.')
                        if parts[0].isdigit() and len(parts[0]) == 4:
                            year = int(parts[0])
                    
                    if year is None:
                        folder = os.path.basename(root)
                        if '-' in folder:
                            try:
                                year = int(folder.split('-')[-1])
                            except:
                                pass
                    
                    if year is None:
                        continue
                    
                    # Count total papers per year
                    year_total_papers[year] += 1
                    all_years.add(year)
                    
                    # Extract tasks
                    tasks = data.get('answer_1', {}).get('tasks', [])
                    if not tasks:
                        continue
                    
                    # Count each task-year combination
                    for task in tasks:
                        if task:  # Skip empty strings
                            task_year_counts[(task, year)] += 1
                            all_tasks.add(task)
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    return task_year_counts, year_total_papers, all_tasks, all_years


def create_heatmap(task_year_counts, year_total_papers, all_tasks, all_years):
    """Create a heatmap of task-year frequency with percentages."""
    
    # Sort tasks by total frequency (most common first)
    task_totals = defaultdict(int)
    for (task, year), count in task_year_counts.items():
        task_totals[task] += count
    
    sorted_tasks = sorted(all_tasks, key=lambda t: task_totals[t], reverse=True)
    sorted_years = sorted(all_years)
    
    # Create matrices for counts and percentages
    count_matrix = np.zeros((len(sorted_tasks), len(sorted_years)), dtype=int)
    pct_matrix = np.zeros((len(sorted_tasks), len(sorted_years)), dtype=float)
    
    for i, task in enumerate(sorted_tasks):
        for j, year in enumerate(sorted_years):
            count = int(task_year_counts.get((task, year), 0))
            count_matrix[i, j] = count
            total_papers = year_total_papers.get(year, 1)  # Avoid division by zero
            if total_papers > 0:
                pct_matrix[i, j] = (count / total_papers) * 100
    
    # Create DataFrame for percentages (for coloring)
    df_pct = pd.DataFrame(pct_matrix, index=sorted_tasks, columns=sorted_years)
    
    # Create DataFrame for counts (for annotations)
    df_counts = pd.DataFrame(count_matrix, index=sorted_tasks, columns=sorted_years)
    
    # Create annotation matrix with count and percentage
    annot_matrix = []
    for i in range(len(sorted_tasks)):
        row = []
        for j in range(len(sorted_years)):
            count = count_matrix[i, j]
            pct = pct_matrix[i, j]
            if count > 0:
                row.append(f"{count}\n({pct:.1f}%)")
            else:
                row.append("")
        annot_matrix.append(row)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_tasks) * 0.4)))
    
    # Use seaborn heatmap with percentages for coloring, but show count+percentage in annotations
    sns.heatmap(df_pct, annot=annot_matrix, fmt='', cmap='YlOrRd', 
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5, linecolor='gray',
                ax=ax, vmin=0)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task', fontsize=14, fontweight='bold')
    # ax.set_title('Task-Year Frequency Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'task_year_heatmap.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    
    # Also save PDF
    output_path_pdf = os.path.join(OUTPUT_DIR, 'task_year_heatmap.pdf')
    fig.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved PDF to {output_path_pdf}")
    
    plt.close()
    
    return df_pct


def main():
    print("Loading task-year data...")
    task_year_counts, year_total_papers, all_tasks, all_years = load_task_year_data()
    
    print(f"\nFound {len(all_tasks)} unique tasks")
    print(f"Years: {sorted(all_years)}")
    print(f"Total task-year combinations: {len(task_year_counts)}")
    
    # Show total papers per year
    print(f"\nTotal papers per year:")
    for year in sorted(all_years):
        print(f"  {year}: {year_total_papers[year]}")
    
    # Show top tasks
    task_totals = defaultdict(int)
    for (task, year), count in task_year_counts.items():
        task_totals[task] += count
    
    print(f"\nTop 10 tasks by total frequency:")
    for task, total in Counter(task_totals).most_common(10):
        print(f"  {task}: {total}")
    
    print("\nCreating heatmap...")
    df = create_heatmap(task_year_counts, year_total_papers, all_tasks, all_years)
    
    print("\n" + "="*80)
    print("Task-Year Heatmap created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

