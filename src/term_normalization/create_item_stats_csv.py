#!/usr/bin/env python3
"""
Create CSV files with item statistics from raw unique items
"""
import json
from pathlib import Path
import csv

def clean_task_name(task):
    """Remove 'Other: ' or 'Other:' prefix from task names"""
    if task.startswith("Other: "):
        return task[7:]
    elif task.startswith("Other:"):
        return task[6:]
    return task

def collect_all_items_raw(base_path):
    """Collect all items from verified papers WITHOUT normalization"""
    all_tasks = []
    all_datasets = []
    all_languages = []
    all_models = []
    all_automatic_metrics = []
    all_llm_criteria = []
    all_human_criteria = []

    # Get all conference directories
    conferences = sorted([d for d in base_path.iterdir() if d.is_dir()])

    for conf_dir in conferences:
        for json_file in conf_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                    # Get data from answer_1
                    answer_1 = data.get("answer_1", {})
                    if answer_1.get("answer", "").lower() == "yes":
                        tasks = answer_1.get("tasks", [])
                        datasets = answer_1.get("datasets", [])
                        languages = answer_1.get("languages", [])
                        models = answer_1.get("models", [])

                        # Clean task names but don't normalize
                        cleaned_tasks = [clean_task_name(task) for task in tasks]
                        all_tasks.extend(cleaned_tasks)
                        all_datasets.extend(datasets)
                        all_languages.extend(languages)
                        all_models.extend(models)

                    # Answer 2: Automatic metrics (no normalization)
                    answer_2 = data.get("answer_2", {})
                    if answer_2.get("answer", "").lower() == "yes":
                        metrics = answer_2.get("automatic_metrics", [])
                        all_automatic_metrics.extend(metrics)

                    # Answer 3: LLM criteria (no normalization)
                    answer_3 = data.get("answer_3", {})
                    if answer_3.get("answer", "").lower() == "yes":
                        criteria = answer_3.get("criteria", [])
                        all_llm_criteria.extend(criteria)

                    # Answer 4: Human criteria (no normalization)
                    answer_4 = data.get("answer_4", {})
                    if answer_4.get("answer", "").lower() == "yes":
                        criteria = answer_4.get("criteria", [])
                        all_human_criteria.extend(criteria)

            except (json.JSONDecodeError, KeyError) as e:
                pass

    return {
        'tasks': all_tasks,
        'datasets': all_datasets,
        'languages': all_languages,
        'models': all_models,
        'automatic_metrics': all_automatic_metrics,
        'llm_criteria': all_llm_criteria,
        'human_criteria': all_human_criteria
    }

def create_stats_csv(items, output_file, item_name):
    """Create CSV with item name and count"""
    from collections import Counter

    # Count occurrences
    counter = Counter(items)

    # Sort by count (descending) then by name
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([item_name, 'count'])

        for item, count in sorted_items:
            writer.writerow([item, count])

    print(f"Saved: {output_file}")
    print(f"  Total unique {item_name.lower()}: {len(counter):,}")
    print(f"  Total occurrences: {sum(counter.values()):,}")

def main():
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_path = project_root / "llm-merged-results" 

    # Create CSV output directory
    csv_dir = script_dir / "metadata_unique_counts"
    csv_dir.mkdir(exist_ok=True)

    print("="*80)
    print("Creating Item Statistics CSV Files (Raw Data)")
    print("="*80)

    print("\nCollecting data from verified papers...")
    data = collect_all_items_raw(base_path)

    print(f"\n{'='*80}")
    print("Generating CSV files...")
    print(f"{'='*80}\n")

    # Create CSV for each category
    create_stats_csv(data['tasks'], csv_dir / 'tasks_stats.csv', 'task')
    print()

    create_stats_csv(data['datasets'], csv_dir / 'datasets_stats.csv', 'dataset')
    print()

    create_stats_csv(data['languages'], csv_dir / 'languages_stats.csv', 'language')
    print()

    create_stats_csv(data['models'], csv_dir / 'models_stats.csv', 'model')
    print()

    create_stats_csv(data['automatic_metrics'], csv_dir / 'automatic_metrics_stats.csv', 'metric')
    print()

    create_stats_csv(data['llm_criteria'], csv_dir / 'llm_criteria_stats.csv', 'criterion')
    print()

    create_stats_csv(data['human_criteria'], csv_dir / 'human_criteria_stats.csv', 'criterion')

    print(f"\n{'='*80}")
    print(f"All CSV files saved to: {csv_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
