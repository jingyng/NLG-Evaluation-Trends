#!/usr/bin/env python3
"""Aggregate data from MT papers for visualization."""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd


def extract_year_from_paper_id(paper_id):
    """Extract year from paper ID like '2023.acl-long.10'."""
    parts = paper_id.split('.')
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None


def extract_conference_from_paper_id(paper_id):
    """Extract conference from paper ID like '2023.acl-long.10'."""
    parts = paper_id.split('.')
    if len(parts) > 1:
        conf = parts[1].split('-')[0].upper()
        return conf
    return None


TASK_NAME = 'Machine Translation'


def aggregate_mt_papers(base_path):
    """Aggregate all data from MT papers."""

    papers_data = []
    metric_combinations = []
    human_criteria_list = []
    llm_judge_list = []

    # Conference-level aggregations
    conference_stats = defaultdict(lambda: {
        'total_papers': 0,
        'papers_with_metrics': 0,
        'papers_with_human': 0,
        'papers_with_llm': 0,
        'total_metrics': 0,
        'total_languages': 0,
        'total_datasets': 0,
        'unique_metrics': set(),
        'unique_languages': set(),
        'unique_datasets': set(),
        'years': set()
    })

    # Year-level aggregations
    year_stats = defaultdict(lambda: {
        'total_papers': 0,
        'papers_with_metrics': 0,
        'papers_with_human': 0,
        'papers_with_llm': 0,
        'metrics_counter': Counter(),
        'human_criteria_counter': Counter(),
        'llm_models_counter': Counter()
    })

    for json_file in base_path.glob('*/*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            paper_id = data.get('paper_id', '')
            year = extract_year_from_paper_id(paper_id)
            conference = extract_conference_from_paper_id(paper_id)

            if not year or not conference:
                continue

            # Answer 2: Automatic metrics
            answer_2 = data.get('answer_2', {})
            has_metrics = answer_2.get('answer', '').lower() == 'yes'
            automatic_metrics = answer_2.get('automatic_metrics', [])

            # Answer 3: LLM as judge
            answer_3 = data.get('answer_3', {})
            has_llm_judge = answer_3.get('answer', '').lower() == 'yes'
            llm_models = answer_3.get('models', [])
            llm_methods = answer_3.get('methods', [])
            llm_criteria = answer_3.get('criteria', [])

            # Answer 4: Human evaluation
            answer_4 = data.get('answer_4', {})
            has_human_eval = answer_4.get('answer', '').lower() == 'yes'
            human_guideline = answer_4.get('guideline', [])
            human_criteria = answer_4.get('criteria', [])

            # Answer 1: Tasks, datasets, languages, models
            answer_1 = data.get('answer_1', {})
            languages = answer_1.get('languages', [])
            datasets = answer_1.get('datasets', [])
            models = answer_1.get('models', [])
            tasks = answer_1.get('tasks', [])

            # Filter: keep only single-task papers whose task is the target task.
            if not (len(tasks) == 1 and tasks[0] == TASK_NAME):
                continue

            # Store paper-level data
            paper_record = {
                'paper_id': paper_id,
                'year': year,
                'conference': conference,
                'has_metrics': has_metrics,
                'has_llm_judge': has_llm_judge,
                'has_human_eval': has_human_eval,
                'num_metrics': len(automatic_metrics),
                'num_languages': len(languages),
                'num_datasets': len(datasets),
                'num_models': len(models),
                'metrics': '|'.join(automatic_metrics),
                'languages': '|'.join(languages),
                'datasets': '|'.join(datasets),
                'models': '|'.join(models),
                'llm_models': '|'.join(llm_models),
                'llm_criteria': '|'.join(llm_criteria),
                'human_criteria': '|'.join(human_criteria)
            }
            papers_data.append(paper_record)

            # Store metric combinations for UpSet plot
            if automatic_metrics:
                metric_combinations.append({
                    'paper_id': paper_id,
                    'metrics': automatic_metrics
                })

            # Store human criteria
            if human_criteria:
                for criterion in human_criteria:
                    human_criteria_list.append({
                        'paper_id': paper_id,
                        'criterion': criterion,
                        'year': year,
                        'conference': conference
                    })

            # Store LLM judge info
            if has_llm_judge:
                for criterion in llm_criteria:
                    llm_judge_list.append({
                        'paper_id': paper_id,
                        'llm_model': '|'.join(llm_models) if llm_models else '',
                        'criterion': criterion,
                        'year': year,
                        'conference': conference
                    })

            # Update conference stats
            conference_stats[conference]['total_papers'] += 1
            conference_stats[conference]['years'].add(year)
            if has_metrics:
                conference_stats[conference]['papers_with_metrics'] += 1
                conference_stats[conference]['total_metrics'] += len(automatic_metrics)
                conference_stats[conference]['unique_metrics'].update(automatic_metrics)
            if has_human_eval:
                conference_stats[conference]['papers_with_human'] += 1
            if has_llm_judge:
                conference_stats[conference]['papers_with_llm'] += 1

            conference_stats[conference]['total_languages'] += len(languages)
            conference_stats[conference]['total_datasets'] += len(datasets)
            conference_stats[conference]['unique_languages'].update(languages)
            conference_stats[conference]['unique_datasets'].update(datasets)

            # Update year stats
            year_stats[year]['total_papers'] += 1
            if has_metrics:
                year_stats[year]['papers_with_metrics'] += 1
                year_stats[year]['metrics_counter'].update(automatic_metrics)
            if has_human_eval:
                year_stats[year]['papers_with_human'] += 1
                year_stats[year]['human_criteria_counter'].update(human_criteria)
            if has_llm_judge:
                year_stats[year]['papers_with_llm'] += 1
                year_stats[year]['llm_models_counter'].update(llm_models)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return (papers_data, metric_combinations, human_criteria_list,
            llm_judge_list, conference_stats, year_stats)


def save_aggregated_data(output_dir, papers_data, metric_combinations,
                         human_criteria_list, llm_judge_list,
                         conference_stats, year_stats):
    """Save aggregated data to CSV files."""

    output_dir.mkdir(exist_ok=True)

    # Save paper-level data
    df_papers = pd.DataFrame(papers_data)
    df_papers.to_csv(output_dir / 'mt_papers_aggregated.csv', index=False)
    print(f"Saved {len(df_papers)} papers to mt_papers_aggregated.csv")

    # Save metric combinations
    df_metrics = pd.DataFrame(metric_combinations)
    df_metrics.to_csv(output_dir / 'mt_metric_combinations.csv', index=False)
    print(f"Saved {len(df_metrics)} metric combinations")

    # Save human criteria
    if human_criteria_list:
        df_human = pd.DataFrame(human_criteria_list)
        df_human.to_csv(output_dir / 'mt_human_criteria.csv', index=False)
        print(f"Saved {len(df_human)} human criteria entries")

    # Save LLM judge data
    if llm_judge_list:
        df_llm = pd.DataFrame(llm_judge_list)
        df_llm.to_csv(output_dir / 'mt_llm_judges.csv', index=False)
        print(f"Saved {len(df_llm)} LLM judge entries")

    # Save conference stats
    conf_stats_records = []
    for conf, stats in conference_stats.items():
        conf_stats_records.append({
            'conference': conf,
            'total_papers': stats['total_papers'],
            'papers_with_metrics': stats['papers_with_metrics'],
            'papers_with_human': stats['papers_with_human'],
            'papers_with_llm': stats['papers_with_llm'],
            'avg_metrics_per_paper': stats['total_metrics'] / stats['total_papers'] if stats['total_papers'] > 0 else 0,
            'avg_languages_per_paper': stats['total_languages'] / stats['total_papers'] if stats['total_papers'] > 0 else 0,
            'avg_datasets_per_paper': stats['total_datasets'] / stats['total_papers'] if stats['total_papers'] > 0 else 0,
            'unique_metrics': len(stats['unique_metrics']),
            'unique_languages': len(stats['unique_languages']),
            'unique_datasets': len(stats['unique_datasets']),
            'years': '|'.join(map(str, sorted(stats['years'])))
        })

    df_conf_stats = pd.DataFrame(conf_stats_records)
    df_conf_stats.to_csv(output_dir / 'mt_conference_stats.csv', index=False)
    print(f"Saved conference stats for {len(df_conf_stats)} conferences")

    # Save year stats
    year_stats_records = []
    for year, stats in sorted(year_stats.items()):
        year_stats_records.append({
            'year': year,
            'total_papers': stats['total_papers'],
            'papers_with_metrics': stats['papers_with_metrics'],
            'papers_with_human': stats['papers_with_human'],
            'papers_with_llm': stats['papers_with_llm'],
            'metric_diversity': len(stats['metrics_counter']),
            'top_metric': stats['metrics_counter'].most_common(1)[0][0] if stats['metrics_counter'] else '',
            'top_metric_count': stats['metrics_counter'].most_common(1)[0][1] if stats['metrics_counter'] else 0
        })

    df_year_stats = pd.DataFrame(year_stats_records)
    df_year_stats.to_csv(output_dir / 'mt_year_stats.csv', index=False)
    print(f"Saved year stats for {len(df_year_stats)} years")

    return df_papers, df_conf_stats, df_year_stats


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_path = project_root / 'data' / 'llm-merged-results-top30-tasks'
    output_dir = script_dir / 'mt_analysis_data'

    print("Aggregating MT paper data...")
    print(f"Reading from: {base_path}  (filter: single-task = {TASK_NAME!r})")

    (papers_data, metric_combinations, human_criteria_list,
     llm_judge_list, conference_stats, year_stats) = aggregate_mt_papers(base_path)

    print("\nSaving aggregated data...")
    save_aggregated_data(output_dir, papers_data, metric_combinations,
                        human_criteria_list, llm_judge_list,
                        conference_stats, year_stats)

    print("\n" + "="*60)
    print("Data aggregation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
