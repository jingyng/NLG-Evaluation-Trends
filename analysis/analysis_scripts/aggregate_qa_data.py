#!/usr/bin/env python3
"""Aggregate question answering paper data from JSON files."""

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


def extract_paper_info(json_file):
    """Extract information from a single paper JSON file."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    paper_id = data.get('paper_id', '')

    # Extract year and conference from paper_id
    year = int(paper_id.split('.')[0])
    conference = paper_id.split('.')[1].split('-')[0].upper()

    # Answer 2: Automatic metrics
    answer_2 = data.get('answer_2', {})
    has_metrics = answer_2.get('answer', 'No') == 'Yes'
    automatic_metrics = answer_2.get('automatic_metrics', [])

    # Answer 3: LLM-as-judge
    answer_3 = data.get('answer_3', {})
    has_llm_judge = answer_3.get('answer', 'No') == 'Yes'
    llm_models = answer_3.get('models', [])
    llm_criteria = answer_3.get('criteria', [])

    # Answer 4: Human evaluation
    answer_4 = data.get('answer_4', {})
    has_human_eval = answer_4.get('answer', 'No') == 'Yes'
    human_criteria = answer_4.get('criteria', [])

    # Answer 1: Basic info
    answer_1 = data.get('answer_1', {})
    datasets = answer_1.get('datasets', [])
    languages = answer_1.get('languages', [])
    models = answer_1.get('models', [])
    tasks = answer_1.get('tasks', [])

    return {
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
        'metrics': automatic_metrics,
        'languages': languages,
        'datasets': datasets,
        'models': models,
        'llm_models': llm_models,
        'llm_criteria': llm_criteria,
        'human_criteria': human_criteria,
        'tasks': tasks,
    }


TASK_NAME = 'Question Answering'


def aggregate_qa_papers(base_path):
    """Aggregate all QA paper data."""

    base_path = Path(base_path)

    papers_data = []
    metric_combinations = defaultdict(int)
    human_criteria_list = []
    llm_judge_list = []

    # Process all JSON files
    json_files = list(base_path.glob('*/*.json'))

    print(f"Found {len(json_files)} QA papers")

    for json_file in json_files:
        try:
            info = extract_paper_info(json_file)
            if not (len(info['tasks']) == 1 and info['tasks'][0] == TASK_NAME):
                continue
            papers_data.append(info)

            # Track metric combinations
            if info['has_metrics']:
                metrics_tuple = tuple(sorted(info['metrics']))
                metric_combinations[metrics_tuple] += 1

            # Track human criteria
            if info['has_human_eval']:
                for criterion in info['human_criteria']:
                    human_criteria_list.append({
                        'paper_id': info['paper_id'],
                        'year': info['year'],
                        'conference': info['conference'],
                        'criterion': criterion
                    })

            # Track LLM judges
            if info['has_llm_judge']:
                for criterion in info['llm_criteria']:
                    llm_judge_list.append({
                        'paper_id': info['paper_id'],
                        'year': info['year'],
                        'conference': info['conference'],
                        'llm_model': '|'.join(info['llm_models']) if info['llm_models'] else '',
                        'criterion': criterion
                    })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return papers_data, metric_combinations, human_criteria_list, llm_judge_list


def save_aggregated_data(papers_data, metric_combinations, human_criteria_list, llm_judge_list, output_dir):
    """Save aggregated data to CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save papers data
    papers_df = pd.DataFrame(papers_data)

    # Convert lists to pipe-separated strings for CSV
    papers_df['metrics'] = papers_df['metrics'].apply(lambda x: '|'.join(x) if x else '')
    papers_df['languages'] = papers_df['languages'].apply(lambda x: '|'.join(x) if x else '')
    papers_df['datasets'] = papers_df['datasets'].apply(lambda x: '|'.join(x) if x else '')
    papers_df['models'] = papers_df['models'].apply(lambda x: '|'.join(x) if x else '')
    papers_df['llm_models'] = papers_df['llm_models'].apply(lambda x: '|'.join(x) if x else '')
    papers_df['llm_criteria'] = papers_df['llm_criteria'].apply(lambda x: '|'.join(x) if x else '')
    papers_df['human_criteria'] = papers_df['human_criteria'].apply(lambda x: '|'.join(x) if x else '')

    papers_csv = output_dir / 'qa_papers_aggregated.csv'
    papers_df.to_csv(papers_csv, index=False)
    print(f"Saved aggregated papers to: {papers_csv}")

    # Save metric combinations
    metric_combos_df = pd.DataFrame([
        {'metrics': '|'.join(metrics), 'count': count}
        for metrics, count in sorted(metric_combinations.items(), key=lambda x: x[1], reverse=True)
    ])
    combos_csv = output_dir / 'qa_metric_combinations.csv'
    metric_combos_df.to_csv(combos_csv, index=False)
    print(f"Saved metric combinations to: {combos_csv}")

    # Save human criteria
    if human_criteria_list:
        human_df = pd.DataFrame(human_criteria_list)
        human_csv = output_dir / 'qa_human_criteria.csv'
        human_df.to_csv(human_csv, index=False)
        print(f"Saved human criteria to: {human_csv}")

    # Save LLM judge data
    if llm_judge_list:
        llm_df = pd.DataFrame(llm_judge_list)
        llm_csv = output_dir / 'qa_llm_judges.csv'
        llm_df.to_csv(llm_csv, index=False)
        print(f"Saved LLM judge data to: {llm_csv}")

    # Print statistics
    print("\n" + "="*60)
    print("QA PAPERS STATISTICS")
    print("="*60)
    print(f"Total papers: {len(papers_df)}")
    print(f"Papers with automatic metrics: {papers_df['has_metrics'].sum()} ({papers_df['has_metrics'].sum() / len(papers_df) * 100:.1f}%)")
    print(f"Papers with human evaluation: {papers_df['has_human_eval'].sum()} ({papers_df['has_human_eval'].sum() / len(papers_df) * 100:.1f}%)")
    print(f"Papers with LLM-as-judge: {papers_df['has_llm_judge'].sum()} ({papers_df['has_llm_judge'].sum() / len(papers_df) * 100:.1f}%)")

    print("\nBy year:")
    year_stats = papers_df.groupby('year').size()
    for year, count in sorted(year_stats.items()):
        print(f"  {year}: {count} papers")

    print("\nBy conference:")
    conf_stats = papers_df.groupby('conference').size()
    for conf, count in sorted(conf_stats.items()):
        print(f"  {conf}: {count} papers")

    print("="*60)


def main():
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / 'data' / 'llm-merged-results-top30-tasks'
    output_dir = script_dir / 'qa_analysis_data'

    print(f"Aggregating QA papers from: {base_path}\n")

    papers_data, metric_combinations, human_criteria_list, llm_judge_list = aggregate_qa_papers(base_path)

    save_aggregated_data(papers_data, metric_combinations, human_criteria_list, llm_judge_list, output_dir)


if __name__ == "__main__":
    main()
