import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import krippendorff


def load_model_extractions(conference_path: Path, paper_id: str, model_dirs: List[str]) -> Dict:
    """Load extractions from all models for a single paper."""
    extractions = {}

    for model_dir in model_dirs:
        file_path = conference_path / model_dir / f"{paper_id}.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                extractions[model_dir] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            extractions[model_dir] = None

    return extractions


def extract_answer_value(answer_dict, answer_key: str) -> int:
    """Extract yes/no answer and convert to numeric (1=Yes, 0=No, NaN=missing)."""
    if answer_dict is None or answer_key not in answer_dict:
        return np.nan

    answer = answer_dict[answer_key].get('answer', '').lower()

    if answer == 'yes':
        return 1
    elif answer == 'no':
        return 0
    else:
        return np.nan


def compute_agreement_for_conference(conference_path: Path, model_dirs: List[str]) -> Dict:
    """Compute Krippendorff's alpha for all answers in a conference."""
    # Get all paper IDs
    first_model_dir = conference_path / model_dirs[0]
    if not first_model_dir.exists():
        return None

    paper_files = list(first_model_dir.glob('*.json'))

    # Initialize data structures for each answer
    # Format: {answer_key: [[model1_ratings], [model2_ratings], [model3_ratings]]}
    answer_data = {
        'answer_1': [[], [], []],
        'answer_2': [[], [], []],
        'answer_3': [[], [], []],
        'answer_4': [[], [], []]
    }

    # Collect ratings for each paper
    for paper_file in paper_files:
        paper_id = paper_file.stem
        extractions = load_model_extractions(conference_path, paper_id, model_dirs)

        for answer_key in answer_data.keys():
            for model_idx, model_dir in enumerate(model_dirs):
                rating = extract_answer_value(extractions[model_dir], answer_key)
                answer_data[answer_key][model_idx].append(rating)

    # Compute Krippendorff's alpha for each answer
    results = {
        'conference': conference_path.name,
        'n_papers': len(paper_files),
        'alphas': {}
    }

    for answer_key, ratings in answer_data.items():
        # Convert to numpy array (models x papers)
        ratings_array = np.array(ratings)

        # Calculate Krippendorff's alpha with error handling
        try:
            alpha = krippendorff.alpha(reliability_data=ratings_array, level_of_measurement='nominal')
        except ValueError:
            # Handle cases where there's only one value in domain (perfect agreement on one value)
            # or other edge cases
            alpha = 1.0 if len(np.unique(ratings_array[~np.isnan(ratings_array)])) <= 1 else np.nan

        results['alphas'][answer_key] = alpha

        # Also calculate simple percent agreement
        valid_mask = ~np.isnan(ratings_array).any(axis=0)
        if valid_mask.sum() > 0:
            valid_ratings = ratings_array[:, valid_mask]
            # Check where all three models agree
            agreement = (valid_ratings[0] == valid_ratings[1]) & (valid_ratings[1] == valid_ratings[2])
            percent_agreement = agreement.sum() / len(agreement) * 100
            results['alphas'][f'{answer_key}_percent_agreement'] = percent_agreement
        else:
            results['alphas'][f'{answer_key}_percent_agreement'] = np.nan

    return results


def compute_overall_agreement(conferences: List[Path], model_dirs: List[str]) -> Dict:
    """Compute Krippendorff's alpha across ALL papers from ALL conferences."""
    # Initialize data structures for each answer
    answer_data = {
        'answer_1': [[], [], []],
        'answer_2': [[], [], []],
        'answer_3': [[], [], []],
        'answer_4': [[], [], []]
    }

    total_papers = 0

    # Collect ratings from all conferences
    for conference_path in conferences:
        first_model_dir = conference_path / model_dirs[0]
        if not first_model_dir.exists():
            continue

        paper_files = list(first_model_dir.glob('*.json'))
        total_papers += len(paper_files)

        for paper_file in paper_files:
            paper_id = paper_file.stem
            extractions = load_model_extractions(conference_path, paper_id, model_dirs)

            for answer_key in answer_data.keys():
                for model_idx, model_dir in enumerate(model_dirs):
                    rating = extract_answer_value(extractions[model_dir], answer_key)
                    answer_data[answer_key][model_idx].append(rating)

    # Compute Krippendorff's alpha for each answer
    results = {
        'n_papers': total_papers,
        'alphas': {}
    }

    for answer_key, ratings in answer_data.items():
        # Convert to numpy array (models x papers)
        ratings_array = np.array(ratings)

        # Calculate Krippendorff's alpha with error handling
        try:
            alpha = krippendorff.alpha(reliability_data=ratings_array, level_of_measurement='nominal')
        except ValueError:
            # Handle cases where there's only one value in domain (perfect agreement on one value)
            # or other edge cases
            alpha = 1.0 if len(np.unique(ratings_array[~np.isnan(ratings_array)])) <= 1 else np.nan

        results['alphas'][answer_key] = alpha

        # Also calculate simple percent agreement
        valid_mask = ~np.isnan(ratings_array).any(axis=0)
        if valid_mask.sum() > 0:
            valid_ratings = ratings_array[:, valid_mask]
            # Check where all three models agree
            agreement = (valid_ratings[0] == valid_ratings[1]) & (valid_ratings[1] == valid_ratings[2])
            percent_agreement = agreement.sum() / len(agreement) * 100
            results['alphas'][f'{answer_key}_percent_agreement'] = percent_agreement
        else:
            results['alphas'][f'{answer_key}_percent_agreement'] = np.nan

    return results


def main():
    """Compute inter-annotator agreement across all conferences.

    Reads the three per-LLM extraction directories from the upstream
    `results/llm-annotations/<CONF>/extracted-*` tree (or the equivalent on disk;
    because of size: ~300MB).  Override --papers-dir if it lives elsewhere.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--papers-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / 'papers',
        help='Path to a directory containing <CONF>/extracted-*/ subdirs.',
    )
    args = parser.parse_args()

    papers_dir = args.papers_dir
    if not papers_dir.exists():
        raise SystemExit(f"papers_dir not found: {papers_dir}\n"
                         f"Pass --papers-dir <path/to/papers-dir>.")
    model_dirs = ['extracted-deepseek-r1', 'extracted-gpt-oss-120b', 'extracted-qwen3-235b']

    # Get all conference directories
    conferences = sorted([d for d in papers_dir.iterdir() if d.is_dir()])

    print("Computing Krippendorff's Alpha for inter-annotator agreement")
    print("=" * 80)
    print(f"\nModels: {', '.join(model_dirs)}")
    print(f"Metric: Krippendorff's Alpha (nominal data)\n")

    all_results = []

    # Compute for each conference
    for conference in conferences:
        print(f"Processing {conference.name}...", end=' ')
        results = compute_agreement_for_conference(conference, model_dirs)

        if results:
            all_results.append(results)
            print(f"✓ ({results['n_papers']} papers)")
        else:
            print("✗ (skipped)")

    # Compute overall agreement across all papers
    print("\nComputing overall agreement across all papers...", end=' ')
    overall_results = compute_overall_agreement(conferences, model_dirs)
    print(f"✓ ({overall_results['n_papers']} papers)")

    # Print detailed results
    print("\n" + "=" * 80)
    print("RESULTS BY CONFERENCE")
    print("=" * 80)

    for result in all_results:
        print(f"\n{result['conference']} (n={result['n_papers']})")
        print("-" * 40)
        for answer_key in ['answer_1', 'answer_2', 'answer_3', 'answer_4']:
            alpha = result['alphas'][answer_key]
            pct = result['alphas'][f'{answer_key}_percent_agreement']
            print(f"  {answer_key}: α = {alpha:.4f}, Agreement = {pct:.1f}%")

    # Print overall agreement (computed across all papers)
    print("\n" + "=" * 80)
    print("OVERALL AGREEMENT (computed across all papers)")
    print("=" * 80)
    print(f"\nTotal papers: {overall_results['n_papers']}\n")

    for answer_key in ['answer_1', 'answer_2', 'answer_3', 'answer_4']:
        alpha = overall_results['alphas'][answer_key]
        pct = overall_results['alphas'][f'{answer_key}_percent_agreement']
        print(f"{answer_key}: α = {alpha:.4f}, Agreement = {pct:.1f}%")

    # Compute statistics by conference (mean/SD of per-conference alphas)
    print("\n" + "=" * 80)
    print("STATISTICS BY CONFERENCE (mean/SD of per-conference alphas)")
    print("=" * 80)

    total_papers = sum(r['n_papers'] for r in all_results)
    print(f"\nTotal papers analyzed: {total_papers}")
    print(f"Total conferences: {len(all_results)}\n")

    for answer_key in ['answer_1', 'answer_2', 'answer_3', 'answer_4']:
        alphas = [r['alphas'][answer_key] for r in all_results if not np.isnan(r['alphas'][answer_key])]
        pcts = [r['alphas'][f'{answer_key}_percent_agreement'] for r in all_results
                if not np.isnan(r['alphas'][f'{answer_key}_percent_agreement'])]

        if alphas:
            mean_alpha = np.mean(alphas)
            std_alpha = np.std(alphas)
            mean_pct = np.mean(pcts)
            print(f"{answer_key}:")
            print(f"  Mean α = {mean_alpha:.4f} (SD = {std_alpha:.4f})")
            print(f"  Mean Agreement = {mean_pct:.1f}%")
            print(f"  Range: α ∈ [{min(alphas):.4f}, {max(alphas):.4f}]\n")

    # Save detailed results to CSV
    output_file = Path(__file__).resolve().parent.parent / 'outputs' / 'inter_annotator_agreement.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in all_results:
        row = {
            'conference': result['conference'],
            'n_papers': result['n_papers']
        }
        for answer_key in ['answer_1', 'answer_2', 'answer_3', 'answer_4']:
            row[f'{answer_key}_alpha'] = result['alphas'][answer_key]
            row[f'{answer_key}_agreement_pct'] = result['alphas'][f'{answer_key}_percent_agreement']
        rows.append(row)

    # Add overall row
    overall_row = {
        'conference': 'OVERALL',
        'n_papers': overall_results['n_papers']
    }
    for answer_key in ['answer_1', 'answer_2', 'answer_3', 'answer_4']:
        overall_row[f'{answer_key}_alpha'] = overall_results['alphas'][answer_key]
        overall_row[f'{answer_key}_agreement_pct'] = overall_results['alphas'][f'{answer_key}_percent_agreement']
    rows.append(overall_row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
Krippendorff's Alpha interpretation:
  α ≥ 0.800 : Excellent agreement
  0.667-0.800 : Tentative conclusions possible
  0.600-0.667 : Substantial disagreement
  α < 0.600 : Poor agreement
    """)


if __name__ == '__main__':
    main()
