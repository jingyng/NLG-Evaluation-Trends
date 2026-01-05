import json
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any


def load_extraction(file_path: Path) -> Dict:
    """Load a single extraction JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def majority_vote(answers: List[str]) -> str:
    """Determine majority vote for yes/no answers."""
    valid_answers = [a.lower() for a in answers if a and a.lower() in ['yes', 'no']]
    if not valid_answers:
        return "No"

    count = Counter(valid_answers)
    # Return the most common answer (capitalized)
    return count.most_common(1)[0][0].capitalize()


def combine_unique_items(lists: List[List]) -> List:
    """Combine multiple lists and return unique items, preserving order."""
    seen = set()
    result = []
    for lst in lists:
        if lst:
            for item in lst:
                if item and item not in seen:
                    seen.add(item)
                    result.append(item)
    return result


def combine_strings(strings: List) -> str:
    """Combine non-empty strings or lists, removing duplicates."""
    unique_strings = []
    seen = set()
    for s in strings:
        # Handle both string and list types
        if isinstance(s, list):
            for item in s:
                if item and item not in seen:
                    seen.add(item)
                    unique_strings.append(item)
        elif s and s not in seen:
            seen.add(s)
            unique_strings.append(s)
    return " | ".join(unique_strings)


def merge_answer(answer_key: str, extractions: Dict[str, Dict], model_names: List[str]) -> Dict:
    """Merge a specific answer across all models."""
    answers = []
    quotes = []

    # Collect data from all models
    for model in model_names:
        if extractions[model] and answer_key in extractions[model]:
            answer_data = extractions[model][answer_key]
            answers.append(answer_data.get('answer', 'No'))
            quote = answer_data.get('quote', '')
            if quote:
                quotes.append(quote)

    # Majority vote for yes/no answer
    final_answer = majority_vote(answers)

    # Build merged answer
    merged = {
        'answer': final_answer,
        'quote': quotes
    }

    # Merge specific fields based on answer type
    if answer_key == 'answer_1':
        tasks = [extractions[m][answer_key].get('tasks', []) for m in model_names if extractions[m] and answer_key in extractions[m]]
        datasets = [extractions[m][answer_key].get('datasets', []) for m in model_names if extractions[m] and answer_key in extractions[m]]
        languages = [extractions[m][answer_key].get('languages', []) for m in model_names if extractions[m] and answer_key in extractions[m]]
        models = [extractions[m][answer_key].get('models', []) for m in model_names if extractions[m] and answer_key in extractions[m]]

        # Collect outputs as list of strings
        outputs_list = []
        for m in model_names:
            if extractions[m] and answer_key in extractions[m]:
                output_data = extractions[m][answer_key].get('outputs', '')
                if output_data:
                    # Handle both string and list types
                    if isinstance(output_data, list):
                        outputs_list.extend(output_data)
                    else:
                        outputs_list.append(output_data)

        # Remove duplicates while preserving order
        seen = set()
        unique_outputs = []
        for item in outputs_list:
            if item and item not in seen:
                seen.add(item)
                unique_outputs.append(item)

        merged['tasks'] = combine_unique_items(tasks)
        merged['datasets'] = combine_unique_items(datasets)
        merged['languages'] = combine_unique_items(languages)
        merged['models'] = combine_unique_items(models)
        merged['outputs'] = unique_outputs

    elif answer_key == 'answer_2':
        metrics = [extractions[m][answer_key].get('automatic_metrics', []) for m in model_names if extractions[m] and answer_key in extractions[m]]
        merged['automatic_metrics'] = combine_unique_items(metrics)

    elif answer_key == 'answer_3':
        models = [extractions[m][answer_key].get('models', []) for m in model_names if extractions[m] and answer_key in extractions[m]]
        methods = [extractions[m][answer_key].get('methods', []) for m in model_names if extractions[m] and answer_key in extractions[m]]
        criteria = [extractions[m][answer_key].get('criteria', []) for m in model_names if extractions[m] and answer_key in extractions[m]]

        merged['models'] = combine_unique_items(models)
        merged['methods'] = combine_unique_items(methods)
        merged['criteria'] = combine_unique_items(criteria)

    elif answer_key == 'answer_4':
        # Collect guidelines as list
        guidelines_list = []
        for m in model_names:
            if extractions[m] and answer_key in extractions[m]:
                guideline_data = extractions[m][answer_key].get('guideline', '')
                if guideline_data:
                    guidelines_list.append(guideline_data)

        criteria = [extractions[m][answer_key].get('criteria', []) for m in model_names if extractions[m] and answer_key in extractions[m]]

        merged['guideline'] = guidelines_list
        merged['criteria'] = combine_unique_items(criteria)

    return merged


def merge_paper_extractions(paper_id: str, conference_path: Path, model_dirs: List[str]) -> Dict:
    """Merge extractions for a single paper from all models."""
    extractions = {}

    # Load extractions from each model
    for model_dir in model_dirs:
        file_path = conference_path / model_dir / f"{paper_id}.json"
        extractions[model_dir] = load_extraction(file_path)

    # Merge each answer
    merged_paper = {}
    for answer_key in ['answer_1', 'answer_2', 'answer_3', 'answer_4']:
        merged_paper[answer_key] = merge_answer(answer_key, extractions, model_dirs)

    return merged_paper


def process_conference(conference_path: Path, model_dirs: List[str]):
    """Process all papers in a conference and create merged directory."""
    print(f"Processing {conference_path.name}...")

    # Create merged directory
    merged_dir = conference_path / 'merged'
    merged_dir.mkdir(exist_ok=True)

    # Get all paper IDs from the first model directory
    first_model_dir = conference_path / model_dirs[0]
    if not first_model_dir.exists():
        print(f"  Skipping - {model_dirs[0]} directory not found")
        return

    paper_files = list(first_model_dir.glob('*.json'))
    print(f"  Found {len(paper_files)} papers")

    # Process each paper
    merged_count = 0
    for paper_file in paper_files:
        paper_id = paper_file.stem

        # Merge extractions
        merged_data = merge_paper_extractions(paper_id, conference_path, model_dirs)

        # Save merged result
        output_path = merged_dir / f"{paper_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)

        merged_count += 1

    print(f"  Merged {merged_count} papers -> {merged_dir}")


def main():
    """Main function to merge all conference extractions."""
    papers_dir = Path('./papers')
    model_dirs = ['extracted-deepseek-r1', 'extracted-gpt-oss-120b', 'extracted-qwen3-235b']

    # Get all conference directories
    conferences = [d for d in papers_dir.iterdir() if d.is_dir()]

    print(f"Found {len(conferences)} conferences\n")

    for conference in sorted(conferences):
        process_conference(conference, model_dirs)

    print("\nMerging complete!")


if __name__ == '__main__':
    main()
