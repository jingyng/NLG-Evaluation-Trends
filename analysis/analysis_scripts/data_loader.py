import json
import os
import glob
from collections import Counter

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Data dir is parallel to analysis folder, inside nlg-eval-llm
# Structure:
# nlg-eval-llm/
#   analysis/
#     data_loader.py
#   llm-merged-results-top30-tasks/
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'llm-merged-results-top30-tasks')

def load_data():
    all_papers = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract Year
                    year = "Unknown"
                    if 'paper_id' in data:
                        parts = data['paper_id'].split('.')
                        if parts[0].isdigit() and len(parts[0]) == 4:
                            year = int(parts[0])
                    
                    if year == "Unknown":
                        folder = os.path.basename(root)
                        if '-' in folder:
                            try:
                                year = int(folder.split('-')[-1])
                            except:
                                pass

                    # Extract fields
                    tasks = data.get('answer_1', {}).get('tasks', [])
                    datasets = data.get('answer_1', {}).get('datasets', [])
                    models = data.get('answer_1', {}).get('models', [])
                    languages = data.get('answer_1', {}).get('languages', [])
                    
                    auto_metrics = data.get('answer_2', {}).get('automatic_metrics', [])
                    
                    laaj_criteria = data.get('answer_3', {}).get('criteria', [])
                    laaj_models = data.get('answer_3', {}).get('models', [])
                    
                    human_criteria = data.get('answer_4', {}).get('criteria', [])
                    
                    paper_info = {
                        'paper_id': data.get('paper_id', 'unknown'),
                        'year': year,
                        'tasks': tasks,
                        'datasets': datasets,
                        'models': models,
                        'languages': languages,
                        'auto_metrics': auto_metrics,
                        'laaj_criteria': laaj_criteria,
                        'laaj_models': laaj_models,
                        'human_criteria': human_criteria
                    }
                    all_papers.append(paper_info)
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return all_papers

if __name__ == "__main__":
    print(f"Loading data from {DATA_DIR}")
    papers = load_data()
    print(f"Loaded {len(papers)} papers.")
    if papers:
        years = sorted(list(set(p['year'] for p in papers if isinstance(p['year'], int))))
        print(f"Years: {years}")
        
        all_tasks = [t for p in papers for t in p['tasks']]
        print(f"Top 5 Tasks: {Counter(all_tasks).most_common(5)}")
