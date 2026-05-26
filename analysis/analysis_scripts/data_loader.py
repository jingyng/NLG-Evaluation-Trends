import csv
import json
import os
import glob
from collections import Counter

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# In the acl2026-nlg-eval layout this file is at
# analysis/analysis_scripts/data_loader.py, so REPO_ROOT is two parents up.
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR   = os.path.join(REPO_ROOT, 'results', 'llm-merged-results-top30-tasks')

# ---------------------------------------------------------------------------
# Criteria are already QCET-normalized at the JSON pre-bake step
# (paper_code/04_postprocessing/normalize_merged_results.py reads
#  metadata_unique_counts/criteria/{llm,human}_criteria_normalization_mapping.csv,
#  which is regenerated from stage4_classifications_simple.csv via
#  paper_code/05_criteria_normalization/apply_qcet_to_metadata.py).
# Short labels for figures live in
#  metadata_unique_counts/criteria/criteria_qcet_short_labels.csv.
# ---------------------------------------------------------------------------

_SHORT_LABELS_CSV = os.path.join(
    REPO_ROOT, 'metadata_unique_counts', 'criteria', 'criteria_qcet_short_labels.csv'
)
_LLM_MAPPING_CSV = os.path.join(
    REPO_ROOT, 'metadata_unique_counts', 'criteria', 'llm_criteria_normalization_mapping.csv'
)
_HUMAN_MAPPING_CSV = os.path.join(
    REPO_ROOT, 'metadata_unique_counts', 'criteria', 'human_criteria_normalization_mapping.csv'
)


def _build_short_label_map() -> dict:
    """Returns {qcet_name_lower: (qcet_name, short_label_bare, short_label_prefixed)}."""
    mapping: dict = {}
    if not os.path.exists(_SHORT_LABELS_CSV):
        return mapping
    with open(_SHORT_LABELS_CSV, newline='') as fh:
        for row in csv.DictReader(fh):
            full = row['qcet_name'].strip()
            bare = row['short_label'].strip()
            prefixed = (row.get('short_label_prefixed') or bare).strip() or bare
            mapping[full.lower()] = (full, bare, prefixed)
    return mapping


_SHORT_LABEL_MAP: dict = _build_short_label_map()


def short_label(criterion: str, prefixed: bool = True) -> str:
    """Map a QCET full name to its short label for figures.  Case-insensitive.

    By default returns the prefixed form '[QO] Coherence' so figures expose
    the QCET frame-of-reference axis. Pass `prefixed=False` to get the bare
    short label 'Coherence' (useful in body/table prose where the axis is
    discussed separately).

    Falls back to the input (preserving its case) if not in the table.
    """
    if criterion is None:
        return ''
    key = criterion.strip()
    hit = _SHORT_LABEL_MAP.get(key.lower())
    if hit is None:
        return key
    return hit[2] if prefixed else hit[1]


# ---------------------------------------------------------------------------
# Raw-string → QCET full-name lookup, used by analyses that read criterion
# strings from sources OTHER than `data/llm-merged-results-top30-tasks/`
# (which is already pre-normalized).  The clearest case is the LaaJ↔Human
# validation pipeline, which extracts criterion strings from validation
# papers separately.  Returns None if the raw string is AUX-Other (caller
# should drop it).
# ---------------------------------------------------------------------------

def _build_raw_to_qcet_map() -> dict:
    """{raw_lower: qcet_full_name | None} merged across LLM + Human mappings."""
    mapping: dict = {}
    for path in (_LLM_MAPPING_CSV, _HUMAN_MAPPING_CSV):
        if not os.path.exists(path):
            continue
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                key = row['original'].strip().lower()
                norm = row['normalized'].strip()
                # Sentinel from apply_qcet_to_metadata: AUX-Other rows
                mapping[key] = None if norm == '__DROP__' else norm
    return mapping


_RAW_TO_QCET_MAP: dict = _build_raw_to_qcet_map()


def normalize_criterion_to_qcet(raw: str):
    """Map a raw criterion string to its QCET full name (case-insensitive).
    Returns the QCET name, or None for AUX-Other / unmapped strings.
    Use short_label() on the result for figure labels."""
    if not raw:
        return None
    key = str(raw).strip().lower()
    if key in _RAW_TO_QCET_MAP:
        return _RAW_TO_QCET_MAP[key]
    return None


def normalize_criteria(criteria_list: list) -> list:
    """Pass-through: criteria are already QCET-normalized in the JSON files.
    Kept for backwards compatibility with callers."""
    return [str(c).strip() for c in criteria_list if c is not None]


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
                    
                    laaj_criteria = normalize_criteria(
                        data.get('answer_3', {}).get('criteria', []))
                    laaj_models = data.get('answer_3', {}).get('models', [])

                    human_criteria = normalize_criteria(
                        data.get('answer_4', {}).get('criteria', []))
                    
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
