#!/usr/bin/env python3
"""
Check pairwise agreement between human annotations (Excel) and LLM-extracted data (JSON)
for the four binary answers (answer_1, answer_2, answer_3, answer_4).
"""

import json
import os
import glob
import pandas as pd
from collections import defaultdict
import numpy as np

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Root directory (nlg-eval-llm parent)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'llm-merged-results-top30-tasks')
# Excel file is in the root experiments/nlg-eval directory
EXCEL_FILE = os.path.join(ROOT_DIR, 'human_final_resolved_110.xlsx')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

def load_excel_manual(excel_file):
    """Manually read Excel file using zipfile and XML parsing (fallback)."""
    import zipfile
    import xml.etree.ElementTree as ET
    
    try:
        with zipfile.ZipFile(excel_file, 'r') as z:
            # Read shared strings
            strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                tree = ET.parse(z.open('xl/sharedStrings.xml'))
                root = tree.getroot()
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                for si in root.findall('.//main:si', ns):
                    t = si.find('main:t', ns)
                    if t is not None and t.text:
                        strings.append(t.text)
                    else:
                        strings.append('')
            
            # Read sheet data - handle sparse cells properly
            if 'xl/worksheets/sheet1.xml' in z.namelist():
                tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
                root = tree.getroot()
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                import re
                
                # Parse all cells with their coordinates
                cells_dict = {}
                for cell in root.findall('.//main:c', ns):
                    r = cell.get('r')  # e.g., 'A1', 'B2'
                    if r:
                        # Parse column and row
                        col_match = re.match(r'([A-Z]+)(\d+)', r)
                        if col_match:
                            col_str = col_match.group(1)
                            row_num = int(col_match.group(2))
                            
                            # Convert column letter to index (A=0, B=1, etc.)
                            col_idx = 0
                            for char in col_str:
                                col_idx = col_idx * 26 + (ord(char) - ord('A') + 1)
                            col_idx -= 1
                            
                            # Get cell value
                            v = cell.find('main:v', ns)
                            if v is not None and v.text:
                                t_attr = cell.get('t', '')
                                if t_attr == 's':  # Shared string
                                    idx = int(v.text)
                                    val = strings[idx] if idx < len(strings) else ''
                                else:
                                    val = v.text
                                cells_dict[(row_num - 1, col_idx)] = val  # 0-indexed
                
                # Find max row and column
                if cells_dict:
                    max_row = max(r for r, c in cells_dict.keys())
                    max_col = max(c for r, c in cells_dict.keys())
                    
                    # Build rows
                    rows = []
                    for r in range(max_row + 1):
                        row_data = []
                        for c in range(max_col + 1):
                            row_data.append(cells_dict.get((r, c), ''))
                        rows.append(row_data)
                    
                    if rows:
                        # First row is headers
                        headers = rows[0] if len(rows) > 0 else []
                        data = rows[1:] if len(rows) > 1 else []
                        
                        # If headers are empty, create generic names
                        if not any(str(h).strip() for h in headers):
                            headers = [f'Column_{i+1}' for i in range(len(headers))]
                        
                        df = pd.DataFrame(data, columns=headers)
                        print(f"Manually loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
                        print(f"Column names: {df.columns.tolist()}")
                        return df
    except Exception as e:
        print(f"Manual Excel reading failed: {e}")
        return None
    
    return None

def load_excel_data(excel_file):
    """Load human annotations from Excel file."""
    try:
        # Try with openpyxl engine
        df = pd.read_excel(excel_file, engine='openpyxl')
        print(f"Loaded Excel file with {len(df)} rows and columns: {df.columns.tolist()}")
        return df
    except ImportError:
        print("openpyxl not installed. Trying manual parsing...")
        df = load_excel_manual(excel_file)
        if df is not None:
            return df
        print("\nPlease install openpyxl: pip install openpyxl")
        print("Or convert the Excel file to CSV format.")
        return None
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print(f"File path: {excel_file}")
        print(f"File exists: {os.path.exists(excel_file)}")
        print("Trying manual parsing...")
        df = load_excel_manual(excel_file)
        if df is not None:
            return df
        return None

def load_json_data(data_dir):
    """Load all JSON files and create a mapping from paper_id to data."""
    paper_data = {}
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    paper_id = data.get('paper_id', '')
                    if paper_id:
                        paper_data[paper_id] = data
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    return paper_data

def extract_binary_answer(json_data, answer_num):
    """
    Extract binary answer from JSON data.
    The JSON structure has answer_1, answer_2, answer_3, answer_4, each with an 'answer' field.
    """
    answer_key = f'answer_{answer_num}'
    answer_data = json_data.get(answer_key, {})
    
    # First, try to get the 'answer' field directly (likely binary: Yes/No, 1/0, True/False)
    answer = answer_data.get('answer', '')
    
    # Convert to binary
    if isinstance(answer, bool):
        return 1 if answer else 0
    elif isinstance(answer, (int, float)):
        return 1 if answer > 0 else 0
    elif isinstance(answer, str):
        answer_lower = answer.lower().strip()
        if answer_lower in ['yes', 'y', '1', 'true', 't']:
            return 1
        elif answer_lower in ['no', 'n', '0', 'false', 'f']:
            return 0
    
    # Fallback: check if relevant data exists
    if answer_num == 1:
        # answer_1: tasks
        tasks = answer_data.get('tasks', [])
        return 1 if tasks and len(tasks) > 0 else 0
    elif answer_num == 2:
        # answer_2: automatic_metrics
        metrics = answer_data.get('automatic_metrics', [])
        return 1 if metrics and len(metrics) > 0 else 0
    elif answer_num == 3:
        # answer_3: LLM criteria
        criteria = answer_data.get('criteria', [])
        return 1 if criteria and len(criteria) > 0 else 0
    elif answer_num == 4:
        # answer_4: human criteria
        criteria = answer_data.get('criteria', [])
        return 1 if criteria and len(criteria) > 0 else 0
    
    return None

def calculate_cohen_kappa(y1, y2):
    """Calculate Cohen's kappa coefficient manually."""
    # Create confusion matrix
    n = len(y1)
    if n == 0:
        return None
    
    # Count agreements
    po = np.sum(y1 == y2) / n  # Observed agreement
    
    # Calculate expected agreement
    p1_yes = np.sum(y1 == 1) / n
    p1_no = np.sum(y1 == 0) / n
    p2_yes = np.sum(y2 == 1) / n
    p2_no = np.sum(y2 == 0) / n
    
    pe = (p1_yes * p2_yes) + (p1_no * p2_no)  # Expected agreement
    
    # Calculate kappa
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    
    kappa = (po - pe) / (1 - pe)
    return kappa

def calculate_agreement(human_values, llm_values):
    """Calculate agreement metrics between human and LLM annotations."""
    if len(human_values) != len(llm_values):
        return None
    
    # Convert to numpy arrays
    human_arr = np.array(human_values)
    llm_arr = np.array(llm_values)
    
    # Calculate agreement metrics
    total = len(human_values)
    agreements = np.sum(human_arr == llm_arr)
    agreement_rate = agreements / total if total > 0 else 0
    
    # Calculate Cohen's Kappa
    kappa = calculate_cohen_kappa(human_arr, llm_arr)
    
    # Confusion matrix
    tp = np.sum((human_arr == 1) & (llm_arr == 1))  # Both positive
    tn = np.sum((human_arr == 0) & (llm_arr == 0))  # Both negative
    fp = np.sum((human_arr == 0) & (llm_arr == 1))  # LLM positive, human negative
    fn = np.sum((human_arr == 1) & (llm_arr == 0))  # Human positive, LLM negative
    
    return {
        'total': total,
        'agreements': int(agreements),
        'agreement_rate': agreement_rate,
        'kappa': kappa,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else None,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None,
    }

def main():
    print("="*80)
    print("Pairwise Agreement Analysis: Human vs LLM")
    print("="*80)
    
    # Load Excel file
    print("\n1. Loading human annotations from Excel...")
    df_human = load_excel_data(EXCEL_FILE)
    if df_human is None:
        print("Failed to load Excel file. Exiting.")
        return
    
    print(f"\nExcel columns: {df_human.columns.tolist()}")
    print(f"First few rows:")
    print(df_human.head())
    
    # Load JSON data
    print("\n2. Loading LLM-extracted data from JSON files...")
    paper_data = load_json_data(DATA_DIR)
    print(f"Loaded {len(paper_data)} papers from JSON files")
    
    # Identify paper ID column in Excel
    # Common column names: paper_id, Paper ID, ID, etc.
    paper_id_col = None
    for col in df_human.columns:
        col_str = str(col).strip()
        if col_str and ('paper' in col_str.lower() or 'id' in col_str.lower() or 'paper_id' in col_str.lower()):
            paper_id_col = col
            break
    
    # If not found, try to use first column that looks like an ID
    if paper_id_col is None:
        # Check if first column has paper-like IDs
        first_col = df_human.columns[0]
        sample_val = str(df_human[first_col].iloc[0] if len(df_human) > 0 else '')
        # Check if it looks like a paper ID (contains year or conference abbreviation)
        if any(x in sample_val for x in ['2020', '2021', '2022', '2023', '2024', '2025', 'acl', 'emnlp', 'naacl', 'inlg']):
            paper_id_col = first_col
            print(f"\nUsing first column '{paper_id_col}' as paper ID (inferred from content)")
    
    if paper_id_col is None:
        print("\nWarning: Could not find paper ID column.")
        print("Available columns:", [str(c) for c in df_human.columns.tolist()])
        print("\nFirst few values from each column:")
        for col in df_human.columns[:5]:
            if len(df_human) > 0:
                vals = df_human[col].iloc[:3].values.tolist()
                print(f"  {col}: {vals}")
            else:
                print(f"  {col}: N/A")
        print("\nPlease specify the paper ID column manually or check the Excel file structure.")
        # Try to use first non-empty column
        for col in df_human.columns:
            if str(col).strip() and df_human[col].notna().any():
                paper_id_col = col
                print(f"\nAttempting to use column '{col}' as paper ID...")
                break
        if paper_id_col is None:
            return
    
    print(f"\nUsing '{paper_id_col}' as paper ID column")
    
    # Identify answer columns in Excel
    # We need to find columns for answer_1, answer_2, answer_3, answer_4
    answer_cols = {}
    for i in range(1, 5):
        # Try different possible column names
        possible_names = [
            f'answer_{i}',
            f'Answer {i}',
            f'Q{i}',
            f'Question {i}',
            f'Binary {i}',
            f'answer {i}',
            f'Answer{i}',
        ]
        for name in possible_names:
            if name in df_human.columns:
                answer_cols[i] = name
                break
    
    # If not found by name, try to infer from column positions
    # Often answers are in sequential columns after paper ID
    if len(answer_cols) < 4:
        paper_id_idx = list(df_human.columns).index(paper_id_col) if paper_id_col in df_human.columns else -1
        if paper_id_idx >= 0:
            # Try next 4 columns after paper ID
            for i in range(1, 5):
                if i not in answer_cols and paper_id_idx + i < len(df_human.columns):
                    col = df_human.columns[paper_id_idx + i]
                    answer_cols[i] = col
                    print(f"Inferred answer_{i} from column position: '{col}'")
    
    if len(answer_cols) < 4:
        print(f"\nWarning: Found only {len(answer_cols)} answer columns: {answer_cols}")
        print("Available columns:", [str(c) for c in df_human.columns.tolist()])
        print("\nPlease check the Excel file structure or manually specify answer columns.")
        # Try to use remaining columns
        used_cols = {paper_id_col} | set(answer_cols.values())
        remaining_cols = [c for c in df_human.columns if c not in used_cols]
        for i in range(1, 5):
            if i not in answer_cols and remaining_cols:
                answer_cols[i] = remaining_cols.pop(0)
                print(f"Using column '{answer_cols[i]}' for answer_{i}")
    
    print(f"\nAnswer columns found: {answer_cols}")
    
    # Match papers and extract values
    print("\n3. Matching papers and extracting binary answers...")
    matched_papers = []
    
    for idx, row in df_human.iterrows():
        paper_id = str(row[paper_id_col]).strip()
        
        # Try to find matching JSON file
        json_data = None
        if paper_id in paper_data:
            json_data = paper_data[paper_id]
        else:
            # Try variations of paper_id
            for pid, data in paper_data.items():
                if paper_id in pid or pid in paper_id:
                    json_data = data
                    paper_id = pid  # Use the matched ID
                    break
        
        if json_data is None:
            continue
        
        # Extract human answers
        human_answers = {}
        for i in range(1, 5):
            if i in answer_cols:
                val = row[answer_cols[i]]
                # Convert to binary (0 or 1)
                if pd.isna(val):
                    human_answers[i] = None
                else:
                    # Try to convert to binary
                    if isinstance(val, (int, float)):
                        human_answers[i] = 1 if val > 0 else 0
                    elif isinstance(val, str):
                        val_lower = val.lower().strip()
                        if val_lower in ['yes', 'y', '1', 'true', 't']:
                            human_answers[i] = 1
                        elif val_lower in ['no', 'n', '0', 'false', 'f']:
                            human_answers[i] = 0
                        else:
                            human_answers[i] = None
                    else:
                        human_answers[i] = None
        
        # Extract LLM answers
        llm_answers = {}
        for i in range(1, 5):
            llm_answers[i] = extract_binary_answer(json_data, i)
        
        matched_papers.append({
            'paper_id': paper_id,
            'human_answers': human_answers,
            'llm_answers': llm_answers,
        })
    
    print(f"\nMatched {len(matched_papers)} papers")
    
    # Calculate agreement for each answer
    print("\n4. Calculating pairwise agreement...")
    results = {}
    
    for answer_num in range(1, 5):
        human_values = []
        llm_values = []
        
        for paper in matched_papers:
            human_val = paper['human_answers'].get(answer_num)
            llm_val = paper['llm_answers'].get(answer_num)
            
            # Only include if both values are not None
            if human_val is not None and llm_val is not None:
                human_values.append(human_val)
                llm_values.append(llm_val)
        
        if len(human_values) > 0:
            agreement = calculate_agreement(human_values, llm_values)
            results[answer_num] = agreement
        else:
            results[answer_num] = None
    
    # Print results
    print("\n" + "="*80)
    print("AGREEMENT RESULTS")
    print("="*80)
    
    for answer_num in range(1, 5):
        print(f"\n--- Answer {answer_num} ---")
        if results[answer_num] is None:
            print("No valid data for comparison")
            continue
        
        r = results[answer_num]
        print(f"Total papers compared: {r['total']}")
        print(f"Agreements: {r['agreements']}")
        print(f"Agreement Rate: {r['agreement_rate']:.3f} ({r['agreement_rate']*100:.1f}%)")
        if r['kappa'] is not None:
            print(f"Cohen's Kappa: {r['kappa']:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {r['tp']}")
        print(f"  True Negatives (TN):  {r['tn']}")
        print(f"  False Positives (FP): {r['fp']}")
        print(f"  False Negatives (FN): {r['fn']}")
        if r['precision'] is not None:
            print(f"\nPrecision: {r['precision']:.3f}")
        if r['recall'] is not None:
            print(f"Recall: {r['recall']:.3f}")
        if r['f1'] is not None:
            print(f"F1 Score: {r['f1']:.3f}")
    
    # Save results to CSV
    print("\n5. Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create summary table
    summary_data = []
    for answer_num in range(1, 5):
        if results[answer_num] is not None:
            r = results[answer_num]
            summary_data.append({
                'Answer': answer_num,
                'Total_Papers': r['total'],
                'Agreements': r['agreements'],
                'Agreement_Rate': r['agreement_rate'],
                'Kappa': r['kappa'] if r['kappa'] is not None else None,
                'TP': r['tp'],
                'TN': r['tn'],
                'FP': r['fp'],
                'FN': r['fn'],
                'Precision': r['precision'] if r['precision'] is not None else None,
                'Recall': r['recall'] if r['recall'] is not None else None,
                'F1': r['f1'] if r['f1'] is not None else None,
            })
    
    df_summary = pd.DataFrame(summary_data)
    output_file = os.path.join(OUTPUT_DIR, 'pairwise_agreement_summary.csv')
    df_summary.to_csv(output_file, index=False)
    print(f"Saved summary to {output_file}")
    
    # Save detailed results
    detailed_data = []
    for paper in matched_papers:
        row = {'paper_id': paper['paper_id']}
        for i in range(1, 5):
            row[f'human_answer_{i}'] = paper['human_answers'].get(i)
            row[f'llm_answer_{i}'] = paper['llm_answers'].get(i)
            row[f'match_{i}'] = (paper['human_answers'].get(i) == paper['llm_answers'].get(i)) if (paper['human_answers'].get(i) is not None and paper['llm_answers'].get(i) is not None) else None
        detailed_data.append(row)
    
    df_detailed = pd.DataFrame(detailed_data)
    detailed_file = os.path.join(OUTPUT_DIR, 'pairwise_agreement_detailed.csv')
    df_detailed.to_csv(detailed_file, index=False)
    print(f"Saved detailed results to {detailed_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

