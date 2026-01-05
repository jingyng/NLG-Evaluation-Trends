import json
import os
import requests
# import fitz  # PyMuPDF
import re


# Load JSON data from file
# json_file = "./papers/acl/acl_2021_papers.json"  # Replace with your JSON file path
# json_file = "./papers/emnlp/emnlp_2024_papers.json"  # Replace with your JSON file path
# json_file = "./papers/inlg/inlg_2024_papers.json"  # Replace with your JSON file path
json_file = "./papers/naacl/naacl_2025_papers.json"  # Replace with your JSON file path


with open(json_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

# Create the directory if it doesn't exist
# output_dir = "./papers/acl/acl-2021"
# output_dir = "./papers/emnlp/emnlp-2024"
# output_dir = "./papers/inlg/inlg-2024"
output_dir = "./papers/naacl/naacl-2025"
os.makedirs(output_dir, exist_ok=True)

# Download PDFs (filtered by "long" in paper_id)
for paper in papers:
    paper_id = paper["paper_id"]
    
    # Skip papers without "long" in their ID
    if "naacl-long" not in paper_id:
        continue
    
    pdf_url = paper["url"]
    pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
    # txt_path = os.path.join(output_dir, f"{paper_id}.txt")  # Text output path

    try:
        # Download PDF
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # # Extract and clean text
        # text = extract_text_from_two_columns(pdf_path)
        
        # # Save text
        # with open(txt_path, "w", encoding="utf-8") as f:
        #     f.write(text)
        
        print(f"Downloaded: {paper_id}")
    
    except Exception as e:
        print(f"Failed for {paper_id}: {str(e)}")




