from bs4 import BeautifulSoup
import json
import requests
import os
# Load the HTML content from the uploaded file

# file_path = './papers/acl/acl-2020.html'
# file_path = './papers/inlg/inlg-2024.html'
# file_path = './papers/emnlp/emnlp-2024.html'
file_path = './papers/naacl/naacl-2025.html'

with open(file_path, 'r', encoding='utf-8') as file:

    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')
# Extract paper details into a JSON file (excluding download logic for papers)
output_data = []
papers = soup.find_all('p', class_='d-sm-flex align-items-stretch')

base_url = "https://aclanthology.org/"

# Extracting papers starting from the specified one
start_processing = False
for paper in papers:
    # Extract the PDF URL
    pdf_tag = paper.find('a', href=True, title="Open PDF")
    if pdf_tag:
        pdf_url = pdf_tag['href']
        
        # Determine when to start processing
        # if "2021.acl-long.1.pdf" in pdf_url:
        # if "2020.acl-main.1" in pdf_url: # for 2020
        # if "2024.inlg-main.1" in pdf_url: # for inlg
        # if "2024.emnlp-main.1" in pdf_url: # for emnlp
        if "2025.naacl-long.1" in pdf_url:
            start_processing = True
        if not start_processing:
            continue

        # Extract title
        title_tag = paper.find('strong')
        title = title_tag.text.strip() if title_tag else None

        # Extract authors
        authors = [a.text.strip() for a in paper.find_all('a', href=True) if '/people/' in a['href']]

        # Extract abstract
        abstract_div = paper.find_next_sibling('div', class_='card bg-light mb-2 mb-lg-3 collapse abstract-collapse')
        abstract = abstract_div.text.strip() if abstract_div else None


        # Extract BibTeX URL
        bib_tag = paper.find('a', href=True, title="Export to BibTeX")
        bib_url = bib_tag['href'] if bib_tag else None

        # Extract volume and category info
        parent_heading = paper.find_previous('h4', class_='d-sm-flex pb-2 border-bottom')
        volume_info = parent_heading.text.strip() if parent_heading else None

        # Derive paper ID from the PDF URL
        paper_id = pdf_url.split('/')[-1].replace('.pdf', '')

        # Add enhanced paper data to the output
        paper_data = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": pdf_url,
            "bibtex_url": bib_url,
            "volume_info": volume_info
        }
        output_data.append(paper_data)

# Save all papers' information into a JSON file
output_path = './papers/naacl/naacl_2025_papers.json'
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)
