import xmltodict
import json
import os

def convert_grobid_xml_to_json(xml_content):
    doc = xmltodict.parse(xml_content)
    tei_header = doc.get('TEI', {}).get('teiHeader', {})
    text = doc.get('TEI', {}).get('text', {})
    
    output = {
        "title": extract_title(tei_header),
        "authors": extract_authors(tei_header),
        "abstract": extract_abstract(tei_header),
        "sections": extract_sections(text),
        "references": extract_references(text)
    }
    return output

def extract_title(header):
    return header.get('fileDesc', {}).get('titleStmt', {}).get('title', {}).get('#text', '')

def extract_authors(header):
    authors = []
    title_stmt = header.get('fileDesc', {}).get('titleStmt', {})
    authors_xml = title_stmt.get('author', [])

    # Fallback: Check for authors under <sourceDesc>
    if not authors_xml:
        source_desc = header.get('fileDesc', {}).get('sourceDesc', {})
        bibl_struct = source_desc.get('biblStruct', {})
        analytic = bibl_struct.get('analytic', {})
        authors_xml = analytic.get('author', [])

    if not isinstance(authors_xml, list):
        authors_xml = [authors_xml]

    for author in authors_xml:
        pers_name = author.get('persName', {})

        # Handle nested/structured forename/surname
        forename = pers_name.get('forename', '')
        if isinstance(forename, dict):
            forename = forename.get('#text', '')  # Extract text from dict
        forename_clean = forename.strip() if isinstance(forename, str) else ''

        surname = pers_name.get('surname', '')
        if isinstance(surname, dict):
            surname = surname.get('#text', '')
        surname_clean = surname.strip() if isinstance(surname, str) else ''

        full_name = f"{forename_clean} {surname_clean}".strip()

        # Extract affiliations (handle strings and structured data)
        affiliations = []
        aff_list = author.get('affiliation', [])
        if not isinstance(aff_list, list):
            aff_list = [aff_list] if aff_list else []

        for aff in aff_list:
            institution = ""
            laboratory = ""

            if isinstance(aff, str):
                # Plain text affiliation (e.g., <affiliation>University X</affiliation>)
                institution = aff.strip()
            elif isinstance(aff, dict):
                # Structured affiliation (e.g., with <orgName>)
                org_names = aff.get('orgName', [])
                if isinstance(org_names, dict):
                    org_names = [org_names]
                
                # Extract institution and laboratory
                if len(org_names) > 0:
                    institution = org_names[0].get('#text', '').strip()
                if len(org_names) > 1:
                    laboratory = org_names[1].get('#text', '').strip()
                elif not org_names:
                    # Fallback to raw text in affiliation
                    institution = aff.get('#text', '').strip()

            affiliations.append({
                "institution": institution,
                "laboratory": laboratory
            })

        authors.append({
            "fullName": full_name,
            "firstName": forename_clean,
            "lastName": surname_clean,
            "affiliations": affiliations,
            "email": author.get('email', '')
        })

    return authors

def extract_abstract(header):
    abstract_content = ""
    profile_desc = header.get('profileDesc', {})
    abstract_data = profile_desc.get('abstract', {})

    paragraphs = []

    # Handle case where <div> contains multiple paragraphs
    abstract_div = abstract_data.get('div', [])
    if isinstance(abstract_div, dict):  # If there's only one <div>
        abstract_div = [abstract_div]  # Convert to list

    for div in abstract_div:  # Iterate over all div elements
        p_content = div.get('p', [])
        if isinstance(p_content, dict):  # Single <p> case
            paragraphs.append(p_content.get('#text', ''))
        elif isinstance(p_content, list):  # Multiple <p> case
            for p in p_content:
                paragraphs.append(p.get('#text', '') if isinstance(p, dict) else str(p))
        else:  # Handle unexpected text format
            paragraphs.append(str(p_content))

    # Fallback to direct <p> extraction if <div> is empty
    if not paragraphs:
        abstract_p = abstract_data.get('p', [])
        if isinstance(abstract_p, dict):
            paragraphs.append(abstract_p.get('#text', ''))
        elif isinstance(abstract_p, list):
            paragraphs.extend([p.get('#text', '') if isinstance(p, dict) else str(p) for p in abstract_p])
        else:
            paragraphs.append(str(abstract_p))

    return ' '.join(paragraphs).strip()


def extract_sections(text):
    sections = []
    body = text.get('body', {}).get('div', [])
    if not isinstance(body, list):
        body = [body]
    
    for div in body:
        sections.append(process_section(div))
    
    return sections

def process_section(div):
    paragraphs = []
    p_content = div.get('p', [])
    
    if isinstance(p_content, list):
        for p in p_content:
            if isinstance(p, dict):
                paragraphs.append(p.get('#text', ''))
            else:
                paragraphs.append(str(p))
    elif isinstance(p_content, dict):
        paragraphs.append(p_content.get('#text', ''))
    else:
        paragraphs.append(str(p_content))
    
    section = {
        "heading": div.get('head', ''),
        "text": ' '.join(paragraphs),
        "subsections": []
    }
    
    if 'div' in div:
        subsections = div['div'] if isinstance(div['div'], list) else [div['div']]
        for sub_div in subsections:
            section['subsections'].append(process_section(sub_div))
    
    return section

def extract_references(text):
    references = []
    back = text.get('back', {}).get('div', {})
    if 'listBibl' in back:
        bibl_entries = back['listBibl'].get('biblStruct', [])
        if not isinstance(bibl_entries, list):
            bibl_entries = [bibl_entries]
        
        for bibl in bibl_entries:
            analytic = bibl.get('analytic', {})
            ref = {
                "title": analytic.get('title', ''),
                "authors": []
            }
            
            authors = analytic.get('author', [])
            if not isinstance(authors, list):
                authors = [authors]
            
            for author in authors:
                pers_name = author.get('persName', {})
                full_name = f"{pers_name.get('forename', '')} {pers_name.get('surname', '')}".strip()
                ref['authors'].append({"fullName": full_name})
            
            references.append(ref)
    
    return references

def batch_convert_xml_to_json(input_dir, output_dir):
    """
    Convert all XML files in input_dir to JSON files in output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all XML files
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(input_dir, filename)
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            try:
                # Read XML content
                with open(xml_path, "r") as xml_file:
                    xml_content = xml_file.read()
                
                # Convert to JSON
                json_data = convert_grobid_xml_to_json(xml_content)
                
                # Save JSON
                with open(json_path, "w") as json_file:
                    json.dump(json_data, json_file, indent=2)
                    
                print(f"Converted: {filename} -> {json_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert GROBID XML files to JSON")
    parser.add_argument("--input", required=True, help="Input directory containing XML files")
    parser.add_argument("--output", required=True, help="Output directory for JSON files")
    
    args = parser.parse_args()

    batch_convert_xml_to_json(args.input, args.output)
# Example usage

# with open("./thesis_example.xml", "r") as f:
#     xml_content = f.read()

# json_output = convert_grobid_xml_to_json(xml_content)
# with open("./thesis_example.json", "w") as f:
#     json.dump(json_output, f, indent=2)