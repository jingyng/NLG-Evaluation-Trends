import json
import os
import time
import argparse
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


def setup_novita_client(api_key: str):
    """Setup Novita AI client."""
    return OpenAI(
        base_url="https://api.novita.ai/v3/openai",
        api_key=api_key
    )


def extract_json_from_response(content: str) -> str:
    """Extract JSON from response that may contain markdown code blocks or surrounding text."""
    # Try to find JSON in markdown code block first
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', content)
    if json_match:
        return json_match.group(1).strip()

    # Try to find JSON object directly (starts with { and ends with })
    start_idx = content.find('{')
    end_idx = content.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return content[start_idx:end_idx + 1]

    # Return original content if no JSON found
    return content


def validate_result(result: dict) -> bool:
    """Validate that the result has the expected structure."""
    if not result or not isinstance(result, dict):
        return False

    # Required fields that should always be present
    required_fields = ['paper_id', 'explicit_validation', 'llm_judge_details', 'human_eval_details', 'criteria_mapping']
    return all(field in result for field in required_fields)


def run_extraction(client, prompt_text: str, model: str = "deepseek/deepseek-v3.1-terminus", max_retries: int = 3):
    """Run validation extraction through Novita API with retry logic."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                response_format={"type": "json_object"},
                # Note: temperature is ignored by reasoning models like deepseek-v3.1-terminus
            )

            # Debug: Check response structure
            if not response.choices:
                error_msg = "No choices in response"
                if attempt < max_retries - 1:
                    print(f"\n  Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None, error_msg

            message = response.choices[0].message
            content = message.content

            # Check if reasoning_content is available separately (for reasoning models)
            reasoning_content = getattr(message, 'reasoning_content', None)
            if reasoning_content:
                print(f"\n  [INFO] Reasoning tokens used. Reasoning length: {len(reasoning_content)} chars")

            # Debug: Check if content is empty
            if not content or content.strip() == "":
                error_msg = f"Empty response content. Finish reason: {response.choices[0].finish_reason}"
                if attempt < max_retries - 1:
                    print(f"\n  Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None, error_msg

            # Extract JSON from content (handles cases where reasoning is mixed in)
            json_str = extract_json_from_response(content)
            result = json.loads(json_str)

            # Validate result structure
            if not validate_result(result):
                error_msg = f"Invalid JSON structure - missing required fields. Got keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
                if attempt < max_retries - 1:
                    print(f"\n  Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None, error_msg

            return result, None

        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {str(e)}. Content preview: {content[:200] if 'content' in dir() and content else 'N/A'}..."
            if attempt < max_retries - 1:
                print(f"\n  Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                time.sleep(2 ** attempt)
                continue
            return None, error_msg

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if attempt < max_retries - 1:
                print(f"\n  Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None, error_msg

    return None, "Max retries exceeded"


def process_conference(conference_dir: Path, client, output_dir: Path, resume: bool = True, start_paper_id: str = None, end_paper_id: str = None):
    """Process all validation prompts for a conference."""

    conference_name = conference_dir.name
    prompt_files = sorted(conference_dir.glob("*_validation_prompt.txt"))

    # Filter by paper ID range if specified
    if start_paper_id or end_paper_id:
        filtered_files = []
        for prompt_file in prompt_files:
            paper_id = prompt_file.stem.replace('_validation_prompt', '')

            # Extract numeric part from paper_id (e.g., "2023.acl-main.100" -> "100")
            parts = paper_id.split('.')
            if len(parts) >= 3:
                try:
                    numeric_id = int(parts[-1])
                except ValueError:
                    # If last part is not numeric, try to extract numbers
                    match = re.search(r'(\d+)$', paper_id)
                    numeric_id = int(match.group(1)) if match else 0
            else:
                # Fallback: try to extract any trailing number
                match = re.search(r'(\d+)$', paper_id)
                numeric_id = int(match.group(1)) if match else 0

            # Convert start/end to integers for comparison
            try:
                start_id = int(start_paper_id) if start_paper_id else None
                end_id = int(end_paper_id) if end_paper_id else None

                # Check if numeric_id is within range
                if start_id is not None and numeric_id < start_id:
                    continue
                if end_id is not None and numeric_id > end_id:
                    continue

                filtered_files.append(prompt_file)
            except ValueError:
                # If conversion fails, use string comparison as fallback
                if start_paper_id and paper_id < start_paper_id:
                    continue
                if end_paper_id and paper_id > end_paper_id:
                    continue
                filtered_files.append(prompt_file)

        print(f"  Filtered to {len(filtered_files)} papers (from {start_paper_id or 'start'} to {end_paper_id or 'end'})")
        prompt_files = filtered_files

    if not prompt_files:
        return {'conference': conference_name, 'total': 0, 'completed': 0, 'errors': 0}

    # Create output directory
    conf_output_dir = output_dir / conference_name
    conf_output_dir.mkdir(exist_ok=True, parents=True)

    stats = {
        'conference': conference_name,
        'total': len(prompt_files),
        'completed': 0,
        'errors': 0,
        'skipped': 0
    }

    for prompt_file in tqdm(prompt_files, desc=f"{conference_name}", leave=False):
        paper_id = prompt_file.stem.replace('_validation_prompt', '')
        output_file = conf_output_dir / f"{paper_id}_validation.json"

        # Check if already processed and valid (for resuming)
        if resume and output_file.exists():
            try:
                # Check if existing file is valid
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)

                if validate_result(existing_result):
                    print(f"  [SKIPPED] {paper_id} (already processed)")
                    stats['skipped'] += 1
                    stats['completed'] += 1
                    continue
                else:
                    # Invalid/empty JSON, needs reprocessing
                    print(f"  [RE-PROCESSING] {paper_id} (invalid existing result)")
            except (json.JSONDecodeError, Exception):
                # Corrupted file, needs reprocessing
                print(f"  [RE-PROCESSING] {paper_id} (corrupted file)")
                pass

        # Print processing status
        print(f"  [PROCESSING] {paper_id}...")

        # Load prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()

        # Run extraction
        result, error = run_extraction(client, prompt_text)

        if error:
            stats['errors'] += 1
            # Save error log
            error_file = conf_output_dir / f"{paper_id}_error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Error: {error}\n")
            print(f"  [ERROR] {paper_id}: {error}")
            continue

        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  [SUCCESS] {paper_id} saved")
        stats['completed'] += 1

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    return stats


def main():
    """Main function to run validation extraction on all prompts."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run LaaJ-Human validation extraction on NLG papers')
    parser.add_argument('--conference', type=str, default=None,
                       help='Specific conference to process (e.g., ACL-2023). If not specified, processes all conferences.')
    parser.add_argument('--start-paper-id', type=str, default=None,
                       help='Start processing from this paper ID (inclusive). Useful for batch processing.')
    parser.add_argument('--end-paper-id', type=str, default=None,
                       help='Stop processing at this paper ID (inclusive). Useful for batch processing.')
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get('NOVITA_API_KEY')
    if not api_key:
        print("Error: NOVITA_API_KEY environment variable not set")
        print("Please set it with: export NOVITA_API_KEY='your-api-key'")
        return

    # Setup client
    client = setup_novita_client(api_key)

    # Paths
    prompts_dir = Path('../laaj_human_validation_prompts')
    output_dir = Path('../laaj_human_validation_results')
    output_dir.mkdir(exist_ok=True)

    # Get conference directories based on argument
    if args.conference:
        conference_dir = prompts_dir / args.conference
        if not conference_dir.exists() or not conference_dir.is_dir():
            print(f"Error: Conference directory '{args.conference}' not found in {prompts_dir}")
            print("\nAvailable conferences:")
            for d in sorted(prompts_dir.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}")
            return
        conference_dirs = [conference_dir]
    else:
        conference_dirs = sorted([d for d in prompts_dir.iterdir() if d.is_dir()])

    print("=" * 80)
    print("Running LaaJ-Human Validation Extraction with DeepSeek v3.1 via Novita AI")
    print("=" * 80)
    print(f"\nModel: deepseek/deepseek-v3.1-terminus")
    if args.conference:
        print(f"Processing conference: {args.conference}")
    else:
        print(f"Total conferences: {len(conference_dirs)}")
    if args.start_paper_id or args.end_paper_id:
        print(f"Paper ID range: {args.start_paper_id or 'start'} to {args.end_paper_id or 'end'}")
    print(f"Resume mode: Enabled (will skip already processed papers)\n")

    all_stats = []
    start_time = time.time()

    for conference_dir in conference_dirs:
        print(f"\nProcessing {conference_dir.name}...")
        stats = process_conference(
            conference_dir,
            client,
            output_dir,
            resume=True,
            start_paper_id=args.start_paper_id,
            end_paper_id=args.end_paper_id
        )
        all_stats.append(stats)

        if stats['total'] > 0:
            print(f"  Completed: {stats['completed']}/{stats['total']} (Skipped: {stats['skipped']}, Errors: {stats['errors']})")

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_papers = sum(s['total'] for s in all_stats)
    total_completed = sum(s['completed'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    total_skipped = sum(s['skipped'] for s in all_stats)

    print(f"\nTotal prompts: {total_papers}")
    print(f"Completed: {total_completed} ({100*total_completed/total_papers:.1f}%)")
    print(f"Skipped (already done): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    if total_completed - total_skipped > 0:
        print(f"Average time per paper: {elapsed_time/(total_completed-total_skipped):.1f} seconds")

    # Save summary
    summary_file = output_dir / 'validation_extraction_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_papers': total_papers,
            'completed': total_completed,
            'errors': total_errors,
            'skipped': total_skipped,
            'elapsed_time_minutes': elapsed_time / 60,
            'by_conference': all_stats
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
