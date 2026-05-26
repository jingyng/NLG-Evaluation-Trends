# Metadata Normalization Prompt

You are a helpful assistant that normalizes and deduplicates metadata extracted from NLG evaluation papers. Your task is to identify and merge items that refer to the same concept.

## Task

Given a list of items from a specific field (e.g., metrics, tasks, models, datasets, criteria), identify which items are semantically equivalent and should be merged, then provide a normalized, deduplicated list.

## Input

**Field Type:** {field_type}
**Original List:** {original_list}

## Instructions

1. Identify items that refer to the same concept (e.g., "BLEU", "bleu", "Bleu" are all the same metric)
2. Choose the most standard/canonical form for each concept
3. Create a deduplicated list with canonical forms
4. Document which original items were merged together

## Output Format

Please return ONLY a JSON object with the following structure:

```json
{{
  "normalized_list": ["canonical_item1", "canonical_item2", ...],
  "merges_made": [
    {{
      "original": ["variant1", "variant2", "variant3"],
      "normalized": "canonical_form",
      "reason": "Brief explanation of why these are the same"
    }},
    ...
  ],
  "kept_separate": [
    {{
      "items": ["item_a", "item_b"],
      "reason": "Brief explanation of why these are kept separate despite similarity"
    }},
    ...
  ]
}}
```

## Guidelines

- **Be accurate**: Only merge items that truly refer to the same thing
- **Use standard forms**: Prefer widely-recognized naming conventions
- **Preserve distinctions**: Keep different versions, variants, or subtypes separate if they are meaningfully different
- **Document reasoning**: Explain merge decisions, especially for non-obvious cases
- **When in doubt, keep separate**: It's better to have duplicates than to incorrectly merge different concepts

## Examples

### Example 1: Automatic Metrics
**Input:**
```
Field Type: automatic_metrics
Original List: ["BLEU", "bleu", "ROUGE-1", "rouge-1", "ROUGE-2", "BERTScore", "bert score", "Accuracy", "acc", "F1", "f1-score"]
```

**Output:**
```json
{{
  "normalized_list": ["BLEU", "ROUGE-1", "ROUGE-2", "BERTScore", "Accuracy", "F1"],
  "merges_made": [
    {{"original": ["BLEU", "bleu"], "normalized": "BLEU", "reason": "Case variations of the same metric"}},
    {{"original": ["ROUGE-1", "rouge-1"], "normalized": "ROUGE-1", "reason": "Case variations"}},
    {{"original": ["BERTScore", "bert score"], "normalized": "BERTScore", "reason": "Spacing/case variations"}},
    {{"original": ["Accuracy", "acc"], "normalized": "Accuracy", "reason": "Full form and common abbreviation"}},
    {{"original": ["F1", "f1-score"], "normalized": "F1", "reason": "Same metric with/without suffix"}}
  ],
  "kept_separate": [
    {{"items": ["ROUGE-1", "ROUGE-2"], "reason": "Different n-gram variants of ROUGE metric"}}
  ]
}}
```

### Example 2: Tasks
**Input:**
```
Field Type: tasks
Original List: ["Machine Translation", "machine translation", "MT", "Summarization", "Text Summarization", "summary generation"]
```

**Output:**
```json
{{
  "normalized_list": ["Machine Translation", "Text Summarization"],
  "merges_made": [
    {{"original": ["Machine Translation", "machine translation", "MT"], "normalized": "Machine Translation", "reason": "Case variations and common abbreviation"}},
    {{"original": ["Summarization", "Text Summarization", "summary generation"], "normalized": "Text Summarization", "reason": "Different phrasings of the same task"}}
  ],
  "kept_separate": []
}}
```

### Example 3: Models
**Input:**
```
Field Type: models
Original List: ["GPT-3", "gpt-3", "GPT-4", "BERT-base", "BERT-large", "BERT"]
```

**Output:**
```json
{{
  "normalized_list": ["GPT-3", "GPT-4", "BERT-base", "BERT-large", "BERT"],
  "merges_made": [
    {{"original": ["GPT-3", "gpt-3"], "normalized": "GPT-3", "reason": "Case variation"}}
  ],
  "kept_separate": [
    {{"items": ["GPT-3", "GPT-4"], "reason": "Different versions"}},
    {{"items": ["BERT-base", "BERT-large", "BERT"], "reason": "Different model sizes/versions, BERT without version kept as it may refer to generic BERT usage"}}
  ]
}}
```
