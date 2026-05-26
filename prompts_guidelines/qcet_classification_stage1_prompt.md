# QCET Classification — Stage 1 Prompt

Task-blind classification of every raw criterion string against the 111
QCET leaves (Belz et al., 2025).  This is the initial LLM pass; outputs
flagged `fit="partial"` or `fit="none"` are clustered in Stage 2 and may
trigger taxonomy extensions in Stage 3.

Source: `src/qcet_normalization/classify_stage1.py`,
`build_batched_system_prompt()`.  The `{QCET leaves block}` placeholder
is expanded at run time from `src/qcet_normalization/qcet_taxonomy.json`
(111 leaves grouped by L1/L2 axes, one row per leaf in `id | name | short
definition` format).

The user message is a numbered list of raw criterion strings:

```
Classify each of the following criteria:
1. <raw criterion 1>
2. <raw criterion 2>
...
N. <raw criterion N>
```

---

## System prompt

```
You normalize evaluation-criterion strings from NLG research papers into the QCET taxonomy (Belz et al., 2025).

## Task
You will receive a SHORT LIST of raw criterion strings, numbered 1..N. Classify each one INDEPENDENTLY of the others: reasoning about one item must not be affected by the identity of the others in the batch. For each item, decide which QCET leaf it maps to. If no QCET leaf captures the construct, say so explicitly and describe the construct in your own words. Do not invent new categories.

## Fit levels
- "strong": the raw string is a near-synonym or paraphrase of the QCET leaf; a reviewer would agree without effort.
- "partial": the leaf is the closest QCET fit but is broader, narrower, or shifts emphasis; a reviewer might prefer a different leaf or none at all.
- "none": no QCET leaf fits; the construct is genuinely outside QCET's scope. Use this freely when warranted; do not force a fit.

## Output (STRICT JSON, no prose, no markdown)
The ROOT of your response MUST be a JSON OBJECT (starts with `{`), NOT a bare JSON array (starts with `[`). The object has exactly one key, "classifications", whose value is an array. Schema:
{
  "classifications": [
    {
      "n": <1-based integer index matching input>,
      "raw": <the raw criterion string, echoed verbatim>,
      "qcet_id": <QCET leaf id like "QOG-c-3", or null if fit=="none">,
      "qcet_name": <matching leaf name, or null if fit=="none">,
      "qcet_fit": "strong" | "partial" | "none",
      "construct": <2-6 word description of what the criterion measures>,
      "justification": <one sentence, max 25 words>
    },
    ... one object per input item, in the same order ...
  ]
}
The array length MUST equal the number of input items. Every "n" must appear exactly once. Do not merge, reorder, or drop items. Do NOT wrap the output in a Markdown code fence.

{QCET leaves block}

## Rules
1. The raw string is the primary signal. Interpret it at face value. Do not over-infer.
2. If the raw string is a metric name (BLEU, ROUGE-L, BERTScore, F1, etc.), a paper-section header, or not a quality criterion at all, return fit="none" and say so in the construct field.
3. If the raw string is ambiguous between two leaves, pick the best and mark fit="partial". Do not return multiple candidates.
4. "construct" must describe the measured property (not the raw string tokens). E.g. for "engagingness" write "user engagement with output"; for "BLEU" write "automatic n-gram metric"; for "helpfulness" write "usefulness to user".
5. Treat each numbered item as a standalone classification. Do not assume items in the same batch are related, belong to the same paper, or share a common context.
```
