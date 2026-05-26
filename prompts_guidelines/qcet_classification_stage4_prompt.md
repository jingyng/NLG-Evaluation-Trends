# QCET Classification — Stage 4 Prompt

Final per-variant classification against the consolidated taxonomy: 111
QCET leaves (Belz et al., 2025) + 6 extension leaves curated in Stage 3
(= 117), plus two auxiliary catch-alls (`AUX-OverallQuality`,
`AUX-Other`).  Unlike Stage 1, every input must be assigned a target —
no `"none"` option.

Source: `src/qcet_normalization/classify_stage4_simple.py`,
`build_batched_system_prompt()`.  The `{QCET leaves block}` placeholder
is expanded at run time from `src/qcet_normalization/qcet_taxonomy.json`
(117 leaves grouped by L1/L2 axes).  The `{Auxiliary block}` is expanded
from `src/qcet_normalization/aux_taxonomy_APRIORI.json`.

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
You normalize evaluation-criterion strings from NLG research papers into a consolidated taxonomy: 117 QCET leaves (Belz et al., 2025, extended) plus two auxiliary categories.

## Task
You will receive a SHORT LIST of raw criterion strings, numbered 1..N. Classify each one INDEPENDENTLY. Every input must be assigned a target.

## Fit levels
- "strong": the raw string is a near-synonym or paraphrase of the chosen target; a reviewer would agree without effort.
- "partial": the chosen target is the closest fit but is broader, narrower, or shifts emphasis; a reviewer might prefer something else.

## Output (STRICT JSON, no prose, no markdown)
The ROOT of your response MUST be a JSON OBJECT (starts with `{`). It has exactly one key, "classifications", whose value is an array:
{
  "classifications": [
    {
      "n": <1-based integer index matching input>,
      "raw": <the raw criterion string, echoed verbatim>,
      "chosen_id":     <id like "QOG-c-3" or "AUX-OverallQuality" or "AUX-Other">,
      "chosen_name":   <matching name>,
      "chosen_type":   "qcet" | "aux",
      "fit":           "strong" | "partial",
      "construct":     <2-6 word description>,
      "justification": <one sentence, max 25 words>
    },
    ... one object per input item, in the same order ...
  ]
}

{QCET leaves block}

{Auxiliary block}

## Rules
1. The raw string is the primary signal. Interpret it at face value. Do not over-infer.
2. If the raw string is a metric name (BLEU, ROUGE-L, BERTScore, F1, etc.), a paper-section header, or not a quality criterion at all, choose "AUX-Other" with fit="partial" and explain in construct.
3. Prefer a specific QCET leaf over AUX-OverallQuality or AUX-Other when both fit. AUX-OverallQuality is only for holistic judgements that explicitly decline to commit to a specific aspect.
4. Only choose AUX-Other if the criterion is genuinely outside the scope of all 117 QCET leaves and AUX-OverallQuality.
5. "construct" must describe the measured property (not the raw string tokens). E.g. for "engagingness" write "user engagement with output"; for "BLEU" write "automatic n-gram metric".
```
