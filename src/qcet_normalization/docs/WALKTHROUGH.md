# QCET Taxonomy Walkthrough: 20 Real Criteria from Our Corpus

This document maps 20 representative raw criterion strings from
`metadata_unique_counts/{llm,human}_criteria_stats.csv` to the QCET taxonomy
(+ our auxiliary categories), showing how the classifier will be expected to
reason. Counts are the number of papers that reported each raw string.

The point of this doc is to make the taxonomy concrete before we spend API
budget. If any mapping below feels wrong, the aux taxonomy or the disambiguation
rules need to change.

## Target label space (extended QCET lattice + meta)

We use a single unified target label space for the classifier:

- **111 QCET leaves** parsed from `qcet_taxonomy.json`, each placed in QCET's
  L1 (frame-of-reference) x L2 (type) x L3 (aspect) lattice.
- **6 AUX extension nodes** placed at specific lattice positions as siblings of
  existing QCET leaves, per the QCET authors' explicit extensibility invitation:
    - `AUX-Safety` is a sibling under `QEC-w` (External / Correctness / Whole)
    - `AUX-Toxicity` is a sibling under `QOC-c` (Own right / Correctness / Content)
    - `AUX-Bias` is a sibling under `QEC-c` (External / Correctness / Content)
    - `AUX-InstructionFollowing` is a sibling under `QIC-w` (Input / Correctness / Whole)
    - `AUX-Empathy` is a sibling under `QEG-w` (External / Goodness / Whole)
    - `AUX-Creativity` is a sibling under `QOF-w` (Own right / Feature / Whole)
- **3 meta-categories** (`AUX-OverallQuality`, `AUX-TaskSpecificPerformance`,
  `AUX-Other`) tracked as flat buckets; these deliberately refuse QCET's
  decomposition axes and are excluded from L1/L2/L3 aggregations.

Legend:
- `QXX-y-n` = a QCET leaf node (with its L1 frame-of-reference / L2 type / L3 aspect in the annotation)
- `AUX-Name` = one of our 9 auxiliary categories (6 tree-placed, 3 meta)
- Rationale spells out which test (QCET non-reducibility, aux disambiguation rule, aux tree placement) decided the mapping.

---

## Group A — Straightforward QCET matches (high-frequency head of the distribution)

These are raw strings where the classifier should have very high confidence.
They validate that QCET, as-is, handles the most common criteria in our corpus.

| # | Raw string (count) | Target | Rationale |
|---|---|---|---|
| 1 | `relevance` (187 LLM) | **QIG-c-3** *Relevance to Input* (L1=Input, L2=Goodness, L3=Content) | "Relevance" in LaaJ is always relative to the input (instruction, question, document). Exact semantic match with QCET's definition: "produces outputs that are in a given sense more relevant to the input." |
| 2 | `fluency` (88 LLM) | **QOG-w-3** *Fluency* (L1=Own right, L2=Goodness, L3=Whole) | Exact name match; QCET definition covers the standard sense. |
| 3 | `coherence` (93 LLM) | **QOG-c-3** *Coherence* (L1=Own right, L2=Goodness, L3=Content) | Exact name match. Note QCET has two sub-leaves (`3.1` Wellorderedness, `3.2` Cohesiveness) but we map to the parent unless the paper is specific. |
| 4 | `grammaticality` (human corpus) | **QOC-f-1** *Grammaticality* (L1=Own right, L2=Correctness, L3=Form) | Exact name match. |
| 5 | `helpfulness` (137 LLM) | **QEG-w-3** *Usefulness (nonspecific)* (L1=External, L2=Goodness, L3=Whole) | QCET's Usefulness leaf covers the HHH sense of "helpful": how useful the output is to the user. |
| 6 | `informativeness` (35 LLM) | **QOG-c-2** *Informativeness* (L1=Own right, L2=Goodness, L3=Content) | Exact name match. |
| 7 | `humor` / `humorous` (human) | **QOF-w-5** *Humorousness/Non-humorousness* (L1=Own right, L2=Feature, L3=Whole) | This is a nice proof that QCET covers more than expected — yes, humor is a QCET leaf. |
| 8 | `conciseness` (20 LLM) | **QOG-f-1** *Nonredundancy (form)* (L1=Own right, L2=Goodness, L3=Form) | "Concise" = less redundant in form. QCET uses the "nonredundancy" framing. (This one is worth calling out in the paper — our raw corpus uses "conciseness" far more than "nonredundancy"; the LLM classifier must recognize them as the same construct.) |
| 9 | `readability` (human) | **QOG-w-2** *Readability* (L1=Own right, L2=Goodness, L3=Whole) | Exact name match. QCET explicitly distinguishes from Fluency in its definition — we keep them separate. |
| 10 | `level of detail` (37 LLM) | **QOG-c-2** *Informativeness* (L1=Own right, L2=Goodness, L3=Content) | QCET's definition explicitly covers information density, which is what "level of detail" measures. Important example: the existing normalizer's key-noun extractor would produce "Detail" as a separate label, over-fragmenting. QCET brings it under Informativeness where it belongs. |

**Takeaway for Group A**: QCET handles the high-frequency head of our
distribution cleanly. The existing normalizer's biggest failure is *surface-word
shyness* — it treats "conciseness", "level of detail", and "informativeness" as
three different things. QCET puts them in two cohesive constructs.

---

## Group B — Ambiguous, context-dependent (this is where lean paper-context helps)

These raw strings do NOT have a single stable QCET mapping. The correct QCET
leaf depends on what the paper is actually doing.

| # | Raw string (count) | Candidate targets | Disambiguation |
|---|---|---|---|
| 11 | `correctness` (173 LLM, 83 Correctness) | Three options: `QOC-w-1` (generic, no frame-of-reference), `QEC-c-1` Factual Truth, `QEC-c-2` Relative Factual Accuracy, `QTC-w-1` Classification Accuracy | If paper is **fact-checking / knowledge QA / generation-vs-world**: `QEC-c-1` or `QEC-c-2`. If paper is **classification / labeling**: `QTC-w-1`. If paper uses "correctness" as a generic holistic-with-known-errors label: `QOC-w-1`. Classifier gets the co-criteria list and paper-level task context. |
| 12 | `accuracy` (111 LLM) | Same as #11 plus `QIC-w-1` Translation Accuracy | If task includes MT: favor `QIC-w-1`. Otherwise same routing as correctness. |
| 13 | `consistency` (56 LLM) | `QOG-c-4` Internal Consistency OR `QIC-c-3` Consistency with Input | If paper measures against a reference source document / input passage: `QIC-c-3`. If paper measures internal logical consistency of a single output: `QOG-c-4`. |
| 14 | `faithfulness` (43 LLM) | `QEC-c-2` Relative Factual Accuracy OR `QIC-c-3` Consistency with Input | Standard summarization usage (faithful to the source document): `QEC-c-2`. Generation-from-structured-data: `QIC-c-3`. Default: `QEC-c-2`. |
| 15 | `factuality` / `factual consistency` / `factual correctness` (25+22+32) | `QEC-c-1` Factual Truth (world-level) OR `QEC-c-2` Relative Factual Accuracy (source-level) | If paper checks against real-world truth without a specific source: `QEC-c-1`. If paper checks against a given knowledge source or retrieved document: `QEC-c-2`. Hallucination criteria land here too, per your earlier decision. |
| 16 | `alignment with instruction` (1) + many `Alignment with X` variants | Many QCET leaves possible | `alignment with instruction` / `task instruction` / `requirements` → `AUX-InstructionFollowing`. `alignment with reference answer` → `QTG-c-1` or `QTG-w-1` (similarity to target). `alignment with ground truth figure` → `QEC-c-2`. `alignment with human preferences` → `AUX-OverallQuality`. `text-image alignment` → `QIC-c-3` (image is the input, output text should be consistent). QCET-first rule kicks in here. |

**Takeaway for Group B**: These are the variants where the current
key-noun-extraction normalizer collapses everything into "Consistency" or
"Correctness" and destroys the signal. The classifier, with lean paper context
(task + co-occurring criteria), can route each occurrence to the correct leaf.
This is what makes the QCET exercise worth doing.

---

## Group C — Auxiliary category examples

| # | Raw string (count) | Target | Rationale |
|---|---|---|---|
| 17 | `safety` (60) / `harmlessness` (61, HHH sense in context) | **AUX-Safety** (at QEC-w) | Default mapping. If paper is a red-teaming / jailbreak benchmark: confirmed AUX-Safety. Placed in QCET's External/Correctness/Whole subtree as a sibling of `QEC-w-1 Functional Correctness`. |
| 18 | `toxicity` (14 LLM, 23 human) / `offensiveness` (9) | **AUX-Toxicity** (at QOC-c) | Content-toxicity construct. Placed in Own-right/Correctness/Content as a sibling of `QOC-c-1 Semantic Correctness`. Disambiguates from AUX-Safety because the evaluation measures properties of generated text content, not refusal behavior. |
| 19 | `Presence of stereotype or counter-stereotype text spans` (1) | **AUX-Bias** (at QEC-c) | Canonical fairness probe (stereotype datasets). Placed in External/Correctness/Content as a sibling of `QEC-c-1 Factual Truth` and `QEC-c-2 Relative Factual Accuracy`. |
| 20 | `empathy` (common in dialogue/mental-health) | **AUX-Empathy** (at QEG-w) | System's emotional stance toward user; distinct from QEF-w-3 Effect on User Emotion (which is about user's resulting emotion change). Placed in External/Goodness/Whole as a sibling of `QEG-w-3 Usefulness`. |

---

## Edge cases (what happens to junk)

| Raw string | Target | Sub-reason |
|---|---|---|
| `text` | AUX-Other | `not_a_criterion` |
| `BLEU` (if found in criteria field) | AUX-Other | `metric_name_in_criterion_field` (also recorded in audit CSV) |
| `position bias` (LaaJ judge bias) | AUX-Other | `judgement_bias` — property of the judge, not the output |
| `Harmonicity` (1) | AUX-Other (probably) | `unclassifiable` without more paper context — could be speech quality (QOG-f-2) if sound evaluation, but the bare word is ambiguous |

---

## What this changes vs. current normalization

The current key-noun extractor collapses:
- `relevance`, `relevant`, `irrelevant`, `Relevance to Input`, `Input Relevance`, `answer relevance`, and 100+ more variants → all become "Relevance" (fine).
- BUT: `correctness`, `factual correctness`, `grammatical correctness`, `semantic correctness`, `answer correctness`, `task correctness`, `logical correctness`, and 170 variants → all become "Correctness" (this over-merges semantically distinct QCET leaves: `QOC-w-1`, `QOC-f-1` / `QEC-f-2`, `QEC-c-1`/`QEC-c-2`, `QOC-c-1`, which are different dimensions).

The QCET-based pipeline restores this distinction automatically by using the
raw string's semantics plus per-paper context, not just the head noun.

---

## What to look for when reviewing

- **Do the Group A mappings feel right?** If not, a QCET leaf is being applied too generously.
- **Do the Group B disambiguations feel useful?** The value of the QCET exercise is precisely that it forces this disambiguation instead of collapsing.
- **Are there any aux categories where a raw string I'd expect to see is missing from the "default labels"?** Add them.
- **Are there any raw strings I should have expected to see in Group A/B/C but didn't?** Tell me and I'll add the example.
