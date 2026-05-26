# QCET-validation Likert scorecard

- Sample size:            155
- Annotated rows scored:  155
- Unannotated (skipped):  0

## Overall

Mean score: **4.45** / 5.0  (SD 1.10)

| score | label | count | % |
|---|---|---|---|
| 5 | Perfect | 114 | 73.5% |
| 4 | Good | 19 | 12.3% |
| 3 | Acceptable | 8 | 5.2% |
| 2 | Poor | 6 | 3.9% |
| 1 | Wrong | 8 | 5.2% |

| threshold | value |
|---|---|
| Score ≥ 4 (Good or Perfect)       | 85.8%  (133/155) |
| Score ≥ 3 (Acceptable or better)  | 91.0%  (141/155) |
| Score ≤ 2 (Poor or Wrong)         | 9.0%  (14/155) |
| Score = 1 (Wrong)                 | 5.2%  (8/155) |

## Per-stratum breakdown

| stratum | n | mean | ≥4 | ≥3 | ≤2 |
|---|---|---|---|---|---|
| `A_qcet_strong_agree` | 25 | 4.44 | 84.0%  (21/25) | 92.0%  (23/25) | 8.0%  (2/25) |
| `B_qcet_strong_rescued` | 20 | 4.70 | 90.0%  (18/20) | 100.0%  (20/20) | 0.0%  (0/20) |
| `C_qcet_partial` | 25 | 3.96 | 76.0%  (19/25) | 76.0%  (19/25) | 24.0%  (6/25) |
| `D_qcet_disagreement` | 20 | 4.40 | 90.0%  (18/20) | 90.0%  (18/20) | 10.0%  (2/20) |
| `E_aux_specific` | 10 | 5.00 | 100.0%  (10/10) | 100.0%  (10/10) | 0.0%  (0/10) |
| `F_aux_other` | 25 | 4.36 | 80.0%  (20/25) | 92.0%  (23/25) | 8.0%  (2/25) |
| `G_stage3_decisions` | 10 | 4.10 | 80.0%  (8/10) | 80.0%  (8/10) | 20.0%  (2/10) |
| `H_new_qcet_nodes` | 20 | 4.90 | 95.0%  (19/20) | 100.0%  (20/20) | 0.0%  (0/20) |

## Per-source breakdown

| chosen_source | n | mean | ≥4 | ≥3 | ≤2 |
|---|---|---|---|---|---|
| `polysemous_override(was:QEC-c-1)` | 1 | 5.00 | 100.0%  (1/1) | 100.0%  (1/1) | 0.0%  (0/1) |
| `polysemous_override(was:QEF-f-1)` | 1 | 5.00 | 100.0%  (1/1) | 100.0%  (1/1) | 0.0%  (0/1) |
| `polysemous_override(was:QIC-c-3)` | 2 | 5.00 | 100.0%  (2/2) | 100.0%  (2/2) | 0.0%  (0/2) |
| `polysemous_override(was:QIG-c-3)` | 1 | 5.00 | 100.0%  (1/1) | 100.0%  (1/1) | 0.0%  (0/1) |
| `polysemous_override(was:QOC-c-1)` | 2 | 5.00 | 100.0%  (2/2) | 100.0%  (2/2) | 0.0%  (0/2) |
| `polysemous_override(was:QOC-w-1)` | 1 | 5.00 | 100.0%  (1/1) | 100.0%  (1/1) | 0.0%  (0/1) |
| `stage1_exact` | 19 | 4.58 | 89.5%  (17/19) | 94.7%  (18/19) | 5.3%  (1/19) |
| `stage3_drop` | 1 | 1.00 | 0.0%  (0/1) | 0.0%  (0/1) | 100.0%  (1/1) |
| `stage3_split_keep_stage1` | 9 | 4.44 | 88.9%  (8/9) | 88.9%  (8/9) | 11.1%  (1/9) |
| `stage4_llm` | 118 | 4.42 | 84.7%  (100/118) | 90.7%  (107/118) | 9.3%  (11/118) |

## Low-scoring rows (score ≤ 2, 14 rows)

| score | stratum | raw_string | predicted_id | predicted_name | notes |
|---|---|---|---|---|---|
| 1 | `A_qcet_strong_agree` | `OutE` | QIC-c-2 | Absence of Additions (relative | Not possible to determine, could be AUX-other |
| 1 | `C_qcet_partial` | `error span identification correctness` | QTC-w-1 | Form Accuracy | Should be Sequence Labelling Accuracy |
| 1 | `C_qcet_partial` | `legal soundness` | QOG-c-4 | Internal Consistency of Output | Relative Factual Accuracy |
| 1 | `C_qcet_partial` | `Manner` | QOG-w-5.1 | Clarity | Appropriateness (form) |
| 1 | `C_qcet_partial` | `Image Groundedness` | QIG-c-3 | Relevance to Input | Relative Factual Accuracy |
| 1 | `F_aux_other` | `Error spans` | AUX-Other | Other / Unclassifiable |  |
| 1 | `G_stage3_decisions` | `Relevant C` | AUX-Other | Other / Unclassifiable | Should be related to relevance |
| 1 | `G_stage3_decisions` | `Inaccurate Predictions` | QTC-w-3 | Complete Target Output Matchin | Correctness of output |
| 2 | `A_qcet_strong_agree` | `validity` | QEC-c-1 | Factual Truth | Recommend mapping to Overall Quality |
| 2 | `C_qcet_partial` | `Chain-of-Thought reasoning presence` | QOC-c-1 | Semantic Correctness | Quality as Explanation of Input |
| 2 | `C_qcet_partial` | `Goal congruence` | QIG-c-3 | Relevance to Input | Could be User Satisfaction as Affected by Outputs, but hard to determine |
| 2 | `D_qcet_disagreement` | `math constraint violations` | QOC-c-1 | Semantic Correctness |  |
| 2 | `D_qcet_disagreement` | `Correct number, incorrect reasoning` | QOC-c-1 | Semantic Correctness | Should be Output corretness |
| 2 | `F_aux_other` | `Interactiveness` | AUX-Other | Other / Unclassifiable |  |

