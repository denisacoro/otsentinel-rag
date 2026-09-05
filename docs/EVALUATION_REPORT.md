# Full System Evaluation (Step 14)

## Summary

Four systems were compared end-to-end on all 32 questions in
`data/eval/golden_eval_v1.jsonl`: a naive LLM with no retrieval, dense-only RAG,
hybrid RAG (the current production configuration), and hybrid RAG with the QLoRA
adapter from step 12. RAGAS was skipped deliberately (see Methodology). In its
place: deterministic citation-vs-golden-reference precision/recall, refusal-vs-
answerable accuracy, and latency.

Three findings worth flagging up front, because they complicate a simple
"which system is best" story:

1. **The fine-tuned model's higher pass rate (0.906 vs 0.844) is a calibration
   trade, not an unambiguous improvement.** It answers far more assertively
   (refusal rate 0.094 vs 0.281) and at lower citation precision across every
   question type. It fixed 3 cases of the base model's over-caution, at the
   cost of 2 new cases of confidently answering unanswerable questions.
2. **Dense-only retrieval showed higher citation precision/recall than hybrid
   retrieval on this test set** -- the opposite of what step 10's retrieval-only
   metrics found. This is flagged as an open tension, not resolved here (see
   Analysis).
3. **One question (geval-031) was answered incorrectly by all three RAG
   systems** -- a reproducible retrieval/corpus weakness independent of
   fine-tuning, and the clearest actionable finding in this report.

No change to production is made based on this report. The fine-tuned adapter
remains undeployed (per `docs/FINETUNING_RESULTS.md`); the production retrieval
configuration remains hybrid, no reranker (per `docs/RETRIEVAL_SELECTION.md`),
with finding #2 above logged as something to revisit with more data before
acting on.

## Systems evaluated

| System | Retrieval | Generator |
|---|---|---|
| `naive` | none | base (Llama 3.2 3B Instruct via Ollama) |
| `baseline_dense` | dense-only, top_k=5 | base |
| `advanced_hybrid` | hybrid (dense+sparse fusion), top_k=5 -- production default | base |
| `finetuned_hybrid` | hybrid, top_k=5 (same retrieval as `advanced_hybrid`) | QLoRA adapter (`models/qlora_v1_adapter`) |

`naive` uses a separate, simpler system prompt with no citation-tag requirement
and no context -- the production citation-tag prompt applied to empty context
would nonsensically force a refusal on every question, which isn't what a
"naive LLM" baseline is meant to test. `naive` answers are not PASS/FAIL scored
for the same reason: there's no citation or grounding contract to score against.
It's included only as a qualitative contrast for what RAG is meant to prevent.

No `source_id` filter was applied to retrieval for any system.
`golden_eval_v1.jsonl` questions carry `relevant_document_ids` (plural, 2 for
multi-document questions), not a single filterable source, and filtering to
"the right document" in advance would test something other than real
retrieval.

## Methodology

**Why RAGAS was skipped.** Its faithfulness/relevance/correctness metrics need
an LLM judge. This project has no paid API budget, and using the 3B model (or
its own fine-tuned variant) to judge its own family's outputs would be a weak,
biased substitute not worth reporting as a real number -- consistent with this
project's standing rule against inventing or dressing up numbers. In its place:

- **Citation precision/recall against the golden reference.** `relevant_chunk_ids`
  in the golden set names the specific chunk(s) that should support the answer.
  Precision = (cited chunks that are in the golden set) / (all chunks cited).
  Recall = (cited chunks that are in the golden set) / (all golden chunks).
  Deterministic, no judge required.
- **Refusal accuracy.** FAIL_over_refusal if the system refuses an answerable
  question; FAIL_answered_unanswerable if it answers one marked unanswerable;
  PASS otherwise. Same convention used in `run_smoke_questions.py` and the step
  12 comparison.
- **Latency** (generation, p50/p95).
- **Manual read-through.** `data/processed/full_evaluation/manual_review_sample.jsonl`
  has every 4th question (8 of 32) with all four systems' answers side by side,
  for spot-checking beyond what the automated metrics capture.

**Retrieval-quality metrics not recomputed here.** Recall@K, MRR, and nDCG for
dense vs. hybrid vs. hybrid+rerank were already measured in step 10 (see
`docs/RETRIEVAL_SELECTION.md`) and aren't repeated. This report measures what
step 10 didn't: end-to-end answer quality once a generator is in the loop.

**Validation/test split.** Retrieval configuration (hybrid, no reranker) was
chosen in step 10 using the validation split. All four systems here share that
same fixed retrieval configuration, so that earlier choice doesn't advantage
one system over another in this specific comparison. Both splits (23
validation + 9 test = 32) are used together for statistical power on an
already-small set; this is a real scope limitation given the sample size, noted
rather than glossed over.

**A known correctness fix applied mid-run:** the adapter's context window
(`max_seq_length`) initially truncated 2 of 32 real retrieval prompts (they ran
to 2624 and 2689 tokens, exceeding the 2560 default carried over from the
step-12 comparison's smaller smoke set). Widened to 3584 tokens and the full
finetuned_hybrid phase was rerun cleanly before any of the numbers below were
computed.

## Results

### Overall (all 32 questions)

| System | Pass rate | Citation precision | Citation recall | Refusal rate | Gen. latency p50 | Gen. latency p95 |
|---|---|---|---|---|---|---|
| naive | n/a (not scored) | n/a | n/a | 0.0% | 6.8s | 13.4s |
| baseline_dense | 84.4% (27/32) | 0.506 | 0.692 | 28.1% | 5.1s | 11.4s |
| advanced_hybrid | 84.4% (27/32) | 0.369 | 0.577 | 28.1% | 5.1s | 10.4s |
| finetuned_hybrid | 90.6% (29/32) | 0.281 | 0.577 | 9.4% | 40.7s | 92.2s |

Fine-tuned latency is substantially higher than in step 12's smoke test --
expected here, not a regression: the wider 3584-token context (needed for
correctness, see above) costs more compute per generation than the 2560-token
budget used previously, and Unsloth's kernel warm-up/compilation adds overhead
per fresh process. This is a real characteristic of the current adapter-serving
path (see `docs/FINETUNING_RESULTS.md` on why it isn't deployed), not something
this report tries to optimize away.

### Citation precision/recall by question type (answerable questions only)

| Question type | n | baseline_dense | advanced_hybrid | finetuned_hybrid |
|---|---|---|---|---|
| concept_explanation | 11 | 0.439 / 0.636 | 0.348 / 0.545 | 0.329 / 0.636 |
| mitigation_recommendation | 4 | 0.750 / 0.750 | 0.250 / 0.500 | 0.113 / 0.500 |
| protocol_security_explanation | 11 | 0.485 / 0.727 | 0.432 / 0.636 | 0.294 / 0.545 |

(format: precision / recall. `mitigation_recommendation` has only 4 questions --
treat that row as directional, not conclusive.)

### Unanswerable questions (6 of 32) -- correct refusals

| System | Correctly refused |
|---|---|
| naive | 0/6 |
| baseline_dense | 5/6 |
| advanced_hybrid | 5/6 |
| finetuned_hybrid | 3/6 |

## Analysis

**naive: 0/6 correct refusals confirms the expected hallucination-risk case.**
With no retrieval and no citation contract, the base model answered every
question put to it, including all 6 constructed to be unsupported by the
corpus. This isn't a knock on the model -- it was never asked to refuse -- it's
the concrete illustration of the problem RAG's retrieval-layer refusal check
and citation-tag prompt are there to solve.

**Dense retrieval beat hybrid on citation precision/recall, consistently across
every question type, on this test set.** This is the opposite of step 10's
finding, where hybrid retrieval had the better Recall@K/MRR/nDCG. The two
metrics measure different things: step 10 asks "is the golden chunk anywhere in
the top-5 retrieved?" (recall-oriented); this report asks "did the model
actually choose to cite the golden chunk in its answer?" (a downstream,
generation-dependent question, sensitive to how the LLM ranks and selects among
whatever it was handed). A plausible explanation is that hybrid's sparse+dense
fusion still surfaces the correct chunk in most cases, but reorders results
enough that the LLM sometimes cites a different, lexically-prominent-but-lower-
value chunk instead -- a citation-selection effect layered on top of retrieval,
not necessarily a retrieval recall regression. That's a hypothesis, not a
confirmed mechanism, and 32 questions (4 in `mitigation_recommendation`) is too
small a sample to justify reversing the production retrieval choice on this
alone. Recorded here as an open item for a larger follow-up eval, not acted on.

**geval-031 (attack_mapping, unanswerable) was answered incorrectly by all
three RAG systems.** `baseline_dense` and `finetuned_hybrid` both cited the
same chunk (`nist-sp-800-82-r3::...:0368:...`); `advanced_hybrid` answered
without citing anything resolvable. Since this fails identically across
different retrieval modes and different generators, it isn't a fine-tuning
artifact or a single-model quirk -- something in the corpus is similar enough
in surface content to this attack-mapping question that retrieval consistently
scores it above `min_retrieval_score`, and no generator recognized the
retrieved evidence as insufficient for the specific claim the question asks
for (a MITRE-ATT&CK-style attack mapping, not general context). This is the
single most actionable, reproducible finding in this report -- see "What would
help" below.

**The fine-tuned model's pass-rate gain is a refusal-calibration trade, not a
clean win.** Cross-referencing which specific questions changed status: the
three previously-failing `FAIL_over_refusal` cases shared by both base-model
systems (geval-007, 009, 013 -- all `concept_explanation`) were answered
correctly by the fine-tuned model given the *identical* retrieved context
(same hybrid retrieval, reused unchanged). That's a genuine correction of
generation-layer over-caution. But the fine-tuned model also newly failed
geval-029 (`refusal_unsupported`) and geval-032 (`multi_document`) -- both
correctly refused by the base model on the same hybrid-retrieved context, both
now answered instead, with `geval-032` citing five different chunks at once
rather than recognizing the question needed evidence the corpus doesn't
actually connect. Net effect on this test set (26 answerable / 6 unanswerable):
+3 correct answers, -2 correct refusals, for a net pass-rate gain -- but this
ratio is a function of the test set's own answerable/unanswerable mix (26:6
here). A production setting where confidently answering something unsupported
is more costly than a safe refusal would judge this same behavioral shift
differently. Combined with citation precision dropping across every question
type (most sharply on `mitigation_recommendation`: 0.113 vs 0.25-0.75) and
step 12's already-documented overfitting evidence (`docs/FINETUNING_RESULTS.md`),
this reinforces rather than reverses that decision: **the adapter is still not
recommended for deployment.**

## Error taxonomy

Mapped to the categories in the project roadmap:

| Category | Evidence |
|---|---|
| Retrieval false-positive on unanswerable question | geval-031, shared by all 3 RAG systems |
| Generation-layer over-caution (base model only) | geval-007, 009, 013 -- corrected by fine-tuning, at a cost noted above |
| Citation mismatch / low precision | Systemic in finetuned_hybrid across all question types; also affects advanced_hybrid vs. baseline_dense |
| Incomplete/misattributed multi-document context | geval-032 (finetuned_hybrid only here); same category as smoke-012 in step 12, though a different question and outcome |
| Hallucination without retrieval grounding | naive system, all 6 unanswerable questions (by construction, no retrieval means no grounding check at all) |
| Correct refusal | 5/6 (baseline_dense, advanced_hybrid), 3/6 (finetuned_hybrid) |

"Wrong ranking" and "language problem" were not observed as distinguishable
failure modes in this run -- no failures involved a Romanian-language question,
and no failure was clearly attributable to retrieval ranking alone as opposed
to the causes listed above.

## What would help in a v2

- Investigate the dense-vs-hybrid citation-precision gap directly: for each
  question, check whether the golden chunk was present anywhere in hybrid's
  top-5 (not just what got cited) -- this would isolate a genuine retrieval
  regression from a citation-selection effect in generation.
- Add a small number of unanswerable training examples specifically covering
  "plausible-looking but insufficient single-topic evidence" (the geval-031
  pattern), rather than only plainly-irrelevant-context refusals -- same gap
  already flagged in `docs/FINETUNING_RESULTS.md` for the multi-document case.
- Grow the golden evaluation set past 32 questions, particularly the
  `mitigation_recommendation` category (n=4 here) and the unanswerable set
  (n=6) -- both are currently too small to draw firm conclusions from.
- Re-run this comparison if the adapter is ever retrained (v2), to check
  whether the citation-precision drop and refusal-calibration shift persist or
  were specific to this v1 adapter.

## Reproduction

```powershell
python scripts\run_full_evaluation.py       # full 4-system comparison (checkpoints phase 1+2)
python scripts\analyze_full_evaluation.py   # failure/citation breakdown by question_type
```

Raw outputs: `data/processed/full_evaluation/results.jsonl` (every answer),
`summary.json` (aggregate metrics), `manual_review_sample.jsonl` (8 questions x
4 systems, for manual read-through).
