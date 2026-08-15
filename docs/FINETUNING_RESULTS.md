# QLoRA Fine-tuning Results v1 (Step 12)

## Summary

QLoRA fine-tuned `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` on the 54-example
`train_v1.jsonl` dataset (44 train / 10 validation). Compared against the baseline
(unmodified) model on 12 real smoke-test questions, using the live retrieval
pipeline for context. **Result: no measurable improvement over the baseline, and
evidence of overfitting to a specific training example.** The fine-tuned adapter is
not deployed to the production pipeline; Ollama continues serving the baseline model.

This is reported as a finding, not a failure. Understanding *why* a fine-tune didn't
help is a legitimate and useful outcome, and the negative result is directly
explainable by dataset scale and composition (see Analysis below).

## Training setup

| Parameter | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit NF4) |
| Method | QLoRA via Unsloth + PEFT + TRL SFTTrainer |
| LoRA rank / alpha | 16 / 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 24,313,856 / 3,237,063,680 (0.75%) |
| Train / validation examples | 44 / 10 (from `data/finetuning/train_v1.jsonl`) |
| Epochs | 4 (44 optimizer steps, effective batch size 4) |
| Learning rate | 2e-4, cosine schedule, 10% warmup |
| max_length | 768 tokens (covers all 54 examples; max observed 711) |
| Optimizer | adamw_8bit |
| Seed | 3407 |
| Hardware | NVIDIA RTX 3050 Ti Laptop GPU, 4GB VRAM |
| Training time | 4 min 42 sec |
| Peak VRAM (training) | 3.23 GB |

## Training metrics

| Epoch | train_loss (last step) | eval_loss |
|---|---|---|
| 1.0 | 1.99 | 1.859 |
| 2.0 | 0.95 | 0.898 |
| 3.0 | 0.73 | 0.803 |
| 4.0 | 0.76 | 0.791 |

`eval_loss` improved monotonically and flattened out -- no divergence signal in the
aggregate loss curve. Most of the improvement happened in epochs 1-2; epochs 3-4
produced diminishing returns (0.898 -> 0.791). Full run logged to MLflow, experiment
`otsentinel-finetuning`.

Note: the aggregate loss curve looking healthy did **not** rule out overfitting to
individual examples -- see the smoke-005 finding below, which loss curves alone
would not have caught.

## Evaluation methodology

`scripts/compare_qlora_finetuning.py` runs the exact retrieval path the live
pipeline uses (`app.retrieval` hybrid search, `app.generation.prompts.build_messages`,
unchanged) for all 12 questions in `data/eval/smoke_questions.jsonl`, then generates
an answer with the baseline model and the fine-tuned adapter using identical
messages, loaded sequentially (4GB VRAM does not fit two 3B models at once).
`max_length=2560` for evaluation, since real retrieval (`top_k=5`) produces prompts
of ~1,300-2,400 tokens -- substantially longer than any single training example,
which only ever used 1-2 chunks.

Scoring matches `scripts/run_smoke_questions.py`: FAIL if the model refuses an
answerable question or answers an unanswerable one; REVIEW if it answers without a
resolvable citation tag; otherwise PASS.

## Results

| | PASS | FAIL | REVIEW |
|---|---|---|---|
| Baseline | 11 | 1 | 0 |
| Fine-tuned | 10 | 1 | 1 |

One regression (smoke-005: PASS -> REVIEW), no improvements, one shared failure
(smoke-012) unchanged by fine-tuning. Full per-question answers in
`data/processed/qlora_comparison/results.jsonl`.

## Analysis

**smoke-005 ("How should MQTT authentication be configured?") -- overfitting
evidence.** The fine-tuned model's answer is a near-verbatim reproduction of the
source chunk behind training example `ft-048` (MQTT section 5.4.3, "Authentication
of the Server by the Client" -- TLS certificates, SNI, VPN), including phrasing not
present in that example's written answer but present in the raw source text. The
live retrieval context for this question assigned different citation tags to its
chunks, and the fine-tuned model cited none of them -- it recited memorized content
instead of grounding in the tags it was actually given. This is a specific,
reproducible sign of overfitting on a 44-example dataset, not a general drop in
citation ability (10/12 fine-tuned answers still cited correctly).

**smoke-012 (multi-document question, answerable: false) -- known gap,
unaddressed.** Both models answer instead of refusing. This question needs evidence
spanning both NIST and MQTT content; a single retrieval pass surfaces partial,
single-document evidence that neither model recognized as insufficient. None of the
6 `refusal_insufficient_context` training examples covered this scenario -- they all
used context that was plainly irrelevant to the question, not partially relevant.
Fine-tuning could not fix a failure mode it never saw an example of. This matches
the limitation already documented in `docs/RETRIEVAL_SELECTION.md`.

**Why no net improvement.** Llama 3.2 3B Instruct was already reasonably capable at
following the citation-tag system prompt in-context (helped by the worked example
in the prompt itself), which narrowed the room fine-tuning had to improve. 44
examples is small enough that the model appears to have partially memorized
individual training passages rather than fully generalizing the underlying
citation-grounding skill -- consistent with general expectations for QLoRA on very
small, narrow datasets against an already-competent base model.

## Decision

**The fine-tuned adapter is not deployed.** `scripts/ask.py` and the production
pipeline continue to use the baseline Llama 3.2 3B Instruct via Ollama. The adapter
is kept at `models/qlora_v1_adapter` for reference and future experiments, but
shipping it would trade a working baseline for a model with an observed
memorization artifact and no measured upside.

## What would likely help in a v2

- More training examples (100+), with the same source passage appearing behind
  multiple different question phrasings and different citation-tag orderings, so
  the model cannot shortcut to memorizing a fixed context-to-answer mapping.
- Refusal examples that specifically cover partial/single-document evidence for a
  question that needs multiple documents, not just plainly irrelevant context.
- Fewer epochs (2 instead of 4) as a quick first experiment, given eval_loss showed
  diminishing returns after epoch 2 -- untested here, deferred by choice to document
  this result rather than keep iterating.
- Evaluating against the full 32-question `golden_eval_v1.jsonl` rather than the
  12-question smoke set, for more statistical power before drawing conclusions.

## Reproduction

```powershell
python scripts/test_qlora_feasibility.py      # VRAM feasibility check
python scripts/audit_finetuning_lengths.py    # token-length audit
python scripts/train_qlora.py                 # real training run
python scripts/compare_qlora_finetuning.py    # baseline vs fine-tuned comparison
```
