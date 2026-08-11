# Fine-tuning Dataset Card: train_v1.jsonl (Step 11)

## Overview

54 hand-authored, chat-format supervised fine-tuning examples for adapting the
generator model (Llama 3.2 3B, served via Ollama) to OTSentinel AI's citation-tag
answer contract. Every example follows the exact system-prompt and context-formatting
structure produced by `app/generation/prompts.py` at inference time (`build_messages()`),
so training and live serving see the same input shape.

- File: `data/finetuning/train_v1.jsonl`
- Schema: `app.schemas.finetuning.FinetuningExample`
- Dataset SHA-256: `271ae670a427beb1f76482f0c5924027ec2c516ac5161d8e661eecf0f893fd10`
- Source documents: NIST SP 800-82r3 (Guide to Operational Technology Security),
  MQTT Version 5.0 (OASIS Standard)

## Composition

| Dimension | Breakdown |
|---|---|
| Total examples | 54 |
| Split | 44 train / 10 validation |
| Language | 46 English / 8 Romanian |
| Source document | 26 NIST SP 800-82r3 / 22 MQTT v5.0 / 6 refusal (no source chunk) |
| task_type | mitigation_recommendation 13, protocol_security_explanation 15, concept_explanation 8, refusal_insufficient_context 6, romanian_response 6, comparison 4, factual_answer 2 |

`advisory_summary` and `attack_mapping` task types are defined in the schema but have
zero examples in v1 -- the corpus has no CISA advisory or MITRE ATT&CK source ingested
yet, so writing those examples now would mean fabricating source material. Add them
once those sources are in the pipeline, consistent with how `golden_eval_v1.jsonl`
handled the same gap.

## Methodology

1. **Candidate sampling.** `scripts/sample_finetuning_candidates.py` sampled 70 chunks
   per source document (evenly spaced by document position), excluding any chunk
   shorter than 40 tokens and any chunk already reserved for `golden_eval_v1.jsonl`.
2. **Manual authoring.** Each example was hand-written against the real chunk text
   returned by the sampler -- nothing was generated from memory of the source
   documents. Context passages in the `user` message are copied verbatim from the
   sampled chunks.
3. **Prompt-format fidelity.** System prompt and context formatting (`[S1]`/`[S2]`
   tags, `Source: {title} | {heading}` prefix, the trailing citation reminder) match
   `app/generation/prompts.py` exactly, so fine-tuning shapes the model toward
   behavior the live pipeline can actually use.
4. **Negative examples.** 6 `refusal_insufficient_context` examples pair a real
   (but non-answering) context passage with a question the context doesn't support --
   including "hard" cases where the context is topically related but lacks the
   specific fact asked for (e.g., vulnerability-scanning cadence, redundancy
   architecture), not just obviously irrelevant context. All 6 use the exact
   `REFUSAL_TEXT` string from the live system, in English regardless of question
   language, since the live system does not localize that string.

## Leakage prevention

Every `source_chunk_id` in this dataset was checked against the union of
`relevant_chunk_ids` across all 32 questions in `golden_eval_v1.jsonl` (23 reserved
chunk IDs). Zero overlap confirmed via `scripts/validate_finetuning_dataset.py`.

## Quality checks (scripts/validate_finetuning_dataset.py)

Run automatically before this dataset is used for training:

- Schema validation against `FinetuningExample`.
- No duplicate `example_id`.
- No `source_chunk_id` overlap with the golden eval set.
- Every non-refusal answer cites only tags defined in its own context (no invented
  `[Sx]` tags) and cites at least one tag.
- Every refusal example's answer matches `REFUSAL_TEXT` verbatim.
- Near-duplicate answer detection via `difflib.SequenceMatcher` (text-similarity,
  not embedding-based -- see the script's docstring for why, and how to upgrade it).

All checks passed on this version (54/54 examples, 0 duplicates, 0 leakage,
0 citation issues, 0 refusal-text mismatches, 0 near-duplicate pairs at threshold 0.90).

## Known limitations

- **Scale.** 54 examples is well short of the original project doc's 1,500-3,000
  target. That target assumed five source families; this corpus has two. Scaling
  further means either ingesting more sources (CISA advisories, MITRE ATT&CK) or
  accepting diminishing returns from resampling the same two documents. Treat this
  as v1, not final.
- **No advisory_summary / attack_mapping coverage** (see above).
- **Comparison examples are intra-document only** -- no cross-document (NIST-vs-MQTT)
  comparisons, consistent with the retrieval system's documented limitation on
  multi-document queries (see `docs/RETRIEVAL_SELECTION.md`).
- **Near-duplicate detection is text-similarity based, not semantic.** It won't catch
  a paraphrase that reuses the same facts in different words. Fine for a 54-example
  set reviewed by hand; would need embedding-based dedup at larger scale.

## Reproducing validation

```powershell
python scripts/validate_finetuning_dataset.py
python scripts/validate_finetuning_dataset.py --similarity-threshold 0.8
```
