# Retrieval Configuration Selection (Step 10)

## Configurations evaluated

All four were measured with the same harness (`scripts/evaluate_retrieval.py`) against the
19 answerable questions in the `validation` split of `data/eval/golden_eval_v1.jsonl`,
logged to MLflow under the `otsentinel-retrieval` experiment.

| Configuration | Recall@5 | Recall@10 | Precision@5 | MRR | nDCG@10 | Avg. latency |
|---|---|---|---|---|---|---|
| Dense only (BGE-M3) | 0.79 | 0.89 | 0.158 | 0.72 | 0.76 | ~465ms |
| Hybrid (dense + BM25 sparse, RRF fusion) | 0.89 | 0.95 | 0.179 | **0.79** | **0.83** | ~434ms |
| Hybrid + reranker (25 candidates) | 0.95 | 0.95 | 0.190 | 0.73 | 0.79 | ~30s |
| Hybrid + reranker (15 candidates) | 0.95 | **1.00** | **0.190** | 0.74 | 0.80 | ~30s |

## Selected configuration: Hybrid (dense + sparse, RRF fusion), no reranker

## Rationale

- Hybrid retrieval beats the dense-only baseline on every metric, confirming sparse
  retrieval recovers cases dense embeddings miss (e.g. geval-005, a total miss under
  dense-only, found at rank 3 under hybrid).
- The cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) further improves recall and
  precision, but is not a strict win: it *reduces* MRR and nDCG relative to hybrid alone,
  because on several already-correct rank-1 results it demoted the right passage to
  rank 2-3. It also introduces ~30 seconds of latency per query on CPU (no GPU available),
  which is impractical for an interactive system on top of the 5-60s the LLM generation
  step already takes.
- Given the live pipeline (`scripts/ask.py`) only sends the top 5 chunks to the generator,
  MRR/nDCG (how well-ranked the best passage is within that window) matter more
  operationally than raw recall@10. Hybrid alone wins there.

## Known limitations of the selected configuration

- Multi-document questions spanning both NIST and MQTT content (geval-032, smoke-012)
  still correctly refuse rather than hallucinate -- evidence for the correct source isn't
  reliably surfaced by a single blended query, even with hybrid retrieval. Revisit if/when
  query decomposition or multi-query retrieval is added.
- Romanian-language citation formatting is occasionally dropped by the 3B generator model
  (smoke-007), independent of retrieval quality. Candidate fix: include Romanian
  citation-formatted examples in the step 11 fine-tuning dataset.

## Reproducing these results

```powershell
python scripts/evaluate_retrieval.py --search-mode dense --run-name <name>
python scripts/evaluate_retrieval.py --search-mode hybrid --run-name <name>
python scripts/evaluate_retrieval.py --search-mode hybrid_rerank --rerank-candidates 25 --run-name <name>
python scripts/evaluate_retrieval.py --search-mode hybrid_rerank --rerank-candidates 15 --run-name <name>
```

The reranker remains available via `--search-mode hybrid_rerank` for future use (e.g. if
this project is ever run with GPU access) but is not wired into the live `scripts/ask.py`
pipeline.
