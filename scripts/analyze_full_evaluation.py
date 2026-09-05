"""Joins data/processed/full_evaluation/results.jsonl against
data/eval/golden_eval_v1.jsonl to break failures and citation metrics down by
question_type and answerable, for step 14's error-taxonomy analysis. Read-only,
prints a text report -- writes nothing.

Usage:
    python scripts/analyze_full_evaluation.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "golden_eval_v1.jsonl"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "full_evaluation" / "results.jsonl"

SYSTEMS = ["naive", "baseline_dense", "advanced_hybrid", "finetuned_hybrid"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    questions = {q["question_id"]: q for q in load_jsonl(GOLDEN_EVAL_PATH)}
    results = load_jsonl(RESULTS_PATH)

    by_system: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_system[row["system"]].append(row)

    print("=" * 100)
    print("FAILURES BY SYSTEM (question_id, question_type, answerable, verdict)")
    print("=" * 100)
    for system in SYSTEMS:
        rows = by_system[system]
        fails = [r for r in rows if r.get("verdict") and r["verdict"] != "PASS"]
        print(f"\n--- {system} ({len(fails)} failing / {len(rows)} total) ---")
        for r in fails:
            q = questions[r["question_id"]]
            print(
                f"  {r['question_id']:<12} type={q['question_type']:<28} "
                f"answerable={q['answerable']!s:<6} verdict={r['verdict']:<28} "
                f"cited={r.get('cited_chunk_ids')}"
            )

    print("\n" + "=" * 100)
    print("CITATION METRICS BY QUESTION_TYPE (per system, answerable questions only)")
    print("=" * 100)
    question_types = sorted({q["question_type"] for q in questions.values()})
    for system in SYSTEMS:
        rows = {r["question_id"]: r for r in by_system[system]}
        print(f"\n--- {system} ---")
        for qtype in question_types:
            qids = [qid for qid, q in questions.items() if q["question_type"] == qtype and q["answerable"]]
            precisions = [rows[qid]["citation_precision"] for qid in qids if qid in rows and rows[qid].get("citation_precision") is not None]
            recalls = [rows[qid]["citation_recall"] for qid in qids if qid in rows and rows[qid].get("citation_recall") is not None]
            if not qids:
                continue
            p = round(mean(precisions), 3) if precisions else None
            r_ = round(mean(recalls), 3) if recalls else None
            print(f"  {qtype:<28} n={len(qids):<3} precision={p!s:<8} recall={r_!s:<8}")

    print("\n" + "=" * 100)
    print("UNANSWERABLE QUESTIONS -- did each system correctly refuse?")
    print("=" * 100)
    unanswerable_qids = [qid for qid, q in questions.items() if not q["answerable"]]
    print(f"{len(unanswerable_qids)} unanswerable questions in the golden set: {unanswerable_qids}\n")
    for system in SYSTEMS:
        rows = {r["question_id"]: r for r in by_system[system]}
        refused_count = sum(1 for qid in unanswerable_qids if qid in rows and rows[qid]["refused"])
        print(f"  {system:<20} correctly refused {refused_count}/{len(unanswerable_qids)}")


if __name__ == "__main__":
    main()