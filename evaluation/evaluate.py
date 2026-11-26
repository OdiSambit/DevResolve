"""Retrieval evaluation: Recall@5, MRR and insufficient-evidence behaviour.

Usage:  python evaluation/evaluate.py            (uses the configured DB / embedder)
        DEVRESOLVE_EMBEDDER=hashing python evaluation/evaluate.py

Generation faithfulness and citation correctness need manual inspection of AI answers
(see README 'Evaluation'); only retrieval metrics are computed automatically here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import database  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402
from seed_data import seed_if_empty  # noqa: E402


def run(engine: RAGEngine, questions: list, k: int = 5) -> dict:
    title_to_id = {e["title"]: e["id"] for e in database.list_experiences(limit=10000)}
    rows, rr, hits_at_k, n_pos = [], [], 0, 0
    neg_total, neg_rejected = 0, 0
    for item in questions:
        expected = {title_to_id[t] for t in item["expected_titles"] if t in title_to_id}
        results = engine.search(item["query"], top_k=k)
        ids = [r["id"] for r in results]
        top_score = results[0]["score"] if results else 0.0
        if expected:
            n_pos += 1
            rank = next((i + 1 for i, rid in enumerate(ids) if rid in expected), None)
            hits_at_k += 1 if rank else 0
            rr.append(1 / rank if rank else 0.0)
            rows.append({"query": item["query"][:60], "type": "positive", "rank": rank or "-",
                         "top_score": round(top_score, 3), "pass": bool(rank)})
        else:
            neg_total += 1
            rejected = top_score < engine.min_score
            neg_rejected += 1 if rejected else 0
            rows.append({"query": item["query"][:60], "type": "no-match", "rank": "-",
                         "top_score": round(top_score, 3), "pass": rejected})
    return {"embedder": engine.embedder.name, "k": k, "positive_queries": n_pos,
            "recall_at_k": round(hits_at_k / n_pos, 3) if n_pos else None,
            "mrr": round(sum(rr) / n_pos, 3) if n_pos else None,
            "no_match_queries": neg_total,
            "insufficient_evidence_rate": round(neg_rejected / neg_total, 3) if neg_total else None,
            "min_score_threshold": engine.min_score, "details": rows}


if __name__ == "__main__":
    database.configure()
    seed_if_empty()
    engine = RAGEngine()
    questions = json.loads((ROOT / "evaluation" / "questions.json").read_text())["questions"]
    report = run(engine, questions)
    print(pd.DataFrame(report["details"]).to_string(index=False))
    print(f"\nEmbedder: {report['embedder']}   threshold: {report['min_score_threshold']}")
    print(f"Recall@{report['k']}: {report['recall_at_k']}   MRR: {report['mrr']}   "
          f"({report['positive_queries']} positive queries)")
    print(f"Insufficient-evidence rate on no-match queries: {report['insufficient_evidence_rate']} "
          f"({report['no_match_queries']} queries)")
    (ROOT / "evaluation" / "results.json").write_text(json.dumps(report, indent=2))
