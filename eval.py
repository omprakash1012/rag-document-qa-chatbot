"""
Retrieval evaluation harness.

Runs a small labeled set of (question, expected_source_file) pairs against
the live FAISS index and reports Hit Rate@k and Mean Reciprocal Rank (MRR) -
i.e. does retrieval actually surface the right document, and how highly does
it rank it. This is a retrieval-quality check, not an answer-quality/LLM-
hallucination check (that would require a judge model and an OPENAI_API_KEY,
which this repo doesn't assume is available).

Usage:
    python eval.py --index-dir vectorstore --k 4
"""
import argparse
import json
import os

from rag_chain import RAGPipeline

EVAL_SET = [
    {"question": "How many days can employees work remotely per week?", "expected_source": "employee_handbook.md"},
    {"question": "How many PTO days do full-time employees accrue per year?", "expected_source": "employee_handbook.md"},
    {"question": "How long do employees have to submit an expense report?", "expected_source": "employee_handbook.md"},
    {"question": "When does open enrollment for health benefits occur?", "expected_source": "employee_handbook.md"},
    {"question": "How often do formal performance reviews happen?", "expected_source": "employee_handbook.md"},
    {"question": "What is the refund policy?", "expected_source": "product_faq.md"},
    {"question": "How do I upgrade or downgrade my subscription plan?", "expected_source": "product_faq.md"},
    {"question": "Does the platform support single sign-on?", "expected_source": "product_faq.md"},
    {"question": "What are the API rate limits?", "expected_source": "product_faq.md"},
    {"question": "How is my data backed up?", "expected_source": "product_faq.md"},
]


def evaluate(index_dir: str, k: int):
    rag = RAGPipeline(index_dir=index_dir, k=k)

    per_question = []
    hits = 0
    reciprocal_ranks = []

    for item in EVAL_SET:
        docs = rag.retrieve(item["question"])
        sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]

        rank = None
        for i, src in enumerate(sources, start=1):
            if src == item["expected_source"]:
                rank = i
                break

        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)

        per_question.append({
            "question": item["question"],
            "expected_source": item["expected_source"],
            "retrieved_sources": sources,
            "hit": hit,
            "rank": rank,
        })

    n = len(EVAL_SET)
    results = {
        "k": k,
        "n_questions": n,
        "hit_rate_at_k": round(hits / n, 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / n, 4),
        "per_question": per_question,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="vectorstore")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default="reports/eval_results.json")
    args = parser.parse_args()

    results = evaluate(args.index_dir, args.k)

    print(f"Hit Rate@{results['k']}: {results['hit_rate_at_k']:.0%} "
          f"({sum(1 for q in results['per_question'] if q['hit'])}/{results['n_questions']})")
    print(f"Mean Reciprocal Rank: {results['mean_reciprocal_rank']}")
    print()
    for q in results["per_question"]:
        status = "HIT " if q["hit"] else "MISS"
        print(f"[{status}] rank={q['rank']}  {q['question']!r} -> expected {q['expected_source']}, "
              f"got {q['retrieved_sources']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")
