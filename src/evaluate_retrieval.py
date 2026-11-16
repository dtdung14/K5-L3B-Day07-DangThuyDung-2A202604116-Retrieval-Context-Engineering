"""Benchmark dense / BM25 / hybrid retrieval on one split with identical questions.

Usage (repo root; DATABASE_URL in env or .env):
  python src/evaluate_retrieval.py --split dev --output results/retrieval-dev
  python src/evaluate_retrieval.py --split test --allow-test --output results/retrieval-test   # only after freezing config

Per method (and per method+filter) it reports, over answerable questions only:
Hit@1, Evidence Hit@3, MRR@top_k, Context Evidence Rate (evidence inside the packed
top-3 context), mean latency. Abstention questions are excluded from every
retrieval denominator. The question's own `filter` field (user-provided, not derived
from gold) is only applied in the '+filter' configurations.
Outputs: summary.json, summary.md, traces.json.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import hit, pack_context  # noqa: E402
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever, load_chunks  # noqa: E402


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def evaluate(name, retriever, questions, *, use_filter, top_k, budget):
    traces, answerable = [], [q for q in questions if q.get("answerable", True)]
    for q in answerable:
        filters = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=top_k)
        ms = (time.perf_counter() - t0) * 1000
        marks = [hit(r, q["gold"]) for r in results]
        _, ctx_ids = pack_context(results[:3], budget)
        in_ctx = any(m for r, m in zip(results[:3], marks) if r["id"] in ctx_ids)
        rank = next((i for i, m in enumerate(marks, 1) if m), None)
        traces.append({"query_id": q["id"], "split": q["split"], "method": name, "question": q["question"],
                       "filter": filters, "top_k": top_k, "budget_chars": budget, "latency_ms": round(ms, 1),
                       "hit_rank": rank, "context_has_evidence": in_ctx, "context_ids": ctx_ids,
                       "results": [{"id": r["id"], "page": r["page"], "score": round(r["score"], 4),
                                    "rank": r["rank"], "hit": m, "text": r["text"][:200]}
                                   for r, m in zip(results, marks)]})
    n = len(traces)
    ranks = [t["hit_rank"] for t in traces]
    summary = {"method": name, "n_answerable": n,
               "hit@1": sum(r == 1 for r in ranks), "hit@3": sum(r is not None and r <= 3 for r in ranks),
               "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else None,
               "context_evidence": sum(t["context_has_evidence"] for t in traces),
               "mean_latency_ms": round(statistics.mean(t["latency_ms"] for t in traces), 1) if n else None,
               "misses_top3": [t["query_id"] for t in traces if t["hit_rank"] is None or t["hit_rank"] > 3]}
    return summary, traces


def to_markdown(summaries, split, top_k, budget):
    lines = [f"Split: {split} · top_k={top_k} · context budget={budget} chars (top-3 context)\n",
             "| Method | Hit@1 | Evidence Hit@3 | MRR | Context Evidence | Latency ms | Missed (top-3) |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        n = s["n_answerable"]
        lines.append(f"| {s['method']} | {s['hit@1']}/{n} | {s['hit@3']}/{n} | {s['mrr']} | "
                     f"{s['context_evidence']}/{n} | {s['mean_latency_ms']} | {', '.join(s['misses_top3']) or '-'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--chunks", type=Path, default=ROOT / "data/day6/chunks.json")
    p.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--methods", default="dense,bm25,hybrid")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=800, help="context budget in characters, incl. source headers")
    p.add_argument("--no-filter-variants", action="store_true", help="skip the '+filter' configurations")
    p.add_argument("--allow-test", action="store_true", help="required to run the test split (do this once, after freezing)")
    a = p.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("Test split bị khóa: chỉ chạy sau khi chốt cấu hình trên dev. Thêm --allow-test.")
    if a.top_k < 3:
        sys.exit("--top-k phải >= 3 để so sánh Hit@3")

    load_env()
    chunks = load_chunks(a.chunks)
    questions = [q for q in json.loads(a.questions.read_text(encoding="utf-8")) if q["split"] == a.split]
    n_filter = sum(bool(q.get("filter")) for q in questions if q.get("answerable", True))

    bm25 = BM25Retriever(chunks)
    retrievers = {"bm25": bm25}
    wanted = [m.strip() for m in a.methods.split(",")]
    if "dense" in wanted or "hybrid" in wanted:
        dense = DenseRetriever()
        retrievers["dense"] = dense
        retrievers["hybrid"] = HybridRetriever(dense, bm25)

    summaries, all_traces = [], []
    for m in wanted:
        variants = [(m, False)]
        if not a.no_filter_variants and n_filter:
            variants.append((f"{m}+filter", True))
        for name, use_filter in variants:
            s, t = evaluate(name, retrievers[m], questions, use_filter=use_filter, top_k=a.top_k, budget=a.budget)
            summaries.append(s); all_traces.extend(t)
            print(f"{name:14} Hit@1 {s['hit@1']}/{s['n_answerable']}  Hit@3 {s['hit@3']}/{s['n_answerable']}  "
                  f"MRR {s['mrr']}  ctx {s['context_evidence']}/{s['n_answerable']}  {s['mean_latency_ms']}ms")

    a.output.mkdir(parents=True, exist_ok=True)
    meta = {"split": a.split, "top_k": a.top_k, "budget_chars": a.budget, "embedding_model": "intfloat/multilingual-e5-base",
            "n_questions": len(questions), "n_answerable": sum(q.get("answerable", True) for q in questions),
            "n_with_filter": n_filter, "n_abstention_excluded": sum(not q.get("answerable", True) for q in questions)}
    (a.output / "summary.json").write_text(json.dumps({"meta": meta, "results": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output / "traces.json").write_text(json.dumps(all_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output / "summary.md").write_text(to_markdown(summaries, a.split, a.top_k, a.budget), encoding="utf-8")
    print(f"Saved to {a.output}")


if __name__ == "__main__":
    main()
