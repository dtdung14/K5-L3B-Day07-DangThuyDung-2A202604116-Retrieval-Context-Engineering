"""Manual grading helper for agent runs.

1) export : build a readable review.md + a reviews.json TEMPLATE from a run's traces.json
     python src/review.py export --run results/agent-dev-hybrid-b800 --questions data/questions.json
2) (you)  : read review.md, then fill reviews.json (null -> value). Never overwritten unless --force.
3) metrics: validate labels and aggregate with starter/agent_metrics.py
     python src/review.py metrics --run results/agent-dev-hybrid-b800

Label meaning (BENCHMARK.md):
  answer_grade (answerable questions): 0 wrong/missing evidence or refused, 1 partly right, 2 right and complete
  citation_support: one true/false per citation = the cited chunk really supports the answer
  abstention_correct (unanswerable questions): true if it refused without adding unsupported claims
  context_grade (optional, answerable): 0 no evidence in final context, 1 partial, 2 enough evidence + source
The gold answer/evidence is shown only to you, the grader; it never reaches the model.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_index(trace: dict) -> dict:
    return {r["id"]: r for e in trace["events"] if e.get("event") == "search" for r in e["results"]}


def last_context(trace: dict) -> tuple[str, list[str]]:
    searches = [e for e in trace["events"] if e.get("event") == "search"]
    return (searches[-1]["context"], searches[-1]["context_ids"]) if searches else ("", [])


def export(a) -> None:
    traces = load(a.run / "traces.json")
    qs = {q["id"]: q for q in load(a.questions)}
    md = [f"# Review: {a.run.name}\n", "Điền nhãn vào reviews.json (xem hướng dẫn nhãn ở đầu src/review.py)."]
    template = []
    for t in traces:
        q = qs[t["query_id"]]
        answerable = q.get("answerable", True)
        chunks, (ctx, ctx_ids) = chunk_index(t), last_context(t)
        md.append(f"\n---\n## {t['query_id']} · {t['split']} · {'có đáp án' if answerable else 'THIẾU BẰNG CHỨNG'} · status={t['status']}")
        md.append(f"**Question:** {q['question']}\n")
        if answerable:
            g = q["gold"][0]
            md.append(f"**Expected:** {q['expected_answer']}  \n**Gold evidence (p{g['page']}):** {g['evidence']}\n")
        else:
            md.append(f"**Expected:** từ chối. Lý do: {q.get('absence_reason')}\n")
        queries = [e["query"] for e in t["events"] if e.get("event") == "search"]
        md.append(f"**Queries ({len(queries)}):** " + " | ".join(queries))
        md.append(f"**Answer:** {t.get('answer')}")
        if t.get("error"):
            md.append(f"**ERROR:** {t['error']}")
        for cid in t["citations"]:
            c = chunks.get(cid)
            md.append(f"- Citation `{cid}` (p{c['page'] if c else '?'}): {(c['text'][:400] if c else '(không có)')}")
        md.append(f"\n<details><summary>Context cuối ({len(ctx_ids)} chunk, {len(ctx)} ký tự)</summary>\n\n{ctx}\n\n</details>")
        entry = {"query_id": t["query_id"], "notes": ""}
        if t["status"] == "error":
            entry["note"] = "lỗi hệ thống; không cần nhãn"
        else:
            entry["citation_support"] = [None] * len(t["citations"])
            if answerable:
                entry["answer_grade"] = 0 if t["status"] == "abstain" else None
                entry["context_grade"] = None
            else:
                entry["abstention_correct"] = None
        template.append(entry)
    (a.run / "review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    target = a.run / "reviews.json"
    if target.exists() and not a.force:
        print(f"{target} đã tồn tại, giữ nguyên (dùng --force để ghi đè). Đã cập nhật review.md.")
    else:
        target.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Đã tạo {a.run/'review.md'} và {target}")


def metrics(a) -> None:
    traces = load(a.run / "traces.json")
    reviews = load(a.run / "reviews.json")
    empty = [r["query_id"] for r in reviews
             if any(v is None for k, v in r.items() if k in ("answer_grade", "abstention_correct"))
             or any(v is None for v in r.get("citation_support", []))]
    if empty:
        sys.exit(f"Còn nhãn chưa điền (null) ở: {', '.join(empty)}")
    cases = [{"id": t["query_id"], "split": t["split"], "answerable": t["answerable"]} for t in traces]
    report = {"run": a.run.name, **aggregate(cases, traces, reviews)}
    ctx = [r["context_grade"] for r in reviews if r.get("context_grade") is not None]
    if ctx:
        report["context_grade"] = {str(i): ctx.count(i) for i in (0, 1, 2)} | {"N": len(ctx)}
    (a.run / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--run", type=Path, required=True)
    e.add_argument("--questions", type=Path, default=ROOT / "data/questions.json"); e.add_argument("--force", action="store_true")
    m = sub.add_parser("metrics"); m.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    {"export": export, "metrics": metrics}[a.cmd](a)


if __name__ == "__main__":
    main()
