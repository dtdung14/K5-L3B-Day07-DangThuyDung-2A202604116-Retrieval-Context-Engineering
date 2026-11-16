"""Export Day 6 chunks (pgvector table chunks_b) and benchmark to Day 7 JSON.

"""
import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import dotenv_values

TABLE = "chunks_b"  # paragraph chunking; Day 6 best config (C) reuses this table


def export_chunks(database_url: str) -> list[dict]:
    with psycopg.connect(database_url, connect_timeout=10) as conn:
        rows = conn.execute(
            f"SELECT chunk_id, document_id, source, page, section, text, metadata "
            f"FROM {TABLE} ORDER BY page, chunk_id"
        ).fetchall()
    chunks = []
    for chunk_id, document_id, source, page, section, text, metadata in rows:
        meta = dict(metadata or {})
        meta["section"] = section
        chunks.append({"id": chunk_id, "text": text, "source_id": document_id,
                       "source": source, "page": page, "metadata": meta})
    return chunks


def convert_question(q: dict) -> dict:
    out = {"id": q["id"], "split": q["split"], "question": q["question"],
           "expected_answer": q["expected_answer"],
           "gold": [{"source_id": q["gold_source_id"], "page": q["gold_page"],
                     "evidence": q["evidence_quote"]}]}
    if q.get("filter"):
        out["filter"] = q["filter"]
    if q.get("type"):
        out["type"] = q["type"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day6", required=True, type=Path, help="Day 6 starter folder (has .env, benchmark/)")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    url = os.environ.get("DATABASE_URL") or dotenv_values(a.day6 / ".env").get("DATABASE_URL")
    if not url:
        sys.exit("Thiếu DATABASE_URL (đặt biến môi trường hoặc có .env trong thư mục Day 6)")
    chunks = export_chunks(url)
    bench = json.loads((a.day6 / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
    questions = [convert_question(q) for q in bench]

    a.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("chunks.json", chunks), ("questions.json", questions)):
        (a.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ids = {c["id"] for c in chunks}
    assert len(ids) == len(chunks), "chunk ID bị trùng"
    bad = [q["id"] for q in questions
           if not any(q["gold"][0]["evidence"] in c["text"] and c["page"] == q["gold"][0]["page"] for c in chunks)]
    print(f"{len(chunks)} chunks, {len(questions)} questions -> {a.output}")
    print(f"Questions whose evidence is not verbatim in one chunk: {bad or 'không có'}")


if __name__ == "__main__":
    main()
