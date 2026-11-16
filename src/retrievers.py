"""Three retrieval methods over the Day 6 corpus (table chunks_b, 279 paragraph chunks).

- DenseRetriever : multilingual-e5-base query embedding + pgvector cosine (Day 6 store)
- BM25Retriever  : lexical BM25 over data/day6/chunks.json (no extra dependency)
- HybridRetriever: Reciprocal Rank Fusion of the dense and BM25 rankings

All share `search(query, *, filters=None, top_k=3)` (the starter's Retriever
protocol) and return dicts: id, text, source_id, source, page, metadata, score, rank.
Gold labels are never read here. Filters must come from the user/query, not gold.
"""
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_FILTERS = {"section", "source_id"}  # whitelist; section lives in metadata
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # must match the Day 6 index (768d)


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise ValueError(f"filter không được hỗ trợ: {sorted(bad)}")
    return filters


def _result(chunk: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"],
            "source": chunk["source"], "page": chunk["page"],
            "metadata": chunk.get("metadata", {}), "score": float(score), "rank": rank}


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class DenseRetriever:
    """Query embedding (with the e5 'query: ' prefix) against the Day 6 pgvector table."""

    def __init__(self, database_url: str | None = None, *, table: str = "chunks_b",
                 model_name: str = EMBEDDING_MODEL) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from sentence_transformers import SentenceTransformer
        self.url = database_url or os.environ.get("DATABASE_URL")
        if not self.url:
            raise RuntimeError("Thiếu DATABASE_URL")
        if not re.fullmatch(r"[a-z_]+", table):
            raise ValueError("tên bảng không hợp lệ")
        self.table, self.model_name = table, model_name
        self._model = SentenceTransformer(model_name)
        self._psycopg, self._register = psycopg, register_vector
        try:
            self._conn = psycopg.connect(self.url, connect_timeout=10, autocommit=True)
        except Exception as e:  # no silent fallback
            raise RuntimeError(f"Không kết nối được PostgreSQL: {e}") from e
        register_vector(self._conn)

    def embed_query(self, query: str):
        prefix = "query: " if "e5" in self.model_name.lower() else ""
        return self._model.encode([prefix + query], normalize_embeddings=True)[0]

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        where, params = [], []
        if "section" in filters:
            where.append("section = %s"); params.append(filters["section"])
        if "source_id" in filters:
            where.append("document_id = %s"); params.append(filters["source_id"])
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        vec = self.embed_query(query)
        rows = self._conn.execute(
            f"SELECT chunk_id, document_id, source, page, section, text, metadata, "
            f"1 - (embedding <=> %s) AS score FROM {self.table} {clause} "
            f"ORDER BY embedding <=> %s LIMIT %s",
            [vec, *params, vec, top_k]).fetchall()
        out = []
        for i, (cid, doc, src, page, section, text, meta, score) in enumerate(rows, 1):
            m = dict(meta or {}); m["section"] = section
            out.append(_result({"id": cid, "text": text, "source_id": doc, "source": src,
                                "page": page, "metadata": m}, score, i))
        return out


class BM25Retriever:
    """Okapi BM25 over the exported chunks (k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, q_tokens: list[str], i: int) -> float:
        d, ln, s = self.docs[i], self.lens[i], 0.0
        for t in q_tokens:
            tf = d.get(t)
            if tf:
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / self.avg))
        return s

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        q_tokens = list(dict.fromkeys(tokenize(query)))
        scored = []
        for i, c in enumerate(self.chunks):
            if "section" in filters and c["metadata"].get("section") != filters["section"]:
                continue
            if "source_id" in filters and c["source_id"] != filters["source_id"]:
                continue
            scored.append((self._score(q_tokens, i), i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_result(self.chunks[i], s, r) for r, (s, i) in enumerate(scored[:top_k], 1) if s > 0]


class HybridRetriever:
    """RRF: score(d) = sum 1 / (k + rank_r(d)) over dense and BM25 rankings (rank-based, no
    score normalization needed). Each retriever contributes its top `pool` candidates."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, *, pool: int = 20, rrf_k: int = 60) -> None:
        self.dense, self.bm25, self.pool, self.rrf_k = dense, bm25, pool, rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        fused: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for r in (self.dense, self.bm25):
            for hit in r.search(query, filters=filters, top_k=self.pool):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (self.rrf_k + hit["rank"])
                by_id.setdefault(hit["id"], hit)
        ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        return [_result(by_id[cid], fused[cid], r) for r, cid in enumerate(ranked, 1)]
