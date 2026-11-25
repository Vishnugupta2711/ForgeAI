"""Hybrid BM25 + dense embedding retriever with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from forge.ingest.models import Chunk
from forge.llm.client import LLMClient
from forge.retrieval.models import RetrievalConfig, SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer for BM25 indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


def _chunk_bm25_tokens(chunk: Chunk) -> list[str]:
    heading = chunk.section_heading or ""
    combined = f"{heading} {chunk.text}".strip()
    tokens = tokenize(combined)
    return tokens or ["_empty_"]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class HybridIndex:
    """Persisted hybrid index storing chunks, BM25 index, and dense embeddings."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray | None = None,
        *,
        embed_model: str = "",
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = (
            _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
            if embeddings is not None and len(chunks) > 0
            else None
        )
        self.embed_model = embed_model
        tokenized_corpus = [_chunk_bm25_tokens(c) for c in self.chunks]
        self.bm25: BM25Okapi | None = BM25Okapi(tokenized_corpus) if self.chunks else None

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        *,
        llm_client: LLMClient | None = None,
        embed_model: str | None = None,
        batch_size: int = 32,
    ) -> HybridIndex:
        """Build a HybridIndex from chunks and optional LLMClient embeddings."""
        if not chunks:
            return cls(chunks=[], embeddings=None, embed_model=embed_model or "")

        resolved_model = embed_model or (
            llm_client.config.models.embed if llm_client is not None else ""
        )
        embeddings: np.ndarray | None = None
        if llm_client is not None:
            vectors: list[list[float]] = []
            texts = [c.text for c in chunks]
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                vectors.extend(llm_client.embed(batch, model=resolved_model))
            embeddings = np.asarray(vectors, dtype=np.float32)

        return cls(chunks=chunks, embeddings=embeddings, embed_model=resolved_model)

    def save(self, index_dir: Path | str) -> Path:
        """Persist chunks, embeddings, and index metadata to a run directory."""
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)

        chunks_path = target / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")

        has_embeddings = self.embeddings is not None and len(self.chunks) > 0
        if has_embeddings and self.embeddings is not None:
            np.save(target / "embeddings.npy", self.embeddings)
        else:
            emb_path = target / "embeddings.npy"
            if emb_path.exists():
                emb_path.unlink()

        meta = {
            "embed_model": self.embed_model,
            "chunk_count": len(self.chunks),
            "has_embeddings": has_embeddings,
            "embedding_dim": (
                int(self.embeddings.shape[1])
                if has_embeddings and self.embeddings is not None and self.embeddings.ndim == 2
                else 0
            ),
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, index_dir: Path | str) -> HybridIndex:
        """Load a previously persisted HybridIndex from disk."""
        target = Path(index_dir)
        meta_path = target / "meta.json"
        chunks_path = target / "chunks.jsonl"
        if not meta_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(f"HybridIndex files not found in: {target}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))

        embeddings: np.ndarray | None = None
        emb_path = target / "embeddings.npy"
        if meta.get("has_embeddings") and emb_path.is_file() and chunks:
            embeddings = np.load(emb_path)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            embed_model=str(meta.get("embed_model", "")),
        )


class HybridRetriever:
    """Hybrid BM25 + dense embedding retriever using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        index: HybridIndex,
        *,
        llm_client: LLMClient | None = None,
        config: RetrievalConfig | None = None,
        query_rewrite_hook: Callable[[str], str] | None = None,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.config = config or RetrievalConfig()
        self.query_rewrite_hook = query_rewrite_hook

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        alpha: float | None = None,
        min_score: float | None = None,
        query_rewrite: bool | None = None,
    ) -> list[SearchHit]:
        """Search the index and return ranked SearchHits fused via RRF."""
        if not self.index.chunks:
            return []

        resolved_top_k = self.config.top_k if top_k is None else top_k
        resolved_alpha = self.config.alpha if alpha is None else alpha
        resolved_min = self.config.min_score if min_score is None else min_score
        use_rewrite = self.config.query_rewrite if query_rewrite is None else query_rewrite
        rrf_k = self.config.rrf_k

        if not 0.0 <= resolved_alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        effective_query = query.strip()
        if use_rewrite and self.query_rewrite_hook is not None:
            effective_query = self.query_rewrite_hook(effective_query).strip() or effective_query

        if not effective_query:
            return []

        n = len(self.index.chunks)

        # 1. BM25 scores and ranks (only positive BM25 scores get a rank)
        bm25_scores = np.zeros(n, dtype=np.float64)
        bm25_ranks: dict[int, int] = {}
        if self.index.bm25 is not None:
            q_tokens = tokenize(effective_query)
            if q_tokens:
                bm25_scores = np.asarray(self.index.bm25.get_scores(q_tokens), dtype=np.float64)
                pos_indices = [idx for idx in range(n) if bm25_scores[idx] > 0.0]
                pos_indices.sort(key=lambda idx: (-bm25_scores[idx], idx))
                for rank, idx in enumerate(pos_indices, start=1):
                    bm25_ranks[idx] = rank

        # 2. Dense cosine similarity scores and ranks
        dense_scores = np.zeros(n, dtype=np.float64)
        dense_ranks: dict[int, int] = {}
        has_dense = (
            self.index.embeddings is not None
            and self.llm_client is not None
            and resolved_alpha > 0.0
        )
        if has_dense and self.index.embeddings is not None and self.llm_client is not None:
            embed_model = self.index.embed_model or self.llm_client.config.models.embed
            q_vecs = self.llm_client.embed([effective_query], model=embed_model)
            q_arr = _l2_normalize_rows(np.asarray(q_vecs, dtype=np.float32))
            dense_scores = (self.index.embeddings @ q_arr[0]).astype(np.float64)
            dense_order = sorted(range(n), key=lambda idx: (-dense_scores[idx], idx))
            for rank, idx in enumerate(dense_order, start=1):
                dense_ranks[idx] = rank

        # 3. Reciprocal Rank Fusion
        hits: list[SearchHit] = []
        for idx, chunk in enumerate(self.index.chunks):
            b_rank = bm25_ranks.get(idx)
            d_rank = dense_ranks.get(idx)

            if has_dense:
                b_part = ((1.0 - resolved_alpha) / (rrf_k + b_rank)) if b_rank is not None else 0.0
                d_part = (resolved_alpha / (rrf_k + d_rank)) if d_rank is not None else 0.0
                fused = b_part + d_part
            else:
                fused = (1.0 / (rrf_k + b_rank)) if b_rank is not None else 0.0

            if fused <= 0.0:
                continue
            if resolved_min is not None and fused < resolved_min:
                continue

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=round(float(fused), 6),
                    bm25_rank=b_rank,
                    dense_rank=d_rank,
                    bm25_score=round(float(bm25_scores[idx]), 6),
                    dense_score=round(float(dense_scores[idx]), 6),
                    metadata={"effective_query": effective_query},
                )
            )

        hits.sort(
            key=lambda h: (-h.score, -h.dense_score, -h.bm25_score, h.chunk.chunk_id)
        )
        return hits[:resolved_top_k]
