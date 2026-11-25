"""Unit tests for forge.retrieval (hybrid BM25 + dense RRF, persistence, hooks)."""

from __future__ import annotations

from pathlib import Path

from forge.ingest import ingest_folder
from forge.llm import FakeLLM, deterministic_hash_embedding
from forge.retrieval import HybridIndex, HybridRetriever, RetrievalConfig

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "phase1_sample"


def test_hybrid_retrieval_recall_and_persistence(tmp_path: Path) -> None:
    _, chunks, _ = ingest_folder(FIXTURE_DIR, chunk_size=300, overlap=40)
    llm = FakeLLM(embed_fn=lambda texts: [deterministic_hash_embedding(t, dim=64) for t in texts])

    index = HybridIndex.build(chunks, llm_client=llm)
    run_index_dir = tmp_path / "run_001" / "index"
    index.save(run_index_dir)
    loaded = HybridIndex.load(run_index_dir)

    retriever = HybridRetriever(
        loaded,
        llm_client=llm,
        config=RetrievalConfig(alpha=0.5, top_k=2),
    )

    eval_pairs = [
        ("What depth should hydrophone pods be deployed?", "acoustic_guide.md"),
        ("How often do sensors undergo salt-fog calibration?", "buoy_manual.txt"),
        ("Which sensor is model SalinityPro-C?", "sensors.csv"),
        ("What is the minimum radio watch interval?", "field_notes.txt"),
    ]
    hits_found = 0
    for query, expected_doc in eval_pairs:
        results = retriever.search(query)
        assert results
        assert all(
            results[i].score >= results[i + 1].score for i in range(len(results) - 1)
        )
        if any(r.chunk.doc_id == expected_doc for r in results):
            hits_found += 1

    recall_at_2 = hits_found / len(eval_pairs)
    assert recall_at_2 == 1.0


def test_alpha_min_score_and_query_rewrite_hook() -> None:
    _, chunks, _ = ingest_folder(FIXTURE_DIR, chunk_size=400, overlap=40)
    llm = FakeLLM()
    index = HybridIndex.build(chunks, llm_client=llm)

    # Query rewrite hook is off by default
    seen_rewrites: list[str] = []

    def rewrite_hook(q: str) -> str:
        seen_rewrites.append(q)
        return "salt-fog calibration Depot North"

    retriever_off = HybridRetriever(
        index,
        llm_client=llm,
        config=RetrievalConfig(alpha=0.0, top_k=3, query_rewrite=False),
        query_rewrite_hook=rewrite_hook,
    )
    retriever_off.search("calibration schedule")
    assert seen_rewrites == []

    # When enabled, hook rewrites the query
    retriever_on = HybridRetriever(
        index,
        llm_client=llm,
        config=RetrievalConfig(alpha=0.5, top_k=3, query_rewrite=True),
        query_rewrite_hook=rewrite_hook,
    )
    rewritten_hits = retriever_on.search("unmatched phrase")
    assert seen_rewrites == ["unmatched phrase"]
    assert rewritten_hits[0].metadata["effective_query"] == "salt-fog calibration Depot North"
    assert rewritten_hits[0].chunk.doc_id == "buoy_manual.txt"

    # High min_score filters out all results
    filtered = retriever_on.search("calibration", min_score=0.99)
    assert filtered == []
