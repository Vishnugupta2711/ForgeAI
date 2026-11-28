"""Document-profile-driven heuristics for deriving retrieval parameters."""

from __future__ import annotations

from typing import Any

from forge.agent.schema import AgentRetrievalSettings
from forge.ingest.models import CorpusProfile
from forge.intake.models import RequirementsDocument


def _estimate_language_profile(profile: CorpusProfile, brief: str) -> str:
    sample = f"{brief} " + " ".join(
        " ".join(doc.headings[:5]) for doc in profile.documents
    )
    if not sample.strip():
        return "ascii"
    non_ascii = sum(1 for ch in sample if ord(ch) > 127)
    ratio = non_ascii / max(1, len(sample))
    return "multilingual" if ratio > 0.05 else "ascii"


def derive_retrieval_settings(
    profile: CorpusProfile,
    requirements: RequirementsDocument,
) -> tuple[AgentRetrievalSettings, dict[str, Any]]:
    """Derive retrieval parameters from CorpusProfile metrics via explicit rules.

    Heuristics:
    - chunk_size: scales with average section length; stays compact (400) for
      predominantly tabular corpora because CSV rows are chunked per row.
    - overlap: 15% of chunk_size (10% for high-tabular corpora), clamped to [40, 180].
    - top_k: scales with document count and total chunk volume, plus +1 when
      many must-answer topics exist (clamped to [3, 10]).
    - hybrid_alpha: starts at 0.50; shifts toward BM25 (-0.15) when tabular share
      is >= 0.40 or heading density is high (exact IDs/codes), shifts toward
      dense (+0.10) for long-prose non-tabular corpora or multilingual text.
    - rerank: enabled when doc_count >= 4 or top_k >= 5.
    - query_rewrite: enabled for multi-doc mixed-format corpora (doc_count >= 6
      or mixed tabular + prose with >= 4 docs).
    - min_score: soft normalized score floor (0.25 on [0, 1] scale) for larger
      corpora with >= 5 docs and explicit refusal topics; None for small corpora.
    """
    doc_count = max(1, profile.loaded_docs)
    total_chars = sum(doc.char_count for doc in profile.documents)
    avg_doc_chars = total_chars / doc_count
    total_sections = sum(max(1, len(doc.headings)) for doc in profile.documents)
    avg_section_len = total_chars / max(1, total_sections)

    tabular_docs = sum(
        1
        for doc in profile.documents
        if doc.doc_type in {"csv", "xlsx"} or doc.csv_profile is not None
    )
    tabular_share = tabular_docs / doc_count
    language_profile = _estimate_language_profile(profile, requirements.brief)

    # 1. chunk_size
    if tabular_share >= 0.7:
        chunk_size = 400
    elif avg_section_len <= 350:
        chunk_size = 350
    elif avg_section_len <= 950:
        rounded = int(round((avg_section_len * 0.9) / 50.0) * 50)
        chunk_size = max(350, min(800, rounded))
    else:
        chunk_size = 1000 if total_chars > 25000 else 850

    # 2. overlap
    overlap_ratio = 0.10 if tabular_share >= 0.5 else 0.15
    raw_overlap = int(round((chunk_size * overlap_ratio) / 10.0) * 10)
    overlap = max(40, min(180, raw_overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    # 3. top_k
    if profile.loaded_docs <= 2 and avg_doc_chars < 1500:
        base_k = 3
    elif profile.loaded_docs <= 5:
        base_k = 4 if profile.total_chunks <= 30 else 5
    elif profile.loaded_docs <= 12:
        base_k = 6
    else:
        base_k = 8

    if len(requirements.must_answer_topics) >= 5:
        base_k += 1
    top_k = max(3, min(10, base_k))

    # 4. hybrid_alpha
    alpha = 0.50
    if tabular_share >= 0.40:
        alpha -= 0.15
    elif avg_section_len > 800 and tabular_share == 0.0:
        alpha += 0.10
    if language_profile == "multilingual":
        alpha += 0.10
    hybrid_alpha = round(max(0.25, min(0.75, alpha)), 2)

    # 5. rerank & query_rewrite & min_score
    rerank = profile.loaded_docs >= 4 or top_k >= 5
    mixed_formats = 0.0 < tabular_share < 1.0
    query_rewrite = profile.loaded_docs >= 6 or (mixed_formats and profile.loaded_docs >= 4)
    min_score = (
        0.25
        if (profile.loaded_docs >= 5 and len(requirements.must_refuse_topics) >= 2)
        else None
    )

    settings = AgentRetrievalSettings(
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        rerank=rerank,
        query_rewrite=query_rewrite,
        min_score=min_score,
    )
    rationale = {
        "inputs": {
            "loaded_docs": profile.loaded_docs,
            "total_chunks": profile.total_chunks,
            "total_chars": total_chars,
            "avg_doc_chars": round(avg_doc_chars, 1),
            "avg_section_len": round(avg_section_len, 1),
            "tabular_share": round(tabular_share, 3),
            "language_profile": language_profile,
            "must_answer_count": len(requirements.must_answer_topics),
            "must_refuse_count": len(requirements.must_refuse_topics),
        },
        "derived": settings.model_dump(),
    }
    return settings, rationale
