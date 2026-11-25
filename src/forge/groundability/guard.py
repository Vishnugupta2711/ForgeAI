"""Evidence-backed Groundability Guard with strict quote substring verification."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from forge.config import load_prompt
from forge.groundability.models import (
    AtomicTopic,
    EvidenceQuote,
    GroundabilityReport,
    GroundabilityStatus,
    StrictGroundabilityError,
    TopicGroundability,
)
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.retrieval.hybrid import HybridIndex, HybridRetriever, tokenize
from forge.retrieval.models import RetrievalConfig, SearchHit

if TYPE_CHECKING:
    from forge.intake.models import RefusalTopic, RequirementsDocument

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "an",
        "and",
        "answer",
        "any",
        "are",
        "as",
        "assistant",
        "at",
        "be",
        "by",
        "can",
        "do",
        "docs",
        "document",
        "documents",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "question",
        "questions",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "user",
        "users",
        "using",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    }
)

_MIN_VERIFIED_QUOTE_CHARS = 4


class _DecomposedTopicList(BaseModel):
    topics: list[AtomicTopic] = Field(default_factory=list)


class _LLMQuoteItem(BaseModel):
    chunk_id: str = ""
    quote: str = ""


class _LLMGroundabilityVerdict(BaseModel):
    status: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    quotes: list[_LLMQuoteItem] = Field(default_factory=list)
    reasoning: str = ""
    gap_note: str | None = None


def _content_terms(text: str) -> list[str]:
    terms = [tok for tok in tokenize(text) if tok not in _STOPWORDS and len(tok) > 1]
    seen: set[str] = set()
    unique: list[str] = []
    for tok in terms:
        if tok not in seen:
            seen.add(tok)
            unique.append(tok)
    return unique


def _compute_coverage(
    topic: AtomicTopic, hits: list[SearchHit]
) -> tuple[float, float, float]:
    """Return (coverage_score, key_term_overlap, max_retrieval_score)."""
    if not hits:
        return 0.0, 0.0, 0.0

    topic_terms = _content_terms(f"{topic.title} {topic.description}")
    corpus_tokens: set[str] = set()
    for hit in hits:
        corpus_tokens.update(tokenize(f"{hit.chunk.section_heading or ''} {hit.chunk.text}"))

    if topic_terms:
        matched = sum(1 for t in topic_terms if t in corpus_tokens)
        key_term_overlap = round(matched / len(topic_terms), 4)
    else:
        key_term_overlap = 0.0

    max_rrf = max(h.score for h in hits)
    # In RRF with rrf_k=60, rank-1 single-side score is ~1/61 = 0.01639
    retrieval_strength = min(1.0, max_rrf / (1.0 / 61.0))
    coverage = round(0.65 * key_term_overlap + 0.35 * retrieval_strength, 4)
    return coverage, key_term_overlap, round(max_rrf, 6)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _verify_quote_against_hits(
    quote_text: str,
    cited_chunk_id: str,
    hits: list[SearchHit],
) -> tuple[bool, str, str]:
    """Verify quote is an actual substring of a retrieved chunk.

    Returns (verified, resolved_doc_id, resolved_chunk_id).
    """
    cleaned = quote_text.strip()
    if len(cleaned) < _MIN_VERIFIED_QUOTE_CHARS:
        return False, "", cited_chunk_id

    hit_map = {h.chunk.chunk_id: h.chunk for h in hits}

    # Check the cited chunk first, then fall back to any retrieved chunk
    candidates = []
    if cited_chunk_id in hit_map:
        candidates.append(hit_map[cited_chunk_id])
    for hit in hits:
        if hit.chunk.chunk_id != cited_chunk_id:
            candidates.append(hit.chunk)

    norm_quote = _normalize_ws(cleaned)
    for chunk in candidates:
        if cleaned in chunk.text or norm_quote in _normalize_ws(chunk.text):
            return True, chunk.doc_id, chunk.chunk_id

    return False, "", cited_chunk_id


def _format_passages(hits: list[SearchHit]) -> str:
    if not hits:
        return "(No matching passages retrieved.)"
    parts: list[str] = []
    for hit in hits:
        c = hit.chunk
        header = f"[chunk_id={c.chunk_id} | doc_id={c.doc_id} | score={hit.score:.4f}]"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


class GroundabilityGuard:
    """Decompose requirements into atomic topics and verify document grounding."""

    def __init__(
        self,
        index: HybridIndex,
        llm_client: LLMClient,
        *,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.top_k = top_k
        self.alpha = alpha

    def decompose_topics(self, requirements: RequirementsDocument) -> list[AtomicTopic]:
        """Decompose brief, Q&A, and must-answer topics into atomic verifiable topics."""
        qa_block = "\n".join(
            f"- {qa.question_id}: {qa.question} -> {qa.answer}"
            for qa in requirements.clarifying_qa
        ) or "(none)"
        topics_block = "\n".join(
            f"- {t.id}: {t.title} — {t.description}"
            for t in requirements.must_answer_topics
        ) or "(none)"

        prompt = load_prompt("groundability_decompose.v1.md").format(
            brief=requirements.brief,
            qa_block=qa_block,
            topics_block=topics_block,
        )

        topics: list[AtomicTopic] = []
        try:
            parsed = self.llm_client.json(
                prompt,
                schema=_DecomposedTopicList,
                role="judge",
                max_repairs=2,
            )
            topics = list(parsed.topics)
        except Exception:
            topics = []

        # Deterministic safeguard: ensure every explicit REQ-xxx is covered
        covered_req_ids = {t.source_req_id for t in topics if t.source_req_id}
        next_idx = len(topics) + 1
        for req in requirements.must_answer_topics:
            if req.id not in covered_req_ids:
                topics.append(
                    AtomicTopic(
                        topic_id=f"TOP-{next_idx:03d}",
                        title=req.title,
                        description=req.description,
                        source_req_id=req.id,
                        is_implicit=False,
                    )
                )
                next_idx += 1

        return topics

    def _retrieve_for_topic(self, topic: AtomicTopic) -> list[SearchHit]:
        query = f"{topic.title} {topic.description}".strip()
        use_dense = self.index.embeddings is not None and self.alpha > 0.0
        retriever = HybridRetriever(
            self.index,
            llm_client=self.llm_client if use_dense else None,
            config=RetrievalConfig(
                alpha=self.alpha if use_dense else 0.0,
                top_k=self.top_k,
            ),
        )
        try:
            return retriever.search(query)
        except LLMProviderError:
            return HybridRetriever(
                self.index,
                llm_client=None,
                config=RetrievalConfig(alpha=0.0, top_k=self.top_k),
            ).search(query)

    def verify_topic(self, topic: AtomicTopic) -> TopicGroundability:
        """Retrieve evidence, compute deterministic coverage, and verify LLM quotes."""
        hits = self._retrieve_for_topic(topic)
        coverage, key_overlap, max_rrf = _compute_coverage(topic, hits)
        chunk_ids = [h.chunk.chunk_id for h in hits]

        if not hits or key_overlap == 0.0:
            reason = (
                "No relevant document passages retrieved."
                if not hits
                else "Zero key-term overlap between topic and retrieved passages."
            )
            return TopicGroundability(
                topic_id=topic.topic_id,
                title=topic.title,
                description=topic.description,
                source_req_id=topic.source_req_id,
                status="UNSUPPORTED",
                coverage_score=coverage,
                key_term_overlap=key_overlap,
                max_retrieval_score=max_rrf,
                retrieved_chunk_ids=chunk_ids,
                quotes=[],
                reasoning=reason,
                gap_note=f"Topic '{topic.title}' has no supporting evidence in the corpus.",
            )

        prompt = load_prompt("groundability_verify.v1.md").format(
            topic_id=topic.topic_id,
            title=topic.title,
            description=topic.description,
            passages_block=_format_passages(hits),
        )

        try:
            verdict = self.llm_client.json(
                prompt,
                schema=_LLMGroundabilityVerdict,
                role="judge",
                max_repairs=2,
            )
        except Exception as exc:
            return TopicGroundability(
                topic_id=topic.topic_id,
                title=topic.title,
                description=topic.description,
                source_req_id=topic.source_req_id,
                status="UNSUPPORTED",
                coverage_score=coverage,
                key_term_overlap=key_overlap,
                max_retrieval_score=max_rrf,
                retrieved_chunk_ids=chunk_ids,
                quotes=[],
                reasoning=f"Groundability verification LLM call failed: {exc}",
                gap_note=f"Could not verify evidence for '{topic.title}'.",
            )

        verified_quotes: list[EvidenceQuote] = []
        for q_item in verdict.quotes:
            ok, doc_id, resolved_chunk_id = _verify_quote_against_hits(
                q_item.quote, q_item.chunk_id, hits
            )
            verified_quotes.append(
                EvidenceQuote(
                    doc_id=doc_id,
                    chunk_id=resolved_chunk_id,
                    quote=q_item.quote,
                    verified_substring=ok,
                )
            )

        has_verified_quote = any(q.verified_substring for q in verified_quotes)
        status: GroundabilityStatus = verdict.status
        gap_note = verdict.gap_note
        quote_failed = False

        if status in ("SUPPORTED", "PARTIAL") and not has_verified_quote:
            status = "UNSUPPORTED"
            quote_failed = True
            note = (
                "Returned quote(s) did not match any retrieved chunk substring; "
                "downgraded to UNSUPPORTED."
            )
            gap_note = f"{gap_note} ({note})" if gap_note else note

        if status == "PARTIAL" and not gap_note:
            gap_note = f"Topic '{topic.title}' is only partially covered by the corpus."
        if status == "UNSUPPORTED" and not gap_note:
            gap_note = f"Topic '{topic.title}' is not supported by the corpus."

        return TopicGroundability(
            topic_id=topic.topic_id,
            title=topic.title,
            description=topic.description,
            source_req_id=topic.source_req_id,
            status=status,
            coverage_score=coverage,
            key_term_overlap=key_overlap,
            max_retrieval_score=max_rrf,
            retrieved_chunk_ids=chunk_ids,
            quotes=verified_quotes,
            reasoning=verdict.reasoning,
            gap_note=gap_note,
            quote_verification_failed=quote_failed,
        )

    def evaluate(
        self,
        requirements: RequirementsDocument,
        *,
        strict: bool = False,
    ) -> tuple[RequirementsDocument, GroundabilityReport]:
        """Evaluate all decomposed topics and update requirements with gaps and scope."""
        from forge.intake.models import RefusalTopic

        topics = self.decompose_topics(requirements)
        results = [self.verify_topic(t) for t in topics]

        supported = sum(1 for r in results if r.status == "SUPPORTED")
        partial = sum(1 for r in results if r.status == "PARTIAL")
        unsupported = sum(1 for r in results if r.status == "UNSUPPORTED")
        has_gaps = (partial + unsupported) > 0

        flagged_gaps: list[str] = []
        unsupported_req_ids: set[str] = set()
        partial_notes: list[str] = []

        updated_refusals: list[RefusalTopic] = list(requirements.must_refuse_topics)
        existing_refusal_titles = {r.title.lower() for r in updated_refusals}

        req_status_by_id: dict[str, GroundabilityStatus] = {}
        for item in results:
            if item.source_req_id:
                req_status_by_id[item.source_req_id] = item.status
            if item.status == "UNSUPPORTED":
                msg = f"[UNSUPPORTED] {item.title}: {item.gap_note or 'Missing from corpus.'}"
                flagged_gaps.append(msg)
                if item.source_req_id:
                    unsupported_req_ids.add(item.source_req_id)
                if item.title.lower() not in existing_refusal_titles:
                    updated_refusals.append(
                        RefusalTopic(
                            id=f"GAP-{item.topic_id}",
                            title=item.title,
                            reason=item.gap_note or "Unsupported by provided documents.",
                            from_gap=True,
                        )
                    )
                    existing_refusal_titles.add(item.title.lower())
            elif item.status == "PARTIAL":
                msg = f"[PARTIAL] {item.title}: {item.gap_note or 'Only partially covered.'}"
                flagged_gaps.append(msg)
                partial_notes.append(f"{item.title} ({item.gap_note or 'partial coverage'})")

        updated_must_answer = []
        for req in requirements.must_answer_topics:
            if req.id in unsupported_req_ids:
                continue
            status_val = req_status_by_id.get(req.id)
            updated_must_answer.append(
                req.model_copy(update={"groundability_status": status_val})
            )

        updated_scope = requirements.scope.strip()
        if partial_notes:
            narrowing = (
                " Scope narrowed due to partial corpus evidence: "
                + "; ".join(partial_notes)
                + "."
            )
            if narrowing.strip() not in updated_scope:
                updated_scope = f"{updated_scope}{narrowing}"

        report = GroundabilityReport(
            strict_mode=strict,
            has_gaps=has_gaps,
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            flagged_gaps=flagged_gaps,
            topics=results,
        )

        updated_req = requirements.model_copy(
            update={
                "scope": updated_scope,
                "must_answer_topics": updated_must_answer,
                "must_refuse_topics": updated_refusals,
                "groundability_report": report,
            }
        )

        if strict and has_gaps:
            raise StrictGroundabilityError(
                "Groundability guard aborted in --strict mode due to gaps:\n"
                + "\n".join(f"- {g}" for g in flagged_gaps),
                report=report,
                requirements=updated_req,
            )

        return updated_req, report
