"""Deterministic and LLM quality gates for candidate evaluation cases."""

from __future__ import annotations

import difflib
import random
import re

import numpy as np
from pydantic import BaseModel

from forge.config import load_prompt
from forge.evals.models import EvalCase, RejectedCaseLog
from forge.ingest.models import Chunk
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.llm.providers import deterministic_hash_embedding
from forge.retrieval.hybrid import HybridIndex, HybridRetriever
from forge.retrieval.models import RetrievalConfig

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


class _SolvabilityVerdict(BaseModel):
    solvable: bool
    reason: str = ""


class _OosAnswerabilityVerdict(BaseModel):
    answerable_from_docs: bool
    reason: str = ""


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for resilient substring/fuzzy checks."""
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def verify_key_fact(fact: str, source_text: str, *, threshold: float = 0.80) -> bool:
    """Verify that a key fact is grounded in source_text via exact or fuzzy match."""
    norm_fact = _PUNCT_EDGE_RE.sub("", normalize_text(fact))
    norm_source = normalize_text(source_text)
    if len(norm_fact) < 3 or not norm_source:
        return False

    if norm_fact in norm_source:
        return True

    # Fuzzy sliding-window check over character spans of similar length
    fact_len = len(norm_fact)
    if len(norm_source) <= fact_len + 10:
        return difflib.SequenceMatcher(None, norm_fact, norm_source).ratio() >= threshold

    step = max(1, fact_len // 6)
    win = int(fact_len * 1.15)
    best = 0.0
    for start in range(0, len(norm_source) - max(1, fact_len // 2) + 1, step):
        snippet = norm_source[start : start + win]
        score = difflib.SequenceMatcher(None, norm_fact, snippet).ratio()
        if score >= threshold:
            return True
        if score > best:
            best = score
    return best >= threshold


def verify_case_key_facts(
    case: EvalCase,
    chunk_map: dict[str, Chunk],
    *,
    require_multi_doc: bool = False,
    corpus_doc_count: int = 2,
    threshold: float = 0.80,
) -> tuple[EvalCase | None, str | None]:
    """Gate 1: verify each key_fact against gold source chunks; drop bad facts/cases."""
    if case.expected_behavior == "refuse":
        return case, None

    source_chunks: list[Chunk] = []
    for ref in case.expected_sources:
        chunk = chunk_map.get(ref.chunk_id)
        if chunk is not None:
            source_chunks.append(chunk)

    if not source_chunks:
        if case.expected_behavior == "clarify":
            return case, None
        return None, "missing_source_chunks"

    if require_multi_doc:
        distinct_chunks = {c.chunk_id for c in source_chunks}
        distinct_docs = {c.doc_id for c in source_chunks}
        if len(distinct_chunks) < 2:
            return None, "multi_doc_needs_two_chunks"
        if corpus_doc_count >= 2 and len(distinct_docs) < 2:
            return None, "multi_doc_needs_two_documents"

    combined_source = "\n".join(c.text for c in source_chunks)
    verified_facts = [
        fact
        for fact in case.key_facts
        if verify_key_fact(fact, combined_source, threshold=threshold)
    ]

    if case.expected_behavior == "answer" and not verified_facts:
        return None, "no_verified_key_facts"

    if case.expected_behavior == "clarify" and case.key_facts and not verified_facts:
        # Allow clarify cases without key_facts, but strip fabricated ones
        verified_facts = []

    return case.model_copy(update={"key_facts": verified_facts}), None


def check_solvability(
    case: EvalCase,
    chunk_map: dict[str, Chunk],
    llm_client: LLMClient,
    *,
    model: str,
    seed: int,
) -> tuple[bool, str]:
    """Gate 2: verify the question is solvable using ONLY its gold source chunk(s)."""
    if case.expected_behavior != "answer":
        return True, "non-answer case skips solvability gate"

    passages: list[str] = []
    for ref in case.expected_sources:
        ch = chunk_map.get(ref.chunk_id)
        if ch is not None:
            passages.append(f"[{ch.doc_id} | {ch.chunk_id}]\n{ch.text}")

    if not passages:
        return False, "no gold source passages found"

    prompt = load_prompt("evalgen_solvability_check.v1.md").format(
        question=case.question,
        passages_block="\n\n".join(passages),
    )
    try:
        verdict = llm_client.json(
            prompt,
            schema=_SolvabilityVerdict,
            model=model,
            temperature=0.0,
            seed=seed,
            role="eval_writer",
            max_repairs=2,
        )
        return verdict.solvable, verdict.reason
    except Exception as exc:
        return False, f"solvability_check_error: {exc}"


def check_oos_unanswerable(
    case: EvalCase,
    index: HybridIndex,
    llm_client: LLMClient,
    *,
    model: str,
    seed: int,
    top_k: int = 4,
) -> tuple[bool, str]:
    """Gate 3: verify an out-of-scope question is NOT actually answerable from corpus."""
    if case.expected_behavior != "refuse":
        return True, "not an out-of-scope refusal case"

    use_dense = index.embeddings is not None
    retriever = HybridRetriever(
        index,
        llm_client=llm_client if use_dense else None,
        config=RetrievalConfig(alpha=0.5 if use_dense else 0.0, top_k=top_k),
    )
    try:
        hits = retriever.search(case.question)
    except LLMProviderError:
        hits = HybridRetriever(
            index,
            llm_client=None,
            config=RetrievalConfig(alpha=0.0, top_k=top_k),
        ).search(case.question)

    if not hits:
        return True, "no corpus hits retrieved"

    passages = "\n\n".join(
        f"[{h.chunk.doc_id} | {h.chunk.chunk_id}]\n{h.chunk.text}" for h in hits
    )
    prompt = load_prompt("evalgen_oos_check.v1.md").format(
        question=case.question,
        passages_block=passages,
    )
    try:
        verdict = llm_client.json(
            prompt,
            schema=_OosAnswerabilityVerdict,
            model=model,
            temperature=0.0,
            seed=seed,
            role="eval_writer",
            max_repairs=2,
        )
        if verdict.answerable_from_docs:
            return False, f"answerable_from_docs: {verdict.reason}"
        return True, verdict.reason
    except Exception as exc:
        return False, f"oos_check_error: {exc}"


def dedupe_by_embedding(
    cases: list[EvalCase],
    llm_client: LLMClient,
    *,
    similarity_threshold: float = 0.92,
) -> tuple[list[EvalCase], list[RejectedCaseLog]]:
    """Gate 4: deduplicate near-identical questions by normalized text & embedding cosine."""
    if not cases:
        return [], []

    questions = [c.question for c in cases]
    try:
        vectors = llm_client.embed(questions)
    except Exception:
        vectors = [deterministic_hash_embedding(q, dim=64) for q in questions]

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = arr / norms

    kept: list[EvalCase] = []
    kept_vecs: list[np.ndarray] = []
    kept_norms: set[str] = set()
    rejected: list[RejectedCaseLog] = []

    for idx, case in enumerate(cases):
        q_norm = normalize_text(case.question)
        if q_norm in kept_norms:
            rejected.append(
                RejectedCaseLog(
                    question=case.question,
                    category=case.category,
                    gate="embedding_dedupe",
                    reason="exact_normalized_duplicate",
                )
            )
            continue

        vec = normed[idx]
        is_dup = False
        if kept_vecs:
            sims = np.asarray(kept_vecs, dtype=np.float32) @ vec
            if float(np.max(sims)) >= similarity_threshold:
                is_dup = True

        if is_dup:
            rejected.append(
                RejectedCaseLog(
                    question=case.question,
                    category=case.category,
                    gate="embedding_dedupe",
                    reason=f"cosine_similarity >= {similarity_threshold}",
                )
            )
            continue

        kept.append(case)
        kept_vecs.append(vec)
        kept_norms.add(q_norm)

    return kept, rejected


def stratified_split(
    cases: list[EvalCase],
    *,
    seed: int = 42,
    held_out_fraction: float = 1.0 / 3.0,
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Gate 5: deterministic stratified split into dev (~2/3) and held_out (~1/3)."""
    rng = random.Random(seed)
    by_category: dict[str, list[EvalCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)

    dev: list[EvalCase] = []
    held_out: list[EvalCase] = []

    for cat in sorted(by_category):
        group = sorted(by_category[cat], key=lambda c: (c.question, c.id))
        rng.shuffle(group)
        if len(group) == 1:
            n_held = 0
        else:
            n_held = max(1, int(round(len(group) * held_out_fraction)))
            if n_held >= len(group):
                n_held = len(group) - 1

        for item in group[:n_held]:
            held_out.append(item.model_copy(update={"split": "held_out"}))
        for item in group[n_held:]:
            dev.append(item.model_copy(update={"split": "dev"}))

    # Re-assign clean sequential IDs deterministically
    numbered_dev = [
        c.model_copy(update={"id": f"DEV-{i:03d}"}) for i, c in enumerate(dev, start=1)
    ]
    numbered_held = [
        c.model_copy(update={"id": f"HELD-{i:03d}"}) for i, c in enumerate(held_out, start=1)
    ]
    return numbered_dev, numbered_held
