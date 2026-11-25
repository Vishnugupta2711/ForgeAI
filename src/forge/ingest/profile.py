"""Compact document and corpus profiling."""

from __future__ import annotations

from pathlib import Path

from forge.ingest.chunking import chunk_documents, extract_headings
from forge.ingest.loaders import (
    DEFAULT_MAX_DOC_CHARS,
    DEFAULT_MAX_FILE_BYTES,
    load_documents,
)
from forge.ingest.models import (
    Chunk,
    CorpusProfile,
    CsvSchemaProfile,
    Document,
    DocumentProfile,
    IngestResult,
)


def profile_document(doc: Document, chunks: list[Chunk]) -> DocumentProfile:
    """Build a compact DocumentProfile for a single Document and its chunks."""
    doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]
    headings = extract_headings(doc.text)
    if not headings and doc.type == "csv":
        headings = [f"Row {i}" for i in range(1, min(len(doc_chunks), 5) + 1)]

    csv_profile = None
    if doc.type == "csv" and "csv_profile" in doc.metadata:
        csv_profile = CsvSchemaProfile.model_validate(doc.metadata["csv_profile"])

    lines = doc.text.splitlines()
    size_bytes = int(doc.metadata.get("size_bytes") or len(doc.text.encode("utf-8")))
    page_count = len(doc.pages) if doc.pages else 1

    summary_stats = {
        "word_count": len(doc.text.split()),
        "avg_chunk_chars": (
            round(sum(len(c.text) for c in doc_chunks) / len(doc_chunks), 1)
            if doc_chunks
            else 0.0
        ),
    }
    if csv_profile is not None:
        summary_stats["csv_rows"] = csv_profile.row_count
        summary_stats["csv_columns"] = len(csv_profile.columns)

    return DocumentProfile(
        doc_id=doc.doc_id,
        name=Path(doc.path).name,
        path=doc.path,
        doc_type=doc.type,
        size_bytes=size_bytes,
        char_count=len(doc.text),
        line_count=len(lines),
        page_count=page_count,
        chunk_count=len(doc_chunks),
        headings=headings,
        csv_profile=csv_profile,
        summary_stats=summary_stats,
    )


def build_profile(ingest_result: IngestResult, chunks: list[Chunk]) -> CorpusProfile:
    """Build a CorpusProfile summarizing all loaded documents and warnings."""
    doc_profiles = [profile_document(doc, chunks) for doc in ingest_result.documents]
    return CorpusProfile(
        root_path=ingest_result.root_path,
        total_files_scanned=ingest_result.total_files_scanned,
        loaded_docs=len(ingest_result.documents),
        skipped_files=ingest_result.skipped_files,
        total_chunks=len(chunks),
        documents=doc_profiles,
        warnings=list(ingest_result.warnings),
    )


def ingest_folder(
    folder: Path | str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> tuple[IngestResult, list[Chunk], CorpusProfile]:
    """Load, chunk, and profile a document folder in one call."""
    ingest_result = load_documents(
        folder, max_file_bytes=max_file_bytes, max_doc_chars=max_doc_chars
    )
    chunks = chunk_documents(
        ingest_result.documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    corpus_profile = build_profile(ingest_result, chunks)
    return ingest_result, chunks, corpus_profile
