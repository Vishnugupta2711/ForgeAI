"""Data models for document ingestion, chunking, and corpus profiling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Text content for a single page of a paginated document."""

    page_number: int
    text: str


class CsvSchemaProfile(BaseModel):
    """Schema and summary metadata extracted from a CSV file."""

    columns: list[str]
    dtypes: dict[str, str]
    sample_rows: list[dict[str, str]]
    row_count: int


class Document(BaseModel):
    """Normalized ingested document."""

    doc_id: str
    path: str
    type: str
    text: str
    pages: list[PageContent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Retrievable document chunk with provenance metadata."""

    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None
    section_heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Output of scanning and loading a folder of documents."""

    root_path: str
    documents: list[Document] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_files_scanned: int = 0
    skipped_files: int = 0


class DocumentProfile(BaseModel):
    """Compact structural summary of a single ingested document."""

    doc_id: str
    name: str
    path: str
    doc_type: str
    size_bytes: int
    char_count: int
    line_count: int
    page_count: int
    chunk_count: int
    headings: list[str] = Field(default_factory=list)
    csv_profile: CsvSchemaProfile | None = None
    summary_stats: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CorpusProfile(BaseModel):
    """Compact corpus profile consumed by downstream factory stages."""

    root_path: str
    total_files_scanned: int
    loaded_docs: int
    skipped_files: int
    total_chunks: int
    documents: list[DocumentProfile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_compact_text(self) -> str:
        """Render a concise plain-text summary suitable for small-model prompts."""
        lines = [
            f"Corpus: {self.root_path}",
            (
                f"Files scanned: {self.total_files_scanned} | "
                f"Loaded: {self.loaded_docs} | "
                f"Skipped: {self.skipped_files} | "
                f"Total chunks: {self.total_chunks}"
            ),
        ]
        for doc in self.documents:
            heading_preview = ", ".join(doc.headings[:6]) if doc.headings else "none"
            line = (
                f"- {doc.doc_id} ({doc.doc_type}, {doc.size_bytes} B, "
                f"{doc.chunk_count} chunks, headings: {heading_preview})"
            )
            if doc.csv_profile is not None:
                cols = ", ".join(doc.csv_profile.columns)
                line += f" | CSV rows={doc.csv_profile.row_count} cols=[{cols}]"
            lines.append(line)
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  * {warning}")
        return "\n".join(lines)
