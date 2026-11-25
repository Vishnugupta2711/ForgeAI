"""Section-aware recursive chunker and CSV row chunking."""

from __future__ import annotations

import re

from forge.ingest.models import Chunk, Document

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+([A-Z][^\n]{2,80})$")
_ALLCAPS_HEADING_RE = re.compile(r"^([A-Z][A-Z0-9\s\-/&]{3,60})$")
_UNDERLINE_RE = re.compile(r"^[=-]{3,}\s*$")

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _detect_heading_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    md_match = _MD_HEADING_RE.match(stripped)
    if md_match:
        return md_match.group(2).strip()
    num_match = _NUMBERED_HEADING_RE.match(stripped)
    if num_match and not stripped.endswith("."):
        return f"{num_match.group(1)}. {num_match.group(2).strip()}"
    if _ALLCAPS_HEADING_RE.match(stripped) and not stripped.endswith("."):
        return stripped
    return None


def extract_headings(text: str) -> list[str]:
    """Extract ordered unique section headings from document text."""
    lines = text.splitlines()
    headings: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 1 < len(lines)
            and line.strip()
            and _UNDERLINE_RE.match(lines[i + 1].strip())
            and len(line.strip()) <= 80
        ):
            heading = line.strip()
            if heading not in seen:
                headings.append(heading)
                seen.add(heading)
            i += 2
            continue
        detected = _detect_heading_line(line)
        if detected and detected not in seen:
            headings.append(detected)
            seen.add(detected)
        i += 1
    return headings


def split_into_sections(
    text: str,
    initial_heading: str | None = None,
) -> list[tuple[str | None, str]]:
    """Split text into (section_heading, body_text) pairs."""
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    current_heading = initial_heading
    buffer: list[str] = []

    def _flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))
        buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 1 < len(lines)
            and line.strip()
            and _UNDERLINE_RE.match(lines[i + 1].strip())
            and len(line.strip()) <= 80
        ):
            _flush()
            current_heading = line.strip()
            buffer.append(line.strip())
            i += 2
            continue

        detected = _detect_heading_line(line)
        if detected is not None:
            _flush()
            current_heading = detected
            buffer.append(line.strip())
            i += 1
            continue

        buffer.append(line)
        i += 1

    _flush()
    return sections


def _split_atomic(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - overlap)
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        piece = text[pos : pos + chunk_size].strip()
        if piece:
            out.append(piece)
        if pos + chunk_size >= n:
            break
        pos += step
    return out


def _recursive_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        return _split_atomic(text, chunk_size, overlap=0)

    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        return _split_atomic(text, chunk_size, overlap=0)

    if sep not in text:
        return _recursive_pieces(text, chunk_size, rest)

    parts = text.split(sep)
    pieces: list[str] = []
    current = ""
    for idx, part in enumerate(parts):
        segment = part + (sep if sep == ". " and idx < len(parts) - 1 else "")
        candidate = segment if not current else f"{current}{sep if sep != '. ' else ' '}{segment}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                pieces.append(current.strip())
            if len(segment) > chunk_size:
                pieces.extend(_recursive_pieces(segment, chunk_size, rest))
                current = ""
            else:
                current = segment
    if current.strip():
        pieces.append(current.strip())
    return pieces


def recursive_split(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Recursively split text into bounded chunks with character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    base_pieces = _recursive_pieces(cleaned, chunk_size, _RECURSIVE_SEPARATORS)
    if overlap == 0 or len(base_pieces) <= 1:
        return base_pieces

    overlapped: list[str] = [base_pieces[0]]
    for piece in base_pieces[1:]:
        prev_tail = overlapped[-1][-overlap:].lstrip()
        if prev_tail and not piece.startswith(prev_tail):
            combined = f"{prev_tail} {piece}".strip()
            if len(combined) <= chunk_size:
                overlapped.append(combined)
                continue
            budget = max(0, chunk_size - len(piece) - 1)
            if budget > 0:
                trimmed_tail = prev_tail[-budget:].lstrip()
                combined_trimmed = f"{trimmed_tail} {piece}".strip()
                if len(combined_trimmed) <= chunk_size:
                    overlapped.append(combined_trimmed)
                    continue
        overlapped.append(piece[:chunk_size].strip())

    return [c for c in overlapped if c]


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Chunk a Document into retrievable Chunks with section and page metadata."""
    if document.type == "csv":
        rows: list[dict[str, str]] = list(document.metadata.get("rows") or [])
        csv_profile = document.metadata.get("csv_profile") or {}
        columns: list[str] = list(csv_profile.get("columns") or [])
        chunks: list[Chunk] = []
        for idx, row in enumerate(rows, start=1):
            ordered_cols = columns or list(row.keys())
            row_text = "\n".join(f"{col}: {row.get(col, '')}" for col in ordered_cols)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::row-{idx}",
                    doc_id=document.doc_id,
                    text=row_text,
                    page=None,
                    section_heading=f"Row {idx}",
                    metadata={
                        "doc_type": "csv",
                        "source_path": document.path,
                        "row_index": idx,
                    },
                )
            )
        return chunks

    chunks = []
    counter = 0

    if document.pages:
        active_heading: str | None = None
        for page in document.pages:
            sections = split_into_sections(page.text, initial_heading=active_heading)
            for heading, body in sections:
                active_heading = heading
                for piece in recursive_split(body, chunk_size=chunk_size, overlap=overlap):
                    counter += 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.doc_id}::c{counter}",
                            doc_id=document.doc_id,
                            text=piece,
                            page=page.page_number,
                            section_heading=heading,
                            metadata={
                                "doc_type": document.type,
                                "source_path": document.path,
                            },
                        )
                    )
        return chunks

    for heading, body in split_into_sections(document.text):
        for piece in recursive_split(body, chunk_size=chunk_size, overlap=overlap):
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::c{counter}",
                    doc_id=document.doc_id,
                    text=piece,
                    page=None,
                    section_heading=heading,
                    metadata={
                        "doc_type": document.type,
                        "source_path": document.path,
                    },
                )
            )
    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Chunk a list of Documents into a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
