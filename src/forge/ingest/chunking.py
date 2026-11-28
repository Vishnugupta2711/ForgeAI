"""Section-aware recursive chunker and CSV row chunking."""

from __future__ import annotations

import re

from forge.ingest.models import Chunk, Document

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+([A-Z][^\n]{2,80})$")
_ALLCAPS_HEADING_RE = re.compile(r"^([A-Z][A-Z0-9\s\-/&]{3,60})$")
_UNDERLINE_RE = re.compile(r"^[=-]{3,}\s*$")

_RECURSIVE_SEPARATORS = ["\n\n", ". ", "! ", "? ", "\n", " ", ""]
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _snap_end_to_word(text: str, start: int, limit: int) -> int:
    """Snap slice end <= start + limit backward to whitespace so words stay intact."""
    end = min(len(text), start + limit)
    if end >= len(text):
        return len(text)
    if not text[end].isspace() and not text[end - 1].isspace():
        ws = max(
            text.rfind(" ", start, end),
            text.rfind("\n", start, end),
            text.rfind("\t", start, end),
        )
        if ws > start:
            return ws
    return end


def _snap_start_to_word(text: str, pos: int) -> int:
    """Advance pos to a clean word start if it landed inside a word."""
    n = len(text)
    if pos <= 0 or pos >= n:
        return max(0, min(pos, n))
    if not text[pos - 1].isspace() and not text[pos].isspace():
        next_ws = [
            i for i in (text.find(" ", pos), text.find("\n", pos), text.find("\t", pos)) if i != -1
        ]
        if next_ws:
            pos = min(next_ws)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def _snap_overlap_tail(prev_text: str, overlap: int, max_tail_len: int | None = None) -> str:
    """Extract an overlap tail aligned to a sentence or word boundary (never mid-word)."""
    if overlap <= 0 or not prev_text:
        return ""
    if len(prev_text) <= overlap and (max_tail_len is None or len(prev_text) <= max_tail_len):
        return prev_text.strip()

    limit = min(overlap, max_tail_len) if max_tail_len is not None else overlap
    if limit <= 0:
        return ""

    raw_start = max(0, len(prev_text) - limit)
    # Prefer snapping slightly forward to a sentence start when it preserves a meaningful tail
    if raw_start > 0:
        sent_match = None
        for m in _SENTENCE_SPLIT_RE.finditer(prev_text, raw_start):
            if len(prev_text) - m.end() >= max(8, limit // 3):
                sent_match = m
                break
        if sent_match is not None:
            cand = prev_text[sent_match.end() :].strip()
            if max_tail_len is None or len(cand) <= max_tail_len:
                return cand

        # If raw_start lands mid-word, try backing up to word start if budget allows
        if not prev_text[raw_start - 1].isspace() and not prev_text[raw_start].isspace():
            ws = max(
                prev_text.rfind(" ", 0, raw_start),
                prev_text.rfind("\n", 0, raw_start),
                prev_text.rfind("\t", 0, raw_start),
            )
            word_start = ws + 1 if ws != -1 else 0
            cand_back = prev_text[word_start:].strip()
            if max_tail_len is None or len(cand_back) <= max_tail_len:
                return cand_back
            # Otherwise advance past the partial word
            fwd = _snap_start_to_word(prev_text, raw_start)
            return prev_text[fwd:].strip()

    return prev_text[raw_start:].strip()


def _split_atomic(text: str, chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        end = _snap_end_to_word(text, pos, chunk_size)
        piece = text[pos:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        raw_next = max(pos + 1, end - overlap)
        pos = _snap_start_to_word(text, raw_next)
        if pos <= end - chunk_size:
            pos = end
    return out


def _split_sentence_spans(text: str) -> list[str]:
    """Split text on sentence endings without breaking numbered heading prefixes."""
    boundaries: list[int] = [0]
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        before = text[boundaries[-1] : m.start() + 1]
        # Avoid breaking right after a standalone list/section number like '1.' or '12.'
        tail_line = before.splitlines()[-1].strip() if before.splitlines() else ""
        if re.fullmatch(r"\d+(?:\.\d+)*\.", tail_line):
            continue
        boundaries.append(m.end())
    boundaries.append(len(text))
    return [
        text[boundaries[i] : boundaries[i + 1]].strip()
        for i in range(len(boundaries) - 1)
        if text[boundaries[i] : boundaries[i + 1]].strip()
    ]


def _split_sentences(text: str, chunk_size: int, rest_separators: list[str]) -> list[str]:
    """Split text along sentence boundaries (. ! ? followed by whitespace/newline)."""
    parts = _split_sentence_spans(text)
    if len(parts) <= 1:
        return _recursive_pieces(text, chunk_size, rest_separators)
    pieces: list[str] = []
    current = ""
    for seg in parts:
        candidate = seg if not current else f"{current} {seg}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(seg) > chunk_size:
                pieces.extend(_recursive_pieces(seg, chunk_size, rest_separators))
                current = ""
            else:
                current = seg
    if current:
        pieces.append(current)
    return pieces


def _recursive_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        return _split_atomic(text, chunk_size, overlap=0)

    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        return _split_atomic(text, chunk_size, overlap=0)

    # Check sentence boundaries (including PDF newline wraps after punctuation) before raw "\n"
    if sep == ". ":
        return _split_sentences(text, chunk_size, [s for s in rest if s not in ("! ", "? ")])

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
    """Recursively split text into bounded chunks with word/sentence-aligned overlap."""
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
        budget = max(0, chunk_size - len(piece) - 1)
        prev_tail = _snap_overlap_tail(overlapped[-1], overlap, max_tail_len=budget)
        if prev_tail and not piece.startswith(prev_tail):
            combined = f"{prev_tail} {piece}".strip()
            if len(combined) <= chunk_size:
                overlapped.append(combined)
                continue
        if len(piece) > chunk_size:
            end = _snap_end_to_word(piece, 0, chunk_size)
            overlapped.append(piece[:end].strip())
        else:
            overlapped.append(piece.strip())

    return [c for c in overlapped if c]


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
