"""Document loaders for txt, md, pdf, docx, csv, and xlsx."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader

from forge.ingest.models import CsvSchemaProfile, Document, IngestResult, PageContent

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}
_IGNORED_FILENAMES = {".DS_Store", "Thumbs.db"}
DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_MAX_DOC_CHARS = 300_000
DEFAULT_MAX_CSV_ROWS = 5_000


def _read_text_with_fallback(
    path: Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> tuple[str, bool]:
    raw = path.read_bytes()
    truncated_bytes = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated_bytes = True

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace"), truncated_bytes

    if b"\x00" in raw:
        raise ValueError("Binary null bytes detected in text file.")

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), truncated_bytes
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), truncated_bytes


def _infer_dtype(values: list[str]) -> str:
    non_empty = [v.strip() for v in values if v is not None and v.strip() != ""]
    if not non_empty:
        return "str"
    if all(v.lower() in {"true", "false", "yes", "no"} for v in non_empty):
        return "bool"
    try:
        for v in non_empty:
            int(v)
        return "int"
    except ValueError:
        pass
    try:
        for v in non_empty:
            float(v)
        return "float"
    except ValueError:
        pass
    return "str"


def _load_pdf(
    path: Path,
    rel_path: str,
    size_bytes: int,
    *,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> tuple[Document | None, list[str]]:
    warnings: list[str] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return None, [f"Unreadable PDF '{rel_path}': {exc}"]

    pages: list[PageContent] = []
    total_chars = 0
    truncated = False
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            return None, [f"Failed extracting page {idx} of PDF '{rel_path}': {exc}"]
        if total_chars + len(text) > max_doc_chars:
            remaining = max(0, max_doc_chars - total_chars)
            if remaining > 0:
                pages.append(PageContent(page_number=idx, text=text[:remaining]))
            truncated = True
            break
        pages.append(PageContent(page_number=idx, text=text))
        total_chars += len(text)

    combined = "\n\n".join(p.text for p in pages if p.text).strip()
    if not combined:
        return None, [
            f"Scanned or empty PDF (no extractable text) '{rel_path}' "
            f"({len(reader.pages)} pages)."
        ]

    if truncated:
        warnings.append(
            f"Large PDF '{rel_path}' ({size_bytes} B) truncated at {max_doc_chars} characters."
        )

    doc = Document(
        doc_id=rel_path,
        path=str(path),
        type="pdf",
        text=combined,
        pages=pages,
        metadata={
            "size_bytes": size_bytes,
            "page_count": len(pages),
            "truncated": truncated,
        },
    )
    return doc, warnings


def _load_docx(path: Path, rel_path: str, size_bytes: int) -> tuple[Document | None, str | None]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return None, f"Invalid DOCX (missing word/document.xml) '{rel_path}'."
            xml_bytes = zf.read("word/document.xml")
    except Exception as exc:
        return None, f"Unreadable DOCX '{rel_path}': {exc}"

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        return None, f"Malformed XML in DOCX '{rel_path}': {exc}"

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    blocks: list[str] = []

    body = root.find("w:body", ns)
    if body is None:
        return None, f"Empty DOCX body in '{rel_path}'."

    for child in body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            texts = [t.text for t in child.findall(".//w:t", ns) if t.text]
            para = "".join(texts).strip()
            if not para:
                continue
            pstyle = child.find("./w:pPr/w:pStyle", ns)
            style_val = pstyle.attrib.get(f"{{{ns['w']}}}val", "") if pstyle is not None else ""
            lower_style = style_val.lower()
            if "title" in lower_style or lower_style == "heading1":
                blocks.append(f"# {para}")
            elif lower_style == "heading2":
                blocks.append(f"## {para}")
            elif lower_style.startswith("heading"):
                blocks.append(f"### {para}")
            else:
                blocks.append(para)
        elif tag == "tbl":
            for tr in child.findall(".//w:tr", ns):
                cells = []
                for tc in tr.findall(".//w:tc", ns):
                    cell_text = "".join(t.text for t in tc.findall(".//w:t", ns) if t.text).strip()
                    cells.append(cell_text)
                if any(cells):
                    blocks.append(" | ".join(cells))

    text = "\n\n".join(blocks).strip()
    if not text:
        return None, f"Empty DOCX (no extractable text) '{rel_path}'."

    doc = Document(
        doc_id=rel_path,
        path=str(path),
        type="docx",
        text=text,
        metadata={"size_bytes": size_bytes},
    )
    return doc, None


def _detect_csv_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _load_csv(
    path: Path,
    rel_path: str,
    size_bytes: int,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_rows: int = DEFAULT_MAX_CSV_ROWS,
) -> tuple[Document | None, list[str]]:
    warnings: list[str] = []
    try:
        raw_text, truncated_bytes = _read_text_with_fallback(path, max_bytes=max_file_bytes)
    except Exception as exc:
        return None, [f"Unreadable CSV '{rel_path}': {exc}"]

    if not raw_text.strip():
        return None, [f"Empty CSV file '{rel_path}'."]

    delimiter = _detect_csv_delimiter(raw_text)
    reader = csv.DictReader(io.StringIO(raw_text), delimiter=delimiter)
    if not reader.fieldnames:
        return None, [f"CSV file '{rel_path}' has no header columns."]

    columns = [c.strip() if c is not None else "" for c in reader.fieldnames]
    rows: list[dict[str, str]] = []
    truncated_rows = False
    for raw_row in reader:
        if len(rows) >= max_rows:
            truncated_rows = True
            break
        cleaned = {
            (k.strip() if k is not None else f"col_{idx}"): ("" if v is None else str(v).strip())
            for idx, (k, v) in enumerate(raw_row.items())
            if k is not None or (v is not None and str(v).strip() != "")
        }
        if any(cleaned.values()):
            rows.append(cleaned)

    if not rows:
        return None, [f"CSV file '{rel_path}' contains headers but 0 data rows."]

    if truncated_bytes or truncated_rows:
        warnings.append(
            f"Large CSV '{rel_path}' ({size_bytes} B) capped at {len(rows)} rows."
        )

    dtypes = {col: _infer_dtype([row.get(col, "") for row in rows]) for col in columns}
    profile = CsvSchemaProfile(
        columns=columns,
        dtypes=dtypes,
        sample_rows=rows[:3],
        row_count=len(rows),
    )

    row_texts = [
        "\n".join(f"{col}: {row.get(col, '')}" for col in columns)
        for row in rows
    ]
    doc = Document(
        doc_id=rel_path,
        path=str(path),
        type="csv",
        text="\n\n".join(row_texts),
        metadata={
            "size_bytes": size_bytes,
            "csv_profile": profile.model_dump(),
            "rows": rows,
            "delimiter": delimiter,
            "truncated": truncated_bytes or truncated_rows,
        },
    )
    return doc, warnings


def _col_ref_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(0, idx - 1)


def _load_xlsx(path: Path, rel_path: str, size_bytes: int) -> tuple[Document | None, str | None]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            sheet_names = sorted(
                n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            )
            if not sheet_names:
                return None, f"Invalid XLSX (no worksheets) '{rel_path}'."

            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in names:
                ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in ss_root:
                    parts = [elem.text or "" for elem in si.iter() if elem.tag.endswith("}t")]
                    shared_strings.append("".join(parts))

            sections: list[str] = []
            for idx, sheet_file in enumerate(sheet_names, start=1):
                s_root = ET.fromstring(zf.read(sheet_file))
                row_lines: list[str] = []
                for row_el in s_root.iter():
                    if not row_el.tag.endswith("}row"):
                        continue
                    row_cells: dict[int, str] = {}
                    for cell in row_el:
                        if not cell.tag.endswith("}c"):
                            continue
                        col_idx = _col_ref_to_index(cell.attrib.get("r", "A1"))
                        ctype = cell.attrib.get("t", "")
                        val = ""
                        if ctype == "s":
                            v_el = next((e for e in cell if e.tag.endswith("}v")), None)
                            if v_el is not None and v_el.text is not None:
                                ss_idx = int(v_el.text)
                                if 0 <= ss_idx < len(shared_strings):
                                    val = shared_strings[ss_idx]
                        elif ctype == "inlineStr":
                            val = "".join(
                                e.text or "" for e in cell.iter() if e.tag.endswith("}t")
                            )
                        else:
                            v_el = next((e for e in cell if e.tag.endswith("}v")), None)
                            if v_el is not None and v_el.text is not None:
                                val = v_el.text
                        row_cells[col_idx] = val.strip()
                    if row_cells:
                        max_c = max(row_cells)
                        ordered = [row_cells.get(i, "") for i in range(max_c + 1)]
                        if any(ordered):
                            row_lines.append(" | ".join(ordered))
                if row_lines:
                    sections.append(f"## Sheet {idx}\n" + "\n".join(row_lines))
    except Exception as exc:
        return None, f"Unreadable XLSX '{rel_path}': {exc}"

    text = "\n\n".join(sections).strip()
    if not text:
        return None, f"Empty XLSX '{rel_path}'."

    doc = Document(
        doc_id=rel_path,
        path=str(path),
        type="xlsx",
        text=text,
        metadata={"size_bytes": size_bytes},
    )
    return doc, None


def load_documents(
    folder: Path | str,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> IngestResult:
    """Recursively load supported documents from a folder without crashing on bad files."""
    root = Path(folder).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    documents: list[Document] = []
    warnings: list[str] = []
    total_scanned = 0
    skipped = 0

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in _IGNORED_FILENAMES:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue

        total_scanned += 1
        rel_path = path.relative_to(root).as_posix()
        ext = path.suffix.lower()

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            warnings.append(f"Unreadable file '{rel_path}': {exc}")
            skipped += 1
            continue

        if ext not in SUPPORTED_EXTENSIONS:
            warnings.append(f"Unsupported file type '{ext or '<none>'}': '{rel_path}' skipped.")
            skipped += 1
            continue

        if ext in {".txt", ".md"}:
            try:
                text_raw, trunc_bytes = _read_text_with_fallback(path, max_bytes=max_file_bytes)
                text = text_raw.strip()
            except Exception as exc:
                warnings.append(f"Unreadable text file '{rel_path}': {exc}")
                skipped += 1
                continue
            if not text:
                warnings.append(f"Empty text file '{rel_path}' skipped.")
                skipped += 1
                continue
            trunc_chars = len(text) > max_doc_chars
            if trunc_chars:
                text = text[:max_doc_chars]
            if trunc_bytes or trunc_chars:
                warnings.append(
                    f"Large text file '{rel_path}' ({size_bytes} B) truncated to "
                    f"{len(text)} chars."
                )
            documents.append(
                Document(
                    doc_id=rel_path,
                    path=str(path),
                    type=ext.lstrip("."),
                    text=text,
                    metadata={
                        "size_bytes": size_bytes,
                        "truncated": trunc_bytes or trunc_chars,
                    },
                )
            )
        elif ext == ".pdf":
            doc, pdf_warnings = _load_pdf(
                path, rel_path, size_bytes, max_doc_chars=max_doc_chars
            )
            warnings.extend(pdf_warnings)
            if doc is None:
                skipped += 1
            else:
                documents.append(doc)
        elif ext == ".docx":
            doc, warn = _load_docx(path, rel_path, size_bytes)
            if warn:
                warnings.append(warn)
            if doc is None:
                skipped += 1
            else:
                if len(doc.text) > max_doc_chars:
                    doc.text = doc.text[:max_doc_chars]
                    doc.metadata["truncated"] = True
                    warnings.append(
                        f"Large DOCX '{rel_path}' ({size_bytes} B) truncated to "
                        f"{max_doc_chars} chars."
                    )
                documents.append(doc)
        elif ext == ".csv":
            doc, csv_warnings = _load_csv(
                path, rel_path, size_bytes, max_file_bytes=max_file_bytes
            )
            warnings.extend(csv_warnings)
            if doc is None:
                skipped += 1
            else:
                documents.append(doc)
        elif ext == ".xlsx":
            doc, warn = _load_xlsx(path, rel_path, size_bytes)
            if warn:
                warnings.append(warn)
            if doc is None:
                skipped += 1
            else:
                if len(doc.text) > max_doc_chars:
                    doc.text = doc.text[:max_doc_chars]
                    doc.metadata["truncated"] = True
                    warnings.append(
                        f"Large XLSX '{rel_path}' ({size_bytes} B) truncated to "
                        f"{max_doc_chars} chars."
                    )
                documents.append(doc)

    return IngestResult(
        root_path=str(root),
        documents=documents,
        warnings=warnings,
        total_files_scanned=total_scanned,
        skipped_files=skipped,
    )
