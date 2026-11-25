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


def _read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError("Binary null bytes detected in text file.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


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


def _load_pdf(path: Path, rel_path: str, size_bytes: int) -> tuple[Document | None, str | None]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return None, f"Unreadable PDF '{rel_path}': {exc}"

    pages: list[PageContent] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            return None, f"Failed extracting page {idx} of PDF '{rel_path}': {exc}"
        pages.append(PageContent(page_number=idx, text=text))

    combined = "\n\n".join(p.text for p in pages if p.text).strip()
    if not combined:
        return None, (
            f"Scanned or empty PDF (no extractable text) '{rel_path}' "
            f"({len(reader.pages)} pages)."
        )

    doc = Document(
        doc_id=rel_path,
        path=str(path),
        type="pdf",
        text=combined,
        pages=pages,
        metadata={"size_bytes": size_bytes, "page_count": len(pages)},
    )
    return doc, None


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


def _load_csv(path: Path, rel_path: str, size_bytes: int) -> tuple[Document | None, str | None]:
    try:
        raw_text = _read_text_with_fallback(path)
    except Exception as exc:
        return None, f"Unreadable CSV '{rel_path}': {exc}"

    if not raw_text.strip():
        return None, f"Empty CSV file '{rel_path}'."

    reader = csv.DictReader(io.StringIO(raw_text))
    if not reader.fieldnames:
        return None, f"CSV file '{rel_path}' has no header columns."

    columns = [c.strip() if c is not None else "" for c in reader.fieldnames]
    rows: list[dict[str, str]] = []
    for raw_row in reader:
        cleaned = {
            (k.strip() if k is not None else f"col_{idx}"): ("" if v is None else str(v).strip())
            for idx, (k, v) in enumerate(raw_row.items())
            if k is not None or (v is not None and str(v).strip() != "")
        }
        if any(cleaned.values()):
            rows.append(cleaned)

    if not rows:
        return None, f"CSV file '{rel_path}' contains headers but 0 data rows."

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
        },
    )
    return doc, None


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


def load_documents(folder: Path | str) -> IngestResult:
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
                text = _read_text_with_fallback(path).strip()
            except Exception as exc:
                warnings.append(f"Unreadable text file '{rel_path}': {exc}")
                skipped += 1
                continue
            if not text:
                warnings.append(f"Empty text file '{rel_path}' skipped.")
                skipped += 1
                continue
            documents.append(
                Document(
                    doc_id=rel_path,
                    path=str(path),
                    type=ext.lstrip("."),
                    text=text,
                    metadata={"size_bytes": size_bytes},
                )
            )
        elif ext == ".pdf":
            doc, warn = _load_pdf(path, rel_path, size_bytes)
            if warn:
                warnings.append(warn)
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
                documents.append(doc)
        elif ext == ".csv":
            doc, warn = _load_csv(path, rel_path, size_bytes)
            if warn:
                warnings.append(warn)
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
                documents.append(doc)

    return IngestResult(
        root_path=str(root),
        documents=documents,
        warnings=warnings,
        total_files_scanned=total_scanned,
        skipped_files=skipped,
    )
