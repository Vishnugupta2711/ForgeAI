"""Unit tests for forge.ingest (loaders, chunking, CSV profiling, warnings)."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from forge.ingest import chunk_document, ingest_folder, load_documents

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "phase1_sample"


def _create_text_pdf(path: Path, pages_text: list[str]) -> None:
    """Create a genuine extractable multi-page PDF using pypdf primitives."""
    writer = PdfWriter()
    for content_lines in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        font_dict = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font_dict)
        resources = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        page[NameObject("/Resources")] = resources

        stream_cmds = ["BT", "/F1 12 Tf", "72 720 Td"]
        for line in content_lines.splitlines():
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_cmds.append(f"({escaped}) Tj")
            stream_cmds.append("0 -16 Td")
        stream_cmds.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(stream_cmds).encode("latin-1"))
        stream_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = ArrayObject([stream_ref])

    with path.open("wb") as fh:
        writer.write(fh)


def _create_blank_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as fh:
        writer.write(fh)


def _create_minimal_docx(path: Path, heading: str, body: str) -> None:
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr>"
        f"<w:r><w:t>{heading}</w:t></w:r></w:p>"
        f"<w:p><w:r><w:t>{body}</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)


def _create_minimal_xlsx(path: Path) -> None:
    ss_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>code</t></si><si><t>status</t></si><si><t>A10</t></si><si><t>ready</t></si>"
        "</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c></row>'
        "</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", ss_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_ingest_fixtures_pdf_and_warnings(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "docs"
    shutil.copytree(FIXTURE_DIR, corpus_dir)

    # 1 multi-page PDF generated in-test
    _create_text_pdf(
        corpus_dir / "deck_guide.pdf",
        [
            "1. Antenna Alignment\nPoint directional yagi east at 85 degrees azimuth.",
            "2. Grounding Check\nVerify earth strap resistance stays below 2 ohms.",
        ],
    )
    # Scanned/blank PDF + unknown binary file should produce warnings without crashing
    _create_blank_pdf(corpus_dir / "scanned_empty.pdf")
    (corpus_dir / "diagram.bin").write_bytes(b"\x00\x01\x02binary")

    ingest_res, chunks, profile = ingest_folder(corpus_dir, chunk_size=140, overlap=25)

    assert ingest_res.total_files_scanned == 7
    assert profile.loaded_docs == 5  # 2 txt, 1 md, 1 csv, 1 text pdf
    assert profile.skipped_files == 2
    assert len(profile.warnings) == 2
    assert any("Scanned or empty PDF" in w for w in profile.warnings)
    assert any("Unsupported file type '.bin'" in w for w in profile.warnings)

    by_id = {d.doc_id: d for d in ingest_res.documents}
    assert by_id["deck_guide.pdf"].type == "pdf"
    assert len(by_id["deck_guide.pdf"].pages) == 2

    # Check chunk boundaries and metadata
    pdf_chunks = [c for c in chunks if c.doc_id == "deck_guide.pdf"]
    assert len(pdf_chunks) == 2
    assert pdf_chunks[0].page == 1
    assert pdf_chunks[0].section_heading == "1. Antenna Alignment"
    assert pdf_chunks[1].page == 2
    assert pdf_chunks[1].section_heading == "2. Grounding Check"

    md_chunks = [c for c in chunks if c.doc_id == "acoustic_guide.md"]
    assert all(len(c.text) <= 140 for c in md_chunks)
    assert any(c.section_heading == "Deployment Depth" for c in md_chunks)
    assert any(c.section_heading == "Power Budget" for c in md_chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    # Check CSV schema profile + 1 chunk per row in "col: value" form
    csv_doc = by_id["sensors.csv"]
    csv_prof = csv_doc.metadata["csv_profile"]
    assert csv_prof["columns"] == ["sensor_id", "model", "depth_m", "battery_days", "active"]
    assert csv_prof["row_count"] == 4
    assert csv_prof["dtypes"]["depth_m"] == "int"
    assert csv_prof["dtypes"]["active"] == "bool"
    assert len(csv_prof["sample_rows"]) == 3

    csv_chunks = [c for c in chunks if c.doc_id == "sensors.csv"]
    assert len(csv_chunks) == 4
    assert "sensor_id: SNS-101" in csv_chunks[0].text
    assert "battery_days: 60" in csv_chunks[0].text

    compact = profile.to_compact_text()
    assert "acoustic_guide.md" in compact
    assert "sensors.csv" in compact


def test_docx_xlsx_and_chunk_overlap(tmp_path: Path) -> None:
    _create_minimal_docx(
        tmp_path / "ops.docx",
        heading="Inspection Rules",
        body=" ".join(f"Sentence {i} checks calibration tolerance." for i in range(12)),
    )
    _create_minimal_xlsx(tmp_path / "inventory.xlsx")

    res = load_documents(tmp_path)
    assert len(res.documents) == 2

    docx_doc = next(d for d in res.documents if d.type == "docx")
    chunks = chunk_document(docx_doc, chunk_size=120, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.section_heading == "Inspection Rules" for c in chunks)
    # Verify overlap tail appears in subsequent chunk
    assert chunks[0].text[-15:] in chunks[1].text
