"""Build spares_catalogue.csv (160 rows) and 2 supplier PDFs under demos/ship_spares/docs/."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Render clean multi-page text PDFs extractable by pypdf."""
    writer = PdfWriter()
    for page_lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        font_dict = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font_dict)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )

        wrapped: list[str] = []
        for raw_line in page_lines:
            if not raw_line.strip():
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(raw_line, width=82))

        cmds = ["BT", "/F1 11 Tf", "54 740 Td"]
        for line in wrapped[:42]:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            cmds.append(f"({escaped}) Tj")
            cmds.append("0 -15 Td")
        cmds.append("ET")

        stream = DecodedStreamObject()
        stream.set_data("\n".join(cmds).encode("latin-1", errors="replace"))
        stream_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = ArrayObject([stream_ref])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        writer.write(fh)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    suppliers = [
        "NordicMarine Supply",
        "BalticValve GmbH",
        "PacificPump Co",
        "AegisElectro Ltd",
    ]
    locations = ["Rotterdam-WH1", "Singapore-WH2", "Hamburg-WH3", "Busan-WH4"]
    families = [
        ("Main Engine Fuel Injector", 420.0, 2, 7, False),
        ("Cylinder Liner Sealing Ring", 68.5, 5, 5, False),
        ("Ballast Centrifugal Impeller", 890.0, 1, 10, False),
        ("High-Pressure Hydraulic Valve", 315.0, 2, 6, False),
        ("Marine Lithium Battery Module", 1250.0, 1, 14, True),
        ("Refrigerant R404A Service Cylinder", 195.0, 2, 8, True),
        ("Bilge Oily-Water Sensor Probe", 260.0, 2, 9, False),
        ("Boiler Burner Igniter Assembly", 145.0, 4, 5, False),
        ("Anti-Corrosion Zinc Anode Block", 42.0, 10, 4, False),
        ("Synthetic Gear Oil 20L Drum", 175.0, 2, 6, True),
    ]

    csv_path = OUT_DIR / "spares_catalogue.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "part_no",
                "description",
                "unit_price_usd",
                "min_order_qty",
                "lead_time_days",
                "supplier",
                "stock_location",
                "hazmat_flag",
            ]
        )
        for idx in range(1, 161):
            fam_name, base_price, moq, base_lead, hazmat = families[(idx - 1) % len(families)]
            supplier = suppliers[(idx - 1) % len(suppliers)]
            location = locations[((idx - 1) // 2) % len(locations)]
            variant = f"MK-{(idx % 5) + 1}"
            price = round(base_price + ((idx % 7) * 12.5), 2)
            lead = base_lead + (idx % 4)
            writer.writerow(
                [
                    f"SP-{1000 + idx}",
                    f"{fam_name} {variant}",
                    f"{price:.2f}",
                    moq,
                    lead,
                    supplier,
                    location,
                    "true" if hazmat else "false",
                ]
            )

    terms_pages = [
        [
            "SUPPLIER COMMERCIAL TERMS AND RETURNS POLICY (DOC SUP-01)",
            "",
            "1. Payment Terms and Early Settlement",
            "Standard invoice terms across all four approved suppliers are Net 30 calendar",
            "days from bill of lading date.",
            "Purchase orders totaling USD 5,000 or more qualify for a 2% early-payment",
            "discount when settled within 10 calendar days of invoice issue.",
            "",
            "2. Line-Item Volume Discounts and Handling Fees",
            "Orders meeting or exceeding 10 units on a single part line receive a 5% unit",
            "price discount. Orders meeting or exceeding 25 units on a single part line",
            "receive a 10% unit price discount (discounts do not stack).",
            "Any purchase order with total merchandise value below USD 500 incurs a flat",
            "USD 125 small-order handling fee.",
            "Every order must meet or exceed the catalogue min_order_qty for each line.",
        ],
        [
            "3. Returns, Restocking, and Hazmat Exclusions",
            "Unused non-hazmat spares in original sealed packaging may be returned within",
            "30 calendar days of delivery subject to a 15% restocking fee.",
            "Items flagged hazmat_flag=true in spares_catalogue.csv are strictly",
            "non-returnable once dispatched from the supplier warehouse.",
            "Factory-defective parts (documented within 14 calendar days) are replaced at",
            "zero restocking fee with return freight covered by the supplier.",
        ],
    ]

    logistics_pages = [
        [
            "MARINE SPARES LOGISTICS AND LEAD-TIME RULES (DOC SUP-02)",
            "",
            "1. Effective Lead-Time Calculation",
            "Base lead_time_days in spares_catalogue.csv measures warehouse dispatch to port",
            "gate under normal non-hazmat conditions.",
            "Add +4 calendar days for dangerous-goods documentation and customs clearance",
            "whenever hazmat_flag is true.",
            "Add +3 calendar days for inter-port consolidation when stock_location differs",
            "from the vessel's designated receiving hub.",
            "Formula: effective_lead_days = lead_time_days + (4 if hazmat) + (3 if transfer).",
        ],
        [
            "2. Expedited Air Freight and Port Cutoffs",
            "Non-hazmat spares may use expedited air freight, cutting base lead_time_days",
            "by 50% (rounded up to the nearest whole day) for a flat USD 450 surcharge.",
            "Items with hazmat_flag=true are prohibited from standard air-freight expedite",
            "and must move via sea/dangerous-goods charter.",
            "Weekly port consolidation cutoffs close every Thursday at 16:00 UTC.",
        ],
    ]

    write_pdf(OUT_DIR / "supplier_terms_and_returns.pdf", terms_pages)
    write_pdf(OUT_DIR / "logistics_and_lead_times.pdf", logistics_pages)
    print("Generated 160-row CSV + 2 supplier PDFs in", OUT_DIR)


if __name__ == "__main__":
    main()
