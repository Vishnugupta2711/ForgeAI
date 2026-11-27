"""Build spares_catalogue.csv (160 rows) and 2 expanded supplier PDFs (v2)."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"
LINES_PER_PAGE = 42


def _paginate_lines(sections: list[list[str]]) -> list[list[str]]:
    """Wrap section paragraphs cleanly and split across pages without dropping lines."""
    wrapped_all: list[str] = []
    for sec in sections:
        for raw_line in sec:
            if not raw_line.strip():
                if wrapped_all and wrapped_all[-1] != "":
                    wrapped_all.append("")
            else:
                wrapped_all.extend(textwrap.wrap(raw_line, width=82))
        if wrapped_all and wrapped_all[-1] != "":
            wrapped_all.append("")

    pages: list[list[str]] = []
    current: list[str] = []
    for line in wrapped_all:
        if len(current) >= LINES_PER_PAGE:
            pages.append(current)
            current = []
        if not current and line == "":
            continue
        current.append(line)
    if current:
        pages.append(current)
    return pages


def write_pdf(path: Path, sections: list[list[str]]) -> None:
    """Render clean multi-page text PDFs extractable by pypdf with automatic pagination."""
    pages = _paginate_lines(sections)
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

        cmds = ["BT", "/F1 11 Tf", "54 740 Td"]
        for line in page_lines:
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

    terms_sections = [
        [
            "SUPPLIER COMMERCIAL TERMS AND RETURNS POLICY (DOC SUP-01, REV 2.0)",
            "",
            "1. Approved Suppliers and Catalogue Pricing Authority",
            (
                "These commercial terms govern all marine spare-parts purchase orders placed "
                "with the four approved framework suppliers in spares_catalogue.csv: "
                "NordicMarine Supply, BalticValve GmbH, PacificPump Co, and AegisElectro Ltd."
            ),
            (
                "Catalogue unit_price_usd, min_order_qty, lead_time_days, stock_location, "
                "and hazmat_flag fields in spares_catalogue.csv represent binding baseline "
                "master data for routine vessel procurement."
            ),
            (
                "Informal quotes from local port agents do not override spares_catalogue.csv "
                "pricing unless backed by a signed Fleet Procurement price addendum valid "
                "for at least 90 calendar days."
            ),
        ],
        [
            "2. Minimum Order Quantities and Multi-Line Consolidation",
            (
                "Every purchase order line item must meet or exceed the catalogue "
                "min_order_qty specified for that part_no in spares_catalogue.csv."
            ),
            (
                "Orders requesting fewer units than min_order_qty are auto-rejected unless "
                "approved as a critical off-hire breakdown AND unit_price_usd >= USD 200 on "
                "a non-hazmat item, incurring a USD 75 broken-pack surcharge."
            ),
            (
                "A single purchase order may combine multiple SKUs stocked at different "
                "warehouses or suppliers; however, volume discount tiers evaluate each "
                "single SKU line independently and never sum across different part numbers."
            ),
        ],
        [
            "3. Line-Item Volume Discounts",
            (
                "Standard catalogue pricing applies to single-part line quantities from 1 "
                "up to 9 units (0 percent discount, subject to min_order_qty)."
            ),
            (
                "Single-part orders of 10 to 24 units on one part_no receive a 5 percent "
                "unit price discount across all units on that line."
            ),
            (
                "Single-part orders of 25 or more units on one part_no receive a 10 percent "
                "unit price discount across all units on that line."
            ),
            (
                "Volume discounts are strictly non-stacking and cannot be retroactively "
                "combined across separate purchase orders placed in the same calendar week."
            ),
        ],
        [
            "4. Small-Order Handling and Dangerous-Goods Fees",
            (
                "Any purchase order whose total discounted merchandise subtotal (sum of "
                "discounted line totals, before fees or freight) is strictly below USD 500 "
                "incurs a flat USD 125 small-order handling fee."
            ),
            (
                "Purchase orders with a discounted merchandise subtotal of USD 500 or more "
                "incur USD 0 small-order handling fee."
            ),
            (
                "Every purchase order containing one or more items with hazmat_flag=true in "
                "spares_catalogue.csv incurs a mandatory USD 85 dangerous-goods (DG) "
                "documentation and IMDG packing fee per purchase order (flat per PO)."
            ),
            (
                "Cross-policy reference: split-shipment administrative fees and air-"
                "expedite surcharges are governed by DOC SUP-02, Sections 4 and 6."
            ),
        ],
        [
            "5. Payment Terms, Early Settlement, and Late Interest",
            (
                "Standard invoice payment terms across all four approved suppliers are "
                "Net 30 calendar days from Bill of Lading (BoL) or warehouse dispatch receipt "
                "date, settled in USD via SWIFT wire."
            ),
            (
                "Purchase orders with a discounted merchandise subtotal of USD 5,000 or more "
                "qualify for a 2 percent early-payment discount when paid in full within "
                "10 calendar days of invoice issue date."
            ),
            (
                "The 2 percent early-payment discount applies solely to the discounted "
                "merchandise subtotal and never discounts small-order, DG, or freight fees."
            ),
            (
                "Invoices overdue > 30 calendar days accrue 1.5 percent interest per 30-day "
                "period and hold open dispatches once an account is > 45 days past due."
            ),
        ],
        [
            "6. Order Cancellation and Pre-Dispatch Amendments",
            (
                "Unshipped non-hazmat orders (hazmat_flag=false) may be cancelled free of "
                "charge at least 48 hours before scheduled warehouse dispatch (or before "
                "Thursday 16:00 UTC weekly cutoff per DOC SUP-02, Section 3)."
            ),
            (
                "Non-hazmat cancellations submitted fewer than 48 hours prior to dispatch "
                "incur a 5 percent cancellation fee (minimum USD 50 per cancelled PO)."
            ),
            (
                "For items flagged hazmat_flag=true, once dangerous-goods booking is filed "
                "(> 24 hours after PO confirmation), cancellation incurs a flat USD 150 DG "
                "carrier cancellation penalty prior to dispatch."
            ),
        ],
        [
            "7. Returns Eligibility, Restocking Windows, and Hazmat Ban",
            (
                "Unused spares flagged hazmat_flag=false in spares_catalogue.csv, kept in "
                "unopened OEM sealed packaging, may be returned within 30 calendar days of "
                "delivery subject to a 15 percent restocking fee (buyer pays return freight)."
            ),
            (
                "Non-hazmat returns between 31 and 60 calendar days require supplier RMA "
                "approval and incur a 25 percent restocking fee. Non-hazmat spares older "
                "than 60 calendar days from delivery are non-returnable."
            ),
            (
                "Strict hazmat exclusion: All items with hazmat_flag=true in "
                "spares_catalogue.csv are strictly non-returnable once dispatched from the "
                "supplier warehouse, regardless of condition or unopened packaging."
            ),
        ],
        [
            "8. OEM Warranty, Transit Damage, and Defective Replacements",
            (
                "All four suppliers provide a 12-month OEM workmanship warranty from "
                "installation or 18 months from port delivery, whichever occurs first."
            ),
            (
                "Factory-defective parts, wrong SKUs shipped in error, or transit-damaged "
                "spares documented with photos within 14 calendar days of port gate receipt "
                "(DOC SUP-02, Section 8) are replaced or credited at 0 percent restocking fee."
            ),
            (
                "On verified defective or wrong-shipment claims, return freight is paid 100 "
                "percent by the supplier; defective hazmat_flag=true stock is collected via "
                "supplier-arranged licensed IMDG DG return pickup at zero cost to the buyer."
            ),
        ],
    ]

    logistics_sections = [
        [
            "MARINE SPARES LOGISTICS AND LEAD-TIME RULES (DOC SUP-02, REV 2.0)",
            "",
            "1. Regional Warehouse Network and Base Lead Times",
            (
                "Approved spares in spares_catalogue.csv ship from four regional bonded "
                "warehouses: Rotterdam-WH1, Singapore-WH2, Hamburg-WH3, and Busan-WH4."
            ),
            (
                "The catalogue column lead_time_days measures baseline calendar days from "
                "PO confirmation to port gate release under standard non-hazmat conditions "
                "when the vessel's receiving hub matches stock_location."
            ),
            (
                "Orders placed after 14:00 local warehouse time count the following business "
                "day as Day 0 for lead-time tracking."
            ),
        ],
        [
            "2. Standard Effective Lead-Time Formula",
            (
                "To compute standard sea/land effective delivery lead time to a vessel's "
                "receiving hub, apply: effective_lead_days = lead_time_days + hazmat_days "
                "+ transfer_days."
            ),
            (
                "Add +4 calendar days (hazmat_days = 4) whenever hazmat_flag is true in "
                "spares_catalogue.csv for IMDG dangerous-goods clearance and documentation."
            ),
            (
                "Add +3 calendar days (transfer_days = 3) whenever catalogue stock_location "
                "differs from the vessel's designated receiving hub (inter-port feeder)."
            ),
            (
                "Example logic: a part with lead_time_days=10, hazmat_flag=true, and "
                "stock_location != receiving hub yields 10 + 4 + 3 = 17 effective days."
            ),
        ],
        [
            "3. Weekly Consolidation Cutoffs and Port Exceptions",
            (
                "Inter-port sea consolidation shuttles close every Thursday at 16:00 UTC "
                "across all hubs. Transfer orders confirmed after Thursday 16:00 UTC wait "
                "for the next weekly Thursday cycle (+1 to +7 calendar days cutoff wait)."
            ),
            (
                "Local same-port collections (stock_location == receiving hub) dispatch "
                "daily and do NOT wait for the Thursday 16:00 UTC consolidation cutoff."
            ),
            (
                "European hub exception: Road shuttle transfers exclusively between "
                "Rotterdam-WH1 and Hamburg-WH3 for non-hazmat orders under 50 kg reduce "
                "transfer_days from +3 days down to +1 calendar day."
            ),
            (
                "Seasonal congestion buffer: During peak dry-docking windows (March 1 to "
                "April 30 and September 1 to October 31), standard sea dispatches out of "
                "Singapore-WH2 and Busan-WH4 add +2 calendar days to effective_lead_days."
            ),
        ],
        [
            "4. Expedited Air Freight Eligibility, Math, and Surcharges",
            (
                "Non-hazmat spares (hazmat_flag=false) with total shipment weight under "
                "150 kg qualify for expedited air freight upon buyer instruction."
            ),
            (
                "Expedited air freight cuts base lead_time_days by 50 percent, rounded UP "
                "to the nearest whole calendar day (ceil(lead_time_days / 2)). For example, "
                "a 7-day base lead time becomes 4 days."
            ),
            (
                "If expedited air freight must transfer between different ports "
                "(stock_location != receiving_hub), add +2 calendar days air-hub transit "
                "instead of the +3 sea-consolidation transfer."
            ),
            (
                "Air expedite incurs a flat USD 450 surcharge per PO up to 30 kg gross "
                "weight, plus USD 6.50 per kilogram for shipments between 31 kg and 149 kg."
            ),
        ],
        [
            "5. Hazardous Materials (IMDG) Prohibitions",
            (
                "Parts flagged hazmat_flag=true in spares_catalogue.csv (Marine Lithium "
                "Battery Modules, Refrigerant R404A Service Cylinders, and Synthetic Gear "
                "Oil 20L Drums) are strictly prohibited from expedited air freight (Sec 4)."
            ),
            (
                "All hazmat_flag=true spares must travel via IMDG-certified sea freight or "
                "coastal DG charter and always incur both the +4 day DG customs window and "
                "the USD 85 DG documentation fee (DOC SUP-01, Section 4)."
            ),
            (
                "Cross-policy return ban: once dispatched, hazmat_flag=true spares cannot "
                "be returned unless factory-defective (DOC SUP-01, Sections 7 and 8)."
            ),
        ],
        [
            "6. Mixed-PO Split Shipments and Late-Delivery Cancellation",
            (
                "When a multi-line PO contains both hazmat_flag=false and hazmat_flag=true "
                "parts, buyers may split dispatch so non-hazmat lines ship by standard or "
                "air freight while hazmat lines follow DG sea schedules."
            ),
            (
                "Split dispatch incurs a USD 65 split-AWB administration fee when total PO "
                "merchandise value is below USD 2,500; waived on POs of USD 2,500 or more."
            ),
            (
                "Suppliers must flag backorders within 1 working day. If dispatch slips > 5 "
                "calendar days past quoted effective_lead_days, the buyer may cancel "
                "delayed lines penalty-free regardless of DOC SUP-01 Section 6."
            ),
        ],
        [
            "7. Incoterms, Outer-Anchorage Launches, and Customs Paperwork",
            (
                "Standard catalogue prices assume DAP (Delivered at Place) delivery to "
                "the bonded receiving port warehouse gate."
            ),
            (
                "Direct vessel delivery at outer anchorage requires a port supply launch, "
                "incurring a flat USD 190 barge fee unless bundled >= USD 3,000."
            ),
            (
                "Missing HS tariff codes delay customs release by 2 to 4 calendar days. "
                "NordicMarine Supply and BalticValve GmbH automatically pre-file EUR.1 "
                "origin certificates for Rotterdam-WH1 and Hamburg-WH3 arrivals."
            ),
        ],
        [
            "8. Storage, Shelf-Life, and 14-Day Receiving Inspection",
            (
                "Marine Lithium Battery Modules (hazmat_flag=true) must ship at 30 to 40 "
                "percent state-of-charge in UN crates and require climate-controlled "
                "storage between 10 and 25 degrees Celsius at the receiving hub."
            ),
            (
                "Synthetic Gear Oil 20L Drums and Refrigerant R404A Cylinders must arrive "
                "with at least 18 months remaining shelf life at dispatch."
            ),
            (
                "Receiving engineers have 14 calendar days from port gate receipt to log "
                "packaging damage, shortages, or defects; timely documentation triggers "
                "zero-fee replacement under DOC SUP-01 Section 8."
            ),
        ],
    ]

    write_pdf(OUT_DIR / "supplier_terms_and_returns.pdf", terms_sections)
    write_pdf(OUT_DIR / "logistics_and_lead_times.pdf", logistics_sections)
    print("Generated 160-row CSV + 2 expanded supplier PDFs (v2) in", OUT_DIR)


if __name__ == "__main__":
    main()
