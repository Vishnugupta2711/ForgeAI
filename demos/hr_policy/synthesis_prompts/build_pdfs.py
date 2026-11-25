"""Build the 3 synthetic HR policy PDFs under demos/hr_policy/docs/."""

from __future__ import annotations

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
    leave_pages = [
        [
            "SHORE STAFF LEAVE POLICY (DOC HR-POL-01)",
            "",
            "1. Scope and Eligibility",
            "This policy applies to all full-time shore staff based at regional port offices.",
            "Offshore vessel crew and external contractors follow separate maritime agreements.",
            "New shore staff complete a 90-calendar-day probation period from hire date.",
            "",
            "2. Annual Leave Entitlement and Accrual",
            "Full-time shore staff accrue 21 working days of paid annual leave per calendar year",
            "(1.75 days per completed month of active service).",
            "During the 90-calendar-day probation period, paid annual leave cannot be taken.",
            "Up to 5 unused annual leave days may carry over to Q1 and expire on March 31.",
            "Annual leave accrual pauses during any unpaid administrative suspension exceeding",
            "5 working days under the Shore Staff Code of Conduct.",
        ],
        [
            "3. Notice Periods and Approval Tiers",
            "Requests of 1 to 5 consecutive working days require manager approval at least",
            "5 working days in advance.",
            "Requests exceeding 5 consecutive working days require at least 14 calendar days",
            "advance notice and secondary Vice President (VP) sign-off.",
            "Emergency bereavement leave grants 5 paid days for immediate family without",
            "advance notice.",
            "",
            "4. Sick and Parental Leave",
            "Shore staff receive 12 paid sick days annually. Consecutive absences of 3 or more",
            "days require a licensed physician fitness certificate.",
            "Employees in their 90-day probation period are limited to 2 paid emergency sick",
            "days; additional sick days during probation are unpaid.",
            "Primary caregivers receive 16 weeks paid parental leave after 6 months tenure;",
            "secondary caregivers receive 4 weeks paid parental leave.",
        ],
    ]

    conduct_pages = [
        [
            "SHORE STAFF CODE OF CONDUCT (DOC HR-POL-02)",
            "",
            "1. Working Hours and Attendance",
            "Core shore-office hours are 09:00 to 17:30 local time, Monday through Friday,",
            "with a 45-minute unpaid midday break. Hybrid staff must work on-site at least",
            "3 days per week unless granted a written medical accommodation.",
            "",
            "2. Conflicts of Interest and Vendor Gifts",
            "Shore staff must disclose any financial interest above 1% in a maritime vendor.",
            "Accepting cash or cash equivalents (gift cards) of any value is strictly prohibited.",
            "Non-cash vendor gifts are capped at USD 50 per single item and USD 150 cumulative",
            "per calendar year from any single supplier.",
            "Gifts exceeding USD 50 must be declined or surrendered to HR Compliance within",
            "5 working days.",
        ],
        [
            "3. Confidentiality and Data Security",
            "Crew manifests, vessel routing, and supplier bid pricing are Confidential.",
            "Confidential records must never be uploaded to unapproved public web services.",
            "",
            "4. Disciplinary Escalation Tiers",
            "Tier 1 (Verbal/Written Coaching): Minor attendance or documentation lapses;",
            "remains active on record for 6 months.",
            "Tier 2 (Formal Written Warning / Suspension): Repeated policy breaches or",
            "undisclosed conflicts; carries up to 10 working days unpaid administrative",
            "suspension. Per the Leave Policy, unpaid suspensions over 5 working days pause",
            "monthly leave accrual. Expense reimbursements are frozen until HR Compliance",
            "clears the Tier 2 review.",
            "Tier 3 (Summary Termination): Fraud, falsified receipts, or safety violations.",
        ],
    ]

    expenses_pages = [
        [
            "SHORE STAFF EXPENSES AND TRAVEL POLICY (DOC HR-POL-03)",
            "",
            "1. Travel Pre-Approval and Flight Classes",
            "All business travel requires line-manager pre-approval at least 7 calendar days",
            "before departure. Flights under 6 hours scheduled duration must book Economy.",
            "Flights of 6 hours or more qualify for Premium Economy.",
            "Employees in their 90-calendar-day probation period (defined in Leave Policy)",
            "require Director pre-approval for any travel and are capped at 75% of standard",
            "hotel limits.",
            "",
            "2. Combining Business Travel with Annual Leave",
            "Shore staff may extend a business trip with personal annual leave only with prior",
            "manager sign-off. Return economy airfare remains reimbursable only if the personal",
            "leave extension does not exceed 5 consecutive working days; longer extensions",
            "trigger the Leave Policy 14-day VP sign-off and shift return airfare to personal",
            "cost if fare variance exceeds USD 150.",
        ],
        [
            "3. Lodging, Per Diem, and Client Meals",
            "Standard hotel caps are USD 180 per night (Tier A port hubs) and USD 135 per",
            "night (Tier B regional offices). Daily meal per diem is USD 60 per full travel",
            "day with itemized receipts required for any single expense of USD 25 or more.",
            "Client entertainment meals are capped at USD 75 per attendee and must never",
            "violate the USD 50 single-gift vendor restriction in the Code of Conduct.",
            "",
            "4. Submission Deadlines and Reimbursement Holds",
            "Expense claims must be submitted within 14 calendar days of trip completion.",
            "Claims submitted more than 30 days late are forfeited unless approved by Finance.",
            "Under the Code of Conduct, employees undergoing a Tier 2 disciplinary review have",
            "expense reimbursements held until HR Compliance issues written clearance.",
        ],
    ]

    write_pdf(OUT_DIR / "leave_policy.pdf", leave_pages)
    write_pdf(OUT_DIR / "code_of_conduct.pdf", conduct_pages)
    write_pdf(OUT_DIR / "expenses_and_travel_policy.pdf", expenses_pages)
    print("Generated 3 HR policy PDFs in", OUT_DIR)


if __name__ == "__main__":
    main()
