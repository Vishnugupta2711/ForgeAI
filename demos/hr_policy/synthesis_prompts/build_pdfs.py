"""Build the 3 expanded HR policy PDFs (v2) under demos/hr_policy/docs/."""

from __future__ import annotations

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
    leave_sections = [
        [
            "SHORE STAFF LEAVE POLICY (DOC HR-POL-01, REV 2.0)",
            "",
            "1. Scope, Eligibility, and Probation Period",
            (
                "This policy applies to all full-time and part-time permanent shore staff "
                "based at regional port offices in Rotterdam, Singapore, Hamburg, and Busan."
            ),
            (
                "Offshore vessel crew, harbor pilots, and external contractors follow "
                "separate maritime collective agreements and are excluded from this handbook."
            ),
            (
                "Every newly hired shore employee completes a 90-calendar-day probation "
                "period starting on their official hire date. Manager confirmation on day 80 "
                "converts the appointment to confirmed shore status."
            ),
            (
                "Part-time shore staff working at least 20 hours per week receive pro-rated "
                "leave entitlements proportional to their contracted weekly hours / 40."
            ),
        ],
        [
            "2. Annual Leave Entitlement and Monthly Accrual",
            (
                "Confirmed full-time shore staff accrue paid annual leave based on completed "
                "continuous service: 21 working days per calendar year (1.75 days/month) for "
                "0 to 3 years of tenure; 24 working days per calendar year (2.00 days/month) "
                "after 3 completed years; and 26 working days per calendar year (2.17 days/"
                "month) after 7 completed years."
            ),
            (
                "During the initial 90-calendar-day probation period, annual leave accrues on "
                "paper at 1.75 days per month but paid annual leave cannot be scheduled or "
                "taken until day 91, except during mandatory local port holiday closures."
            ),
            (
                "In hire or separation years, entitlement is pro-rated by calendar months "
                "served and rounded to the nearest 0.5 working day. Unpaid sabbaticals "
                "exceeding 15 working days halt accrual proportionally."
            ),
            (
                "Cross-policy rule: monthly annual leave accrual pauses immediately during "
                "any unpaid administrative suspension exceeding 5 working days imposed under "
                "the Shore Staff Code of Conduct (DOC HR-POL-02, Section 8)."
            ),
        ],
        [
            "3. Annual Leave Carry-Over, Expiry, and Separation Payout",
            (
                "Up to 5 unused annual leave days may carry over from one calendar year into "
                "Q1 of the following year and expire permanently on March 31 at 17:30 local "
                "time if not taken."
            ),
            (
                "Unused days exceeding the 5-day carry-over cap are forfeited on December 31 "
                "unless a Vice President (VP) grants a written operational-freeze deferral "
                "(for example during a vessel dry-dock emergency), extending expiry to June 30."
            ),
            (
                "Shore staff may not cash out unused annual leave during active employment. "
                "Upon resignation or termination, accrued unused leave is paid out at base "
                "daily salary (annual base salary divided by 260 working days)."
            ),
            (
                "If a departing employee has a negative annual leave balance approved in an "
                "emergency, up to 3 negative days may be deducted from the final settlement."
            ),
        ],
        [
            "4. Notice Periods, Approval Tiers, and Blackout Windows",
            (
                "Short requests of 1 to 2 consecutive working days require direct line-manager "
                "approval at least 3 working days before departure."
            ),
            (
                "Standard requests of 3 to 5 consecutive working days require direct line-"
                "manager approval at least 5 working days in advance."
            ),
            (
                "Extended requests exceeding 5 consecutive working days require at least 14 "
                "calendar days advance notice and secondary Vice President (VP) sign-off "
                "in addition to manager approval."
            ),
            (
                "No more than 10 percent of any port operations team (minimum 1 person at "
                "offices under 8 staff) may be on scheduled annual leave during the final "
                "3 working days of Q2 (June) and Q4 (December) audit close."
            ),
            (
                "When combining business travel with personal annual leave, employees must "
                "also satisfy the return-airfare variance limit and extension rules in the "
                "Expenses and Travel Policy (DOC HR-POL-03, Section 3)."
            ),
        ],
        [
            "5. Sick Leave, Physician Certificates, and Probation Limits",
            (
                "Confirmed shore staff receive 12 paid sick days per calendar year, credited "
                "in full on January 1 and non-cumulative across years."
            ),
            (
                "Single-day or two-day sick absences require notification to the line "
                "manager before 09:30 local time. Consecutive sick absences of 3 or more "
                "working days require a licensed physician fitness-to-work certificate "
                "uploaded to the HR portal within 48 hours of returning."
            ),
            (
                "Employees still within their 90-calendar-day probation period are limited "
                "to a maximum of 2 paid emergency sick days; any sick days taken from the "
                "3rd day onward during probation are unpaid."
            ),
            (
                "More than 6 separate uncertified sick episodes in any rolling 12-month "
                "period triggers a formal occupational attendance review with HR."
            ),
        ],
        [
            "6. Parental, Adoption, and Family Caregiver Leave",
            (
                "Primary caregivers with at least 6 months of continuous shore tenure receive "
                "16 weeks of 100 percent paid parental leave upon birth or legal adoption "
                "placement. Employees with 3 to 6 months tenure receive 8 weeks paid plus "
                "up to 8 weeks unpaid job-protected leave."
            ),
            (
                "Secondary caregivers receive 4 weeks of 100 percent paid parental leave "
                "regardless of gender, split into at most 2 blocks taken within 12 months."
            ),
            (
                "Parental leave requests must be submitted in writing at least 30 calendar "
                "days before the expected start date, except in premature birth emergencies."
            ),
            (
                "Separate from parental leave, shore staff may apply for up to 10 working "
                "days per year of unpaid emergency caregiver leave to care for an immediate "
                "family member with a serious documented health crisis."
            ),
        ],
        [
            "7. Bereavement, Civic Duty, and Certification Study Leave",
            (
                "Emergency bereavement leave grants 5 paid working days without advance "
                "notice for an immediate family member (spouse, partner, child, parent, or "
                "sibling), plus up to 2 extra paid travel days when international travel "
                "exceeding 500 km is required."
            ),
            (
                "Bereavement for extended family (grandparent, grandchild, aunt, uncle, or "
                "parent-in-law) grants 2 paid working days."
            ),
            (
                "Shore staff called for mandatory statutory jury duty or court witness "
                "service receive up to 10 paid working days per calendar year upon "
                "submitting the official court summons within 3 working days of receipt."
            ),
            (
                "Confirmed staff sitting pre-approved maritime logistics or port safety "
                "certification exams receive 3 paid study days per calendar year upon "
                "uploading exam registration."
            ),
        ],
        [
            "8. Public Holidays, Time-Off-in-Lieu (TOIL), and Conduct Link",
            (
                "Shore staff observe the official gazetted public holidays of their assigned "
                "port country (10 to 12 days per year). When a public holiday falls on a "
                "weekend, substitute weekday observance follows the local port calendar."
            ),
            (
                "Salaried exempt shore staff formally directed by a manager to work more "
                "than 4 hours on a weekend vessel casualty or gazetted public holiday earn "
                "1.0 working day of Time-Off-in-Lieu (TOIL), or 0.5 days for 2 to 4 hours."
            ),
            (
                "Accrued TOIL cannot be converted to cash and expires 60 calendar days "
                "after the weekend or holiday worked if unused."
            ),
            (
                "Per DOC HR-POL-02 Section 8, employees under a Tier 2 unpaid administrative "
                "suspension exceeding 5 working days cannot accrue monthly annual leave or "
                "earn TOIL during the suspension period."
            ),
        ],
    ]

    conduct_sections = [
        [
            "SHORE STAFF CODE OF CONDUCT (DOC HR-POL-02, REV 2.0)",
            "",
            "1. Scope, Core Office Hours, and Hybrid Attendance",
            (
                "This Code of Conduct binds all permanent, probationary, and fixed-term "
                "shore staff across every port office. Rules apply equally during office "
                "work, client hospitality, and international business travel."
            ),
            (
                "Core shore-office hours are 09:00 to 17:30 local time, Monday through "
                "Friday, incorporating a mandatory 45-minute unpaid midday meal break. "
                "Teams maintain a compulsory collaboration window from 10:00 to 16:00."
            ),
            (
                "Hybrid shore staff must work physically on-site at their assigned port "
                "office at least 3 days per week, including Tuesday and Wednesday anchor "
                "days, unless granted a written ergonomic accommodation backed by HR."
            ),
            (
                "Unexcused tardiness exceeding 15 minutes more than 3 times in any "
                "30-calendar-day window generates an automatic Tier 1 coaching record."
            ),
        ],
        [
            "2. Conflicts of Interest, Related Parties, and Outside Work",
            (
                "Shore staff must disclose in writing to HR Compliance any direct or "
                "indirect financial interest exceeding 1 percent ownership in a maritime "
                "vendor, shipyard, customs broker, or freight contractor within 10 working "
                "days of hire or acquisition."
            ),
            (
                "If an employee's spouse, partner, or immediate relative works at or holds "
                "decision authority at an active supplier, the employee must recuse "
                "themselves from tender evaluation, purchase orders, and invoice sign-off."
            ),
            (
                "Secondary outside employment, consulting, or paid board service exceeding "
                "8 hours per week—or ANY outside work within maritime or shipping—requires "
                "prior written approval from both the department VP and HR Compliance."
            ),
            (
                "Undisclosed conflicts involving procurement decisions under USD 500 trigger "
                "Tier 2 disciplinary action; conflicts >= USD 500 trigger Tier 3 dismissal."
            ),
        ],
        [
            "3. Vendor Gifts, Hospitality, and Cash Prohibitions",
            (
                "Accepting or soliciting cash, cash equivalents, retail gift cards, pre-paid "
                "debit vouchers, or personal loans of ANY monetary value from a vendor, "
                "candidate, or contractor is strictly prohibited (immediate Tier 3 review)."
            ),
            (
                "Unsolicited non-cash vendor gifts (branded stationery, seasonal hampers, "
                "or promotional merchandise) are capped at USD 50 per single item and "
                "USD 150 cumulative per calendar year from any single supplier group."
            ),
            (
                "Any tangible gift valued above USD 50 must be declined upon delivery or "
                "surrendered physically to HR Compliance within 5 working days for return "
                "to the vendor or placement in the annual charity raffle."
            ),
            (
                "Cross-policy boundary: business hospitality and client meals remain "
                "governed by the Expenses and Travel Policy (DOC HR-POL-03, Section 6), "
                "which caps business meals at USD 75 per attendee, and vendor-hosted meals "
                "must never circumvent the USD 50 single-gift / USD 150 annual vendor cap."
            ),
        ],
        [
            "4. Anti-Bribery, Customs Integrity, and Facilitation Payments",
            (
                "Company policy strictly forbids offering, authorizing, or paying bribes, "
                "kickbacks, or unofficial cash facilitation ('grease') payments to customs "
                "officers, port inspectors, surveyors, or officials in any jurisdiction."
            ),
            (
                "Where local agents demand an unreceipted expedition fee to clear vessel "
                "spares or crew papers, shore staff must refuse and escalate immediately "
                "to Port Legal and HR Compliance."
            ),
            (
                "In rare cases where a cash payment is extorted under imminent physical "
                "safety or vessel detention threat, the incident must be reported to Legal "
                "and HR Compliance within 24 hours and logged accurately in expense filings "
                "(DOC HR-POL-03, Section 5) rather than disguised as miscellaneous freight."
            ),
        ],
        [
            "5. Information Security, Confidentiality, and Device Rules",
            (
                "Crew manifests, vessel live routing schedules, port gate access codes, "
                "supplier tender bids, and shore-staff personnel records are Confidential."
            ),
            (
                "Confidential files must never be uploaded to unapproved public websites, "
                "personal cloud drives, or consumer generative AI tools, nor forwarded to "
                "personal email addresses."
            ),
            (
                "Company laptops and tablets must enforce automatic screen lock after "
                "5 minutes of inactivity and full-disk encryption. Loss or theft of any "
                "work device or port badge must be reported to IT and HR within 2 hours."
            ),
            (
                "Clean-desk rules apply outside core hours: physical manifests, bid "
                "spreadsheets, and security passes must be locked in pedestals overnight."
            ),
        ],
        [
            "6. Respectful Workplace and Quayside Safety Compliance",
            (
                "The company maintains zero tolerance for harassment, discrimination, "
                "bullying, or demeaning conduct in offices, vessels, or digital channels."
            ),
            (
                "Shore staff visiting quayside terminals or dry-dock berths must complete "
                "the 2-hour terminal safety induction and wear mandatory PPE: ISO high-"
                "visibility vest, steel-toe safety footwear, and hard hat."
            ),
            (
                "A first non-willful quayside PPE lapse requires immediate berth exit and "
                "triggers a Tier 1 written coaching notice. A deliberate bypass or repeat "
                "infraction within 6 months escalates directly to Tier 2 suspension."
            ),
        ],
        [
            "7. Confidential Whistleblowing and Non-Retaliation Guarantee",
            (
                "Employees suspecting fraud, safety cover-ups, bribery, harassment, or "
                "policy breaches may report 24/7 via the ethics hotline or HR Compliance; "
                "anonymous submissions are accepted and investigated like named reports."
            ),
            (
                "HR Compliance acknowledges receipt of non-anonymous reports within 3 "
                "working days and completes triage and investigator assignment within 10 days."
            ),
            (
                "Retaliation, demotion, schedule penalty, or intimidation directed at any "
                "good-faith reporter or witness is itself an automatic Tier 3 summary "
                "termination offense regardless of whether the report is substantiated."
            ),
        ],
        [
            "8. Disciplinary Escalation Tiers and Cross-Policy Sanctions",
            (
                "Tier 1 (Verbal or Written Coaching): Applies to minor attendance lapses, "
                "incomplete timesheets, or first non-willful documentation errors. Remains "
                "active on HR file for 6 months and clears if no repeat infraction occurs."
            ),
            (
                "Tier 2 (Formal Written Warning and Administrative Suspension): Applies to "
                "repeated Tier 1 violations within 6 months, undisclosed vendor conflicts "
                "under USD 500, or repeat quayside PPE breaches. Active 12 months and "
                "carries up to 10 working days of unpaid administrative suspension."
            ),
            (
                "Cross-policy sanctions on Tier 2: Under the Leave Policy (DOC HR-POL-01, "
                "Section 2), unpaid administrative suspensions exceeding 5 working days "
                "pause monthly annual leave accrual. Under the Expenses and Travel Policy "
                "(DOC HR-POL-03, Section 8), employees under an active Tier 2 review have "
                "all pending and new expense payouts frozen until HR Compliance clears them."
            ),
            (
                "Tier 3 (Summary Termination): Immediate dismissal without notice or "
                "severance for fraud, falsified receipts, bribery, data exfiltration, or "
                "whistleblower retaliation. Accrued unused leave follows DOC HR-POL-01 Sec 3."
            ),
        ],
    ]

    expenses_sections = [
        [
            "SHORE STAFF EXPENSES AND TRAVEL POLICY (DOC HR-POL-03, REV 2.0)",
            "",
            "1. Travel Pre-Approval and Probation Restrictions",
            (
                "All domestic and regional business travel requires line-manager portal "
                "pre-approval at least 7 calendar days prior to departure. Intercontinental "
                "trips with flights exceeding 6 hours require approval >= 14 days in advance."
            ),
            (
                "Emergency vessel-breakdown dispatches waive advance windows but require "
                "written VP ratification within 24 hours of ticket issuance."
            ),
            (
                "Cross-policy probation rule: Employees within their 90-calendar-day "
                "probation period (DOC HR-POL-01, Section 1) require written Director-level "
                "pre-approval for ANY business travel and are capped at 75 percent of "
                "standard hotel nightly limits."
            ),
        ],
        [
            "2. Flight Cabin Classes, Rail Preference, and Ancillaries",
            (
                "Scheduled single-leg flights under 6 hours duration must book Standard "
                "Economy class. Lowest logical fares are mandatory unless client shifts "
                "require flexible rebooking."
            ),
            (
                "Scheduled single-leg flights of 6 hours or longer qualify for Premium "
                "Economy class. Business Class requires Senior Vice President (SVP) sign-off "
                "and is limited to single sectors >= 10 hours with immediate work duty."
            ),
            (
                "Rail is mandatory instead of short-haul flights under 4 hours door-to-door; "
                "2nd Class applies under 3 hours, while 1st Class rail is permitted on "
                ">= 3-hour routes when priced below an Economy airfare."
            ),
            (
                "Paid seat-selection fees above USD 45, priority boarding, and lounge day "
                "passes (unless layover exceeds 4 consecutive hours) are non-reimbursable."
            ),
        ],
        [
            "3. Combining Business Travel with Personal Annual Leave",
            (
                "Shore staff may extend a business trip with personal annual leave before "
                "or after the work itinerary only with prior manager sign-off in the portal."
            ),
            (
                "Return economy airfare remains company-reimbursable ONLY when the personal "
                "leave extension does not exceed 5 consecutive working days AND adjusted "
                "return airfare does not exceed the business-only fare quote by > USD 150."
            ),
            (
                "Cross-policy rule: Personal leave extensions exceeding 5 consecutive "
                "working days trigger the Leave Policy 14-calendar-day advance notice and "
                "secondary VP sign-off (DOC HR-POL-01, Section 4) and shift any return "
                "airfare variance exceeding USD 150 completely to personal cost."
            ),
            (
                "Hotel accommodation, per diem allowances, and local ground transport on "
                "personal leave days are strictly excluded from reimbursement."
            ),
        ],
        [
            "4. Lodging Caps, Probation Adjustment, and Ground Transport",
            (
                "Standard hotel nightly caps (excluding local occupancy taxes) are USD 180 "
                "per night in Tier A major hubs (Rotterdam, Singapore, Hamburg, London, "
                "Tokyo) and USD 135 per night in Tier B regional offices (including Busan)."
            ),
            (
                "Under the 75 percent probation cap (Section 1), probationary staff are "
                "limited to USD 135.00 per night in Tier A hubs and USD 101.25 in Tier B."
            ),
            (
                "When host hotels exceed city caps by > 15 percent, advance written manager "
                "approval is required. Private short-term room rentals are not reimbursed."
            ),
            (
                "Personal vehicle mileage for approved port errands is reimbursed at USD 0.62 "
                "per kilometer (capped at 250 km round trip per journey). Rental cars are "
                "restricted to compact or intermediate classes with standard collision damage "
                "waiver (CDW); fuel must be refilled using regular receipts (no prepay plans)."
            ),
        ],
        [
            "5. Daily Per Diem, Receipt Thresholds, and Exclusions",
            (
                "Daily meal per diem during overnight travel is USD 60 per full calendar "
                "travel day. Partial departure and return travel days under 12 hours away "
                "from home port are capped at 75 percent (USD 45)."
            ),
            (
                "Original itemized merchant receipts showing date, vendor, and line items "
                "are mandatory for EVERY single expense of USD 25 or more (card slips alone "
                "are rejected). Missing-receipt forms up to USD 40 allowed twice/quarter."
            ),
            (
                "Strictly non-reimbursable items: hotel minibar charges, movies, spa fees, "
                "traffic or parking fines, personal grooming, and > 2 alcoholic drinks/meal."
            ),
            (
                "Checked baggage fees beyond 1 bag are non-reimbursable unless carrying "
                "calibrated vessel inspection equipment exceeding 23 kg."
            ),
        ],
        [
            "6. Client Meals and Gift-Policy Alignment",
            (
                "Client and port-partner business meals are capped at USD 75 per attendee "
                "(before mandatory tax and up to 15 percent gratuity) and require listing "
                "every attendee's full name, title, company, and meeting agenda."
            ),
            (
                "Alcohol must never exceed 35 percent of a client hospitality bill; bar-only "
                "receipts without substantial food service are rejected."
            ),
            (
                "Cross-policy boundary: Client meals, vendor lunches, and event invitations "
                "remain strictly subject to the USD 50 single-item and USD 150 annual vendor "
                "gift caps under the Code of Conduct (DOC HR-POL-02, Section 3)."
            ),
        ],
        [
            "7. Corporate Travel Cards and Cash Advances",
            (
                "All confirmed staff traveling more than twice per year must use the issued "
                "corporate travel card for airfare, rail, and hotel charges above USD 200. "
                "Personal card loyalty points do not exempt corporate-card compliance."
            ),
            (
                "Emergency cash advances up to USD 500 per trip are permitted solely for "
                "ports lacking card acceptance and must reconcile within 5 working days."
            ),
            (
                "Any cash advance remaining unsettled more than 15 calendar days after "
                "return suspends corporate card privileges and triggers automatic "
                "advance-balance settlement upon separation."
            ),
        ],
        [
            "8. Submission Deadlines, Audits, and Tier 2 Reimbursement Holds",
            (
                "All business expense claims must be submitted in the expense portal within "
                "14 calendar days of trip completion."
            ),
            (
                "Claims submitted 15 to 30 days late require Department Head exception "
                "sign-off. Claims > 30 calendar days late are forfeited unless Finance "
                "Controller approves an exceptional written waiver."
            ),
            (
                "Finance runs automated policy checks plus random audit sampling on 15 "
                "percent of all expense claims exceeding USD 500."
            ),
            (
                "Cross-policy hold: Under the Code of Conduct (DOC HR-POL-02, Section 8), "
                "employees under active Tier 2 disciplinary review have ALL pending and new "
                "expense payouts frozen until HR Compliance issues clearance. Falsified "
                "receipts trigger immediate Tier 3 summary termination."
            ),
        ],
    ]

    write_pdf(OUT_DIR / "leave_policy.pdf", leave_sections)
    write_pdf(OUT_DIR / "code_of_conduct.pdf", conduct_sections)
    write_pdf(OUT_DIR / "expenses_and_travel_policy.pdf", expenses_sections)
    print("Generated 3 expanded HR policy PDFs (v2) in", OUT_DIR)


if __name__ == "__main__":
    main()
