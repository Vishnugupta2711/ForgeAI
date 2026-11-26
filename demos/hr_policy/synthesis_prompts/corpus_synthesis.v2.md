# Synthesis Prompt v2 — HR Policy Demo Corpus (`demos/hr_policy`)

Generate 3 realistic, internally consistent corporate HR policy documents for shore staff (`leave_policy.pdf`, `code_of_conduct.pdf`, `expenses_and_travel_policy.pdf`), each roughly 5,500–7,500 characters across 8 detailed sections so RAG ingestion yields substantial chunk diversity (~45–60 chunks total) for evaluation and optimization.

Requirements:
1. Each PDF must have 8 numbered sections with concrete numeric rules, tiered thresholds, exceptions, port/office edge cases, and explicit cross-document references (`HR-POL-01`, `HR-POL-02`, `HR-POL-03`).
2. Preserve and expand the 4 cross-document rules that only emerge by combining two policies:
   - Business trip leave extensions (>5 consecutive working days trigger 14-day advance notice & VP sign-off in Leave Policy `HR-POL-01` Sec 4, plus <=USD 150 return-airfare variance limit in Expenses Policy `HR-POL-03` Sec 3).
   - Shore-staff 90-day probation rules (no annual leave / max 2 paid emergency sick days in `HR-POL-01` Sec 1 & 5 + Director travel pre-approval and 75% hotel cap in `HR-POL-03` Sec 1 & 4).
   - Tier 2 disciplinary suspension (>5 unpaid working days pauses monthly leave accrual in `HR-POL-01` Sec 2 & 8 and freezes expense claim payouts until HR Compliance clearance in `HR-POL-03` Sec 8).
   - Vendor gift vs client hospitality boundary (USD 50 single-item / USD 150 annual non-cash gift cap and zero cash equivalents in `HR-POL-02` Sec 3 vs USD 75/attendee client meal cap in `HR-POL-03` Sec 6).
3. Deliberately omit ALL content on employee medical insurance claim adjudication and statutory payroll tax withholding tables (and avoid ambiguous overlap terms) so the Groundability Guard catches those unsupported brief topics cleanly.
4. Keep PDF generation 100% reproducible from `build_pdfs.py` with automatic page wrapping (zero clipped lines).
