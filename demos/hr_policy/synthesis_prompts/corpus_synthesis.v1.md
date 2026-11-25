# Synthesis Prompt — HR Policy Demo Corpus (`demos/hr_policy`)

Generate 3 realistic, internally consistent corporate HR policy documents for shore staff (`leave_policy.pdf`, `code_of_conduct.pdf`, `expenses_and_travel_policy.pdf`).

Requirements:
1. Include concrete numeric rules, caps, eligibility windows, and exceptions.
2. Include at least 3 cross-document facts that only emerge by combining two policies:
   - Business trip leave extensions (>5 consecutive working days trigger 14-day advance notice & VP sign-off from Leave Policy, plus return-airfare eligibility rules from Expenses Policy).
   - Shore-staff 90-day probation rules (no annual leave / 2 emergency sick days in Leave Policy + 75% hotel cap & Director pre-approval in Expenses Policy).
   - Tier 2 disciplinary suspension (>5 unpaid working days pauses 1.75 days/month leave accrual in Leave Policy and holds expense claim payouts in Expenses Policy).
   - Gift vs client meal compliance boundary ($50/$150 gift caps in Code of Conduct vs $75/person client meal cap in Expenses Policy).
3. Deliberately omit medical insurance claim adjudication and statutory payroll tax withholding tables so the Groundability Guard catches those gaps in `brief.txt`.
