You are writing the "What Still Fails and Why" diagnosis narrative for a ForgeAI quality report.

STRICT RULES:
1. Every statement MUST cite the exact case ID(s) from the failing cases list below (e.g. `DEV-003`, `HELD-002`).
2. NEVER invent any numbers, percentages, counts, or scores that do not literally appear in the Allowed Numbers or Result Summary below. If unsure, describe patterns qualitatively while citing exact case IDs.
3. Be honest and specific about failure causes based strictly on the reasons below.

Result Summary:
{summary_block}

Allowed Numbers (only numbers appearing in result files):
{allowed_numbers}

Remaining Failing Cases:
{failures_block}

Return ONLY a JSON object:
{{
  "diagnosis_items": [
    {{
      "case_ids": ["HELD-001"],
      "diagnosis": "Concise explanation citing exact case ID(s) and only verified numbers."
    }}
  ],
  "overall_summary": "1-2 sentence honest summary citing remaining failing case IDs."
}}
