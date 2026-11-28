Verify whether EACH numbered claim below is strictly supported by the provided retrieved passages and tool outputs.

CRITICAL RULES:
1. Rely ONLY on the retrieved passages and tool outputs below. Never use outside knowledge.
2. For each claim, set `supported` to true ONLY if the evidence explicitly states that fact.
3. Whenever `supported` is true, you MUST copy an exact, verbatim substring `quote` from the passages or tool outputs that proves the claim.
4. If `supported` is false, explain specifically why in `reason` and, when a passage contradicts or limits the claim, copy the relevant verbatim substring in `quote`.
5. Score each claim independently; do not reward length or style.

Claims to verify:
{claims_block}

Evidence (Retrieved Passages & Tool Outputs):
{evidence_block}

Return ONLY a JSON object:
{{
  "verdicts": [
    {{
      "claim_index": 1,
      "supported": true,
      "quote": "exact verbatim substring from evidence",
      "reason": "brief verification or contradiction reason"
    }}
  ]
}}
