Generate the `format_rules` section of a system prompt aligned with the output contract.

Output contract summary:
- answer (string)
- citations (list of {{doc_id, chunk_id, quote}}; required whenever refused is false)
- refused (boolean)
- refusal_reason (string or null; required when refused is true)
- confidence (float between 0.0 and 1.0)

Attached tools: {attached_tools}

Return ONLY a JSON object:
{{"format_rules": "Clear instructions on formatting answers, citations, and refusal fields to match the output contract."}}
