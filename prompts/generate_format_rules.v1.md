Generate the `format_rules` section of a system prompt aligned with the output contract.

Output contract summary:
- answer (string)
- citations (list of objects, each MUST have all 3 keys: {{"doc_id", "chunk_id", "quote"}} where "quote" is an exact verbatim substring from retrieved context; required whenever refused is false)
- refused (boolean)
- refusal_reason (string or null; required when refused is true)
- confidence (float between 0.0 and 1.0)

Attached tools: {attached_tools}

Return ONLY a JSON object:
{{"format_rules": "Clear instructions specifying answer, citations (explicitly listing doc_id, chunk_id, and verbatim quote), refused, refusal_reason, and confidence."}}
