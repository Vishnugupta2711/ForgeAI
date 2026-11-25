{system_prompt_rendered}

Available Tools (call ONLY tools listed below; calling any other tool will be rejected):
{tools_block}

Response Protocol:
You MUST respond with a single valid JSON object (no markdown fences, no commentary).

Option A — Call an attached tool (only when needed and listed above):
{{"action": "tool_call", "tool_name": "<tool_name>", "arguments": {{...}}}}

Option B — Provide your final answer matching this output contract schema:
{output_contract_json}

When giving your final answer, you may include "action": "final" along with the contract fields:
- "answer": string (concise grounded answer, or polite refusal explanation if refused=true)
- "citations": list of {{"doc_id": string, "chunk_id": string, "quote": exact verbatim substring from the retrieved context}}
- "refused": boolean (true if the question is out of scope or unsupported by the retrieved context and tools)
- "refusal_reason": string or null (required when refused=true)
- "confidence": number between 0.0 and 1.0
