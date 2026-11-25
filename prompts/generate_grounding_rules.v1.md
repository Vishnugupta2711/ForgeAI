Generate the `grounding_rules` section of a system prompt.
It MUST require citing quoted evidence verbatim from the retrieved document chunks (doc_id, chunk_id, quote) for every factual claim, and prohibit fabricating facts or relying on unverified outside knowledge.

Attached tools: {attached_tools}
Corpus summary: {corpus_profile}

Return ONLY a JSON object:
{{"grounding_rules": "Explicit grounding rules requiring verbatim quoted evidence citations."}}
