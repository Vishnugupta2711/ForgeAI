You are synthesizing a structured requirements specification from a brief, corpus profile, and clarifying Q&A.
Be domain-neutral and base every topic strictly on the brief, corpus profile, and Q&A.

Standard capability names you may reference in capabilities_needed when truly required by the brief:
- tabular_lookup (filtering/aggregating structured CSV tables)
- arithmetic (numeric calculations beyond plain lookup)
- live_external_data (live external HTTP data such as weather forecasts)
- current_date (knowing today's date or relative time windows)

Brief:
{brief}

Corpus Profile:
{corpus_profile}

Clarifying Q&A:
{qa_block}

Return ONLY a JSON object matching:
{{
  "scope": "Clear statement of what the agent covers.",
  "audience": "Target end users.",
  "tone": "Desired response tone and style.",
  "must_answer_topics": [
    {{
      "id": "REQ-001",
      "title": "Short topic title",
      "description": "Specific topic or question class the agent must answer.",
      "priority": "high"
    }}
  ],
  "must_refuse_topics": [
    {{
      "id": "REF-001",
      "title": "Out-of-scope topic",
      "reason": "Why it must be refused."
    }}
  ],
  "assumptions": ["Any implicit or auto-mode assumptions."],
  "capabilities_needed": [
    {{
      "name": "tabular_lookup",
      "reason": "Why this capability is needed.",
      "linked_req_ids": ["REQ-001"],
      "capability_tag": "csv"
    }}
  ]
}}
