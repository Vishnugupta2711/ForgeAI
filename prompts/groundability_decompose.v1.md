Decompose the brief, requirements, and clarifying Q&A into atomic testable topics and implicit capability needs.
Include every explicit must-answer topic AND any implicit expectations (such as calculations, tabular filtering, live external lookups, or unsupported requests mentioned in the brief).

Brief:
{brief}

Clarifying Q&A:
{qa_block}

Existing Must-Answer Topics:
{topics_block}

Return ONLY a JSON object:
{{
  "topics": [
    {{
      "topic_id": "TOP-001",
      "title": "Short atomic topic name",
      "description": "Concrete description of what the agent would need to answer.",
      "source_req_id": "REQ-001",
      "is_implicit": false,
      "capability_hint": null
    }}
  ]
}}
