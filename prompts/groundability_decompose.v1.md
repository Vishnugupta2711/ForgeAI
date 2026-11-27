Decompose the brief, requirements, and clarifying Q&A into at most 8-10 concise atomic testable topics and implicit capability needs.
Include every explicit must-answer topic AND any distinct implicit or unsupported request mentioned in the brief (such as calculations, tabular filtering, or external lookups), keeping each description to one concise sentence instead of splitting every minor sub-clause into separate topics.

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
