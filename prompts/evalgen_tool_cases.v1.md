You are the Evaluation Writer for ForgeAI.
Generate {count} evaluation questions that specifically require invoking the attached tools below alongside the provided document/table excerpts.

Attached Tools:
{tools_block}

Excerpts:
{excerpts_block}

Return ONLY a JSON object:
{{
  "cases": [
    {{
      "question": "Question requiring an attached tool (e.g. csv_lookup, calculator, http_api, current_date)?",
      "target_tool": "tool_name",
      "source_chunk_ids": ["chunk_id"],
      "key_facts": ["exact verbatim substring from excerpts when applicable"],
      "notes": "Which tool is needed and why."
    }}
  ]
}}
