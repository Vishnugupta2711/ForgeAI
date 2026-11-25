Propose which tools from the registry should be attached for this agent.
Propose a tool ONLY if the requirements explicitly need it and the corpus can support it.

Brief: {brief}
Capabilities needed:
{capabilities_block}

Corpus profile:
{corpus_profile}

Available registry tools:
{tools_block}

Return ONLY a JSON object:
{{
  "proposed_tools": [
    {{
      "name": "tool_name",
      "reason": "Why it is required by a specific REQ ID."
    }}
  ]
}}
