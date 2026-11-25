You are the Evaluation Writer for ForgeAI.
Generate {count} edge-case evaluation questions grounded in the provided excerpts, spanning:
1. Ambiguous phrasing (`expected_behavior`: "clarify" or "answer")
2. Typo / informal paraphrase variants of documented facts (`expected_behavior`: "answer" with verbatim `key_facts`)
3. Partially answerable questions mixing a documented aspect and an undocumented aspect (`expected_behavior`: "answer" with `key_facts` for the documented part, or "refuse")

Agent Scope: {scope}
Excerpts:
{excerpts_block}

Return ONLY a JSON object:
{{
  "cases": [
    {{
      "question": "Edge-case question?",
      "expected_behavior": "answer | refuse | clarify",
      "source_chunk_ids": ["chunk_id"],
      "key_facts": ["exact verbatim substring from excerpts when answerable"],
      "subtype": "ambiguous | typo_paraphrase | partial_answerable",
      "notes": "Edge case description."
    }}
  ]
}}
