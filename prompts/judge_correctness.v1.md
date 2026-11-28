Determine whether the assistant's answer correctly covers the expected key facts for the user's question.
Do NOT reward verbosity or style; judge purely on factual accuracy against the expected key facts.

CRITICAL RULES:
1. If any expected key fact listed under Missing Expected Key Facts is omitted from the answer, you MUST set `correct` to false.
2. You may set `correct` to true when key facts failed verbatim matching ONLY if you quote the exact verbatim substring `answer_quote` from Assistant Answer that expresses EACH missing fact in another surface form with exact matching numbers.

Question: {question}

Expected Key Facts:
{key_facts_block}

Missing Expected Key Facts (unmatched verbatim):
{missing_facts_block}

Assistant Answer:
{answer}

Return ONLY a JSON object:
{{
  "correct": true,
  "missing_fact_quotes": [
    {{
      "fact": "exact missing key fact string",
      "answer_quote": "exact verbatim substring from Assistant Answer proving this fact"
    }}
  ],
  "reason": "Brief explanation referencing the expected key facts."
}}
