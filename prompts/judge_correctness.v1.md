Determine whether the assistant's answer correctly covers the expected key facts for the user's question.
Do NOT reward verbosity or style; judge purely on factual accuracy against the expected key facts.

Question: {question}

Expected Key Facts:
{key_facts_block}

Assistant Answer:
{answer}

Return ONLY a JSON object:
{{
  "correct": true,
  "reason": "Brief explanation referencing the expected key facts."
}}
