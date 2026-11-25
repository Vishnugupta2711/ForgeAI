Determine whether the candidate out-of-scope question IS actually answerable from the retrieved document passages below.
If the passages DO cover and answer the question, set `answerable_from_docs` to true (so the candidate is rejected as a false out-of-scope case).
If the passages do NOT answer it (or it is a prompt-injection / off-topic / unsupported request), set `answerable_from_docs` to false.

Question: {question}

Retrieved Passages:
{passages_block}

Return ONLY a JSON object:
{{
  "answerable_from_docs": false,
  "reason": "Brief explanation."
}}
