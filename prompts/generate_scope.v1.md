Generate the `scope` section of a system prompt listing exactly what topics are in scope and what is excluded.

Scope summary: {scope}
Must-answer topics:
{must_answer_block}

Must-refuse topics:
{must_refuse_block}

Return ONLY a JSON object:
{{"scope": "Clear operational scope statement covering the in-scope topics and boundaries."}}
