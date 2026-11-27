You are simulating a pragmatic requester answering clarifying intake questions for an automated run.
Answer ONLY using facts grounded in the brief and the corpus profile below.
If a question asks for something unspecified in the brief or corpus profile, make a conservative, explicit assumption (for example, defaulting to strict refusal for unsupported requests) and state it concisely.
Do NOT exclude document sections present in the corpus profile that belong to the requested topic; assume all relevant documented sections are in scope.
Never invent document contents that are not in the corpus profile.

Brief:
{brief}

Corpus Profile:
{corpus_profile}

Questions:
{questions_block}

Return ONLY a JSON object:
{{
  "answers": [
    {{
      "question_id": "Q1",
      "answer": "Concise answer grounded in the brief/profile or explicit conservative assumption."
    }}
  ]
}}
