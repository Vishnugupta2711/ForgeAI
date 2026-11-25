Classify this failing evaluation case into EXACTLY ONE primary failure category:
- `retrieval_miss`: gold source document/chunk was missing from retrieved passages
- `hallucination`: answer made unsupported claims, invalid citation quotes, or ungrounded numbers
- `wrong_refusal`: refused an answerable question OR answered an out-of-scope question that should be refused
- `prompt_defect`: right context was retrieved, refusal behavior matched, but answer failed instructions or key facts

Case ID: {case_id}
Question: {question}
Expected Behavior: {expected_behavior}
Verdict Summary: grounded={grounded}, correct={correct}, refusal_ok={refusal_ok}, recall={key_fact_recall}
Reasons: {reasons}
Retrieved Chunks: {retrieved_chunks}
Expected Sources: {expected_sources}

Return ONLY a JSON object:
{{
  "failure_category": "retrieval_miss | hallucination | wrong_refusal | prompt_defect",
  "subtype": "short_subtype_label",
  "evidence": "Concise evidence explaining why this is the primary category."
}}
