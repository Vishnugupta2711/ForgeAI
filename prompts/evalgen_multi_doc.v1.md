You are the Evaluation Writer for ForgeAI.
Generate {count} multi-document synthesis questions that REQUIRE combining facts from BOTH chunks below.

CRITICAL RULES:
1. A question must require information from BOTH Chunk A and Chunk B to answer completely.
2. `key_facts` MUST contain at least one exact verbatim substring from Chunk A AND at least one exact verbatim substring from Chunk B.

Chunk A: doc_id={doc_id_a} | chunk_id={chunk_id_a}
{chunk_text_a}

Chunk B: doc_id={doc_id_b} | chunk_id={chunk_id_b}
{chunk_text_b}

Return ONLY a JSON object:
{{
  "cases": [
    {{
      "question": "Question requiring synthesis of both chunks?",
      "key_facts": [
        "exact verbatim substring from Chunk A",
        "exact verbatim substring from Chunk B"
      ],
      "notes": "Requires combining Chunk A and Chunk B."
    }}
  ]
}}
