Generate the `refusal_policy` section of a system prompt.
It MUST explicitly list the must-refuse topics AND any flagged groundability gaps, instructing the assistant to refuse honestly (e.g. stating that the documents don't cover this) rather than guessing.

Must-refuse topics:
{must_refuse_block}

Groundability gaps:
{gaps_block}

Return ONLY a JSON object:
{{"refusal_policy": "Clear refusal rules naming excluded/ungrounded topics and requiring honest refusal when the documents don't cover a request."}}
