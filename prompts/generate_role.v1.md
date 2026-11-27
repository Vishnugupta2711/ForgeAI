Generate the `role` section of a system prompt for a grounded assistant.
Stay strictly within the supported topics, audience, and narrowed scope. Never claim to perform excluded, unsupported, or must-refuse capabilities even if the raw brief mentioned them.

Brief: {brief}
Narrowed Scope: {scope}
Supported Must-Answer Topics:
{must_answer_block}
Excluded / Must-Refuse Topics:
{must_refuse_block}
Audience: {audience}
Tone: {tone}

Return ONLY a JSON object:
{{"role": "2-4 sentences defining who the assistant is and its grounded responsibilities only."}}
