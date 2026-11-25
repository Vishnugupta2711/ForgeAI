Generate the `role` section of a system prompt for a grounded assistant.
Stay strictly within the provided brief, audience, and scope. Do not invent capabilities.

Brief: {brief}
Scope: {scope}
Audience: {audience}
Tone: {tone}

Return ONLY a JSON object:
{{"role": "2-4 sentences defining who the assistant is and its primary responsibility."}}
