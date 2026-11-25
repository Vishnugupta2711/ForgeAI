Generate the `audience_tone` section of a system prompt specifying the target audience and response style.

Audience: {audience}
Desired tone: {tone}
Assumptions:
{assumptions_block}

Return ONLY a JSON object:
{{"audience_tone": "Concise instructions describing the target audience, clarity level, and tone."}}
