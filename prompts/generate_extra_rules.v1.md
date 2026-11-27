Generate concise `extra_rules` bullets (at most 5) for edge cases, tool usage discipline, and supported assumptions.
Do NOT write any rule telling the assistant to handle excluded or must-refuse topics.

Assumptions:
{assumptions_block}

Attached tools: {attached_tools}
Supported Must-Answer Topics:
{must_answer_block}
Excluded / Must-Refuse Topics:
{must_refuse_block}

Return ONLY a JSON object:
{{"extra_rules": ["Rule 1", "Rule 2"]}}
