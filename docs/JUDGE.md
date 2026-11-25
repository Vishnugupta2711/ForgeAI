# ForgeAI Judge Architecture & Bias Defenses

## Overview
The ForgeAI Judge (`forge.judge`) evaluates agent answers using a two-layer pipeline:
1. **Deterministic Layer (authoritative where applicable)**: checks contract refusal alignment, citation validity (cited `doc_id` and substring `quote` against chunks actually retrieved by the agent), canonicalized key-fact recall (numbers, units, dates, currency, and fuzzy text), and numeric hallucination detection.
2. **LLM Layer (narrowly scoped)**: used only for atomic claim-level groundedness verification (when deterministic checks pass) and ambiguous key-fact correctness.

## Known LLM-as-Judge Failure Modes & Defenses

### 1. Position Bias
- **Risk**: Judges systematically favor items appearing earlier or later in a list.
- **Defense**: Claim verification and ambiguous key-fact correctness are executed **twice** — once in forward order and once in shuffled/reversed order. If the two passes disagree on any verdict, the case is flagged `uncertain = True` and the judge falls back deterministically to the code checks (`citations_valid` and `ungrounded_numbers` for groundedness; `key_fact_recall >= 0.5` for correctness).

### 2. Self-Preference Bias
- **Risk**: Evaluating an agent with the same model (or model family) inflates scores because the judge favors its own phrasing patterns.
- **Defense**: `forge.toml` configures a distinct model family for the judge (`llama3.1:8b`) vs. the agent (`qwen2.5:3b-instruct`). `EvalScorecard` explicitly records `same_model_as_agent` and `same_family_as_agent`; if they match, the scorecard flags a warning that reports must surface.

### 3. Verbosity Bias
- **Risk**: Holistic scoring rewards long, padded responses over concise accurate ones.
- **Defense**:
  - The judge never assigns a holistic 1–5 quality score. Instead, it decomposes answers into atomic factual claims, scores each claim binary (`supported` / `unsupported`) with substring-verified quotes, and marks the answer ungrounded if any claim lacks verified support.
  - Every scorecard computes and reports `length_pass_correlation` (Pearson $r$ between answer character length and pass/fail) so verbosity bias is directly observable.

### 4. Fabricated Judge Quotes & Statistical Over-Confidence
- **Quote Verification**: Every quote returned by the LLM judge to justify claim support is checked in code as an actual substring of the retrieved chunks or tool outputs; unverified quotes are rejected.
- **Wilson 95% Intervals**: Because evaluation splits per category are small, every category pass rate reports a Wilson score 95% confidence interval (`wilson_low`, `wilson_high`) alongside a finite-sample disclaimer.
- **Suspicion Rule**: A 100% pass rate automatically sets `suspicion_flag = True` warning that the evaluation set may be too easy or overfit.
- **Canary Self-Test (`forge judge-selftest`)**: Measures false-pass and false-fail rates across 8 synthetic canary categories (`gold_answer`, `numerically_corrupted`, `hallucinated_claim`, `wrong_document`, `padded_verbose`, `terse_correct`, `correct_refusal`, `wrongful_refusal`) and writes `judge_reliability.json`.
