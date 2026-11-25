# Non-Negotiable Engineering Rules

These rules apply to every phase of ForgeAI development:

1. **No domain-specific code in `src/`.**
   Words like `HR`, `ship`, `supplier`, or `leave` may appear only in `tests/`, `fixtures/`, `demos/`, and `docs/`. The factory must work on unseen briefs without domain assumptions.

2. **All LLM calls go through `forge.llm.LLMClient`.**
   `LLMClient` must handle JSON mode, Pydantic schema validation, repair-retry, disk caching keyed by prompt hash, and token/latency logging. Unit tests must use a `FakeLLM` and never require a real model.

3. **Never fabricate metrics.**
   Every number in any report must come directly from a real run's result files. Demo artifacts are never hand-edited or cherry-picked.

4. **Strict per-phase workflow.**
   Every phase must have passing tests (`pytest -q`), clean lint (`ruff check .`), an updated `docs/PROGRESS.md`, and be committed on a phase branch (`phase-N-<name>`), merged to `main` with `--no-ff`, tagged (`phase-N`), and pushed.

5. **Design for small, imperfect open models.**
   Assume malformed JSON, flaky tool calls, and instruction drift are the normal case. Handle validation, repair-retry, and deterministic fallbacks in code.

6. **Prompts live in `prompts/`.**
   All LLM prompts are stored as versioned `.md` files under `prompts/`, never as inline prompt strings in `src/`.

7. **No secrets in Git.**
   Never commit `.env`, API keys, or credentials. Keep `.env`, caches, and `.venv/` in `.gitignore`.
