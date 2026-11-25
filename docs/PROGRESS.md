# ForgeAI Build Progress

## Non-Negotiable Engineering Rules (Checklist)

- **No domain-specific code in `src/`**: Domain terms (`HR`, `ship`, `supplier`, `leave`, etc.) belong only in `tests/`, `fixtures/`, `demos/`, and `docs/`.
- **Single LLM boundary (`forge.llm.LLMClient`)**: JSON mode, schema validation, repair-retry, prompt-hash disk cache, token/latency logging; unit tests use `FakeLLM`.
- **Never fabricate metrics**: Report numbers come strictly from real run artifacts; demo artifacts are never hand-edited.
- **Per-phase gate**: `pytest -q` passes, `ruff check .` is clean, `docs/PROGRESS.md` is updated, committed on `phase-N-<name>`, merged to `main` with `--no-ff`, tagged `phase-N`, and pushed.
- **Small imperfect models assumed**: Malformed JSON and instruction drift are handled deterministically in code.

---

## Phase Status

### Phase 0 — Bootstrap (`phase-0-bootstrap`)
- **Status**: Complete
- **Deliverables**:
  - Added `docs/ASSIGNMENT.md`, `docs/RULES.md`, `docs/PROGRESS.md`
  - Created `pyproject.toml` (`requires-python >= 3.11`, `forge` CLI entry point via Typer)
  - Created `src/forge/` package skeleton across all 13 subpackages + `cli.py`
  - Created `prompts/`, `tests/`, `fixtures/`, `demos/`, `runs/`, `forge.toml`, `.env.example`, `.gitignore`
  - Added CLI smoke test (`tests/test_smoke.py`) and GitHub Actions CI (`.github/workflows/ci.yml`)
- **Verification**:
  - `pip install -e .`, `forge --help`, `pytest -q`, and `ruff check .` passed in `.venv` (Python 3.11).
