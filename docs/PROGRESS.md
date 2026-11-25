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

### Phase 1 — Foundation (`phase-1-foundation`)
- **Status**: Complete
- **Deliverables**:
  - `forge.config`: `forge.toml` and `.env` configuration loader + `load_prompt()` helper (`prompts/llm_json_repair.v1.md`)
  - `forge.llm`: `LLMClient` with `OllamaProvider`, `OpenAICompatProvider`, `chat()`, `json(schema=...)` with up to 2 repair-retries, `embed()`, SHA-256 disk cache, run-level JSONL usage logging, and `FakeLLM`
  - `forge.ingest`: loaders for `.pdf`, `.md`, `.txt`, `.docx`, `.csv`, `.xlsx` with non-crashing warnings for scanned/unreadable/unsupported files, section-aware recursive chunker, CSV schema profiling + per-row `col: value` chunks, and compact `CorpusProfile`
  - `forge.retrieval`: `HybridIndex` + `HybridRetriever` (BM25 + dense embeddings fused with RRF, configurable `alpha`, `top_k`, `min_score`, optional `query_rewrite_hook`, run-dir persistence)
  - CLI commands `forge ingest <folder>` and `forge search <folder> "<query>"`
  - Fixtures in `fixtures/phase1_sample/` and tests in `tests/test_llm.py`, `tests/test_ingest.py`, `tests/test_retrieval.py`, and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, `forge ingest`, and `forge search` passed cleanly.

### Phase 2 — Agent Schema, Tools & Generic Runtime (`phase-2-agent-runtime`)
- **Status**: Complete
- **Deliverables**:
  - `forge.agent.schema`: `AgentDefinition` (YAML serialization, named `SystemPromptSections`, retrieval config, justified tools, validated `OutputContractConfig`, model settings) and `AnswerRecord`
  - `forge.tools`: `ToolRegistry` with `csv_lookup` (filter/select/sort/limit, no `eval`/`exec`), `calculator` (AST-safe arithmetic), `http_api` (Open-Meteo host allowlist, timeouts, cache, offline fixture mode), and `current_date`
  - `forge.agent.runtime`: generic `AgentRuntime` with optional query rewrite, hybrid retrieval + `min_score` gate, bounded JSON action tool loop (`max_tool_calls=3`), unattached-tool rejection, and output-contract repair-retry + versioned prompts under `prompts/`
  - `forge chat <definition.yaml> --docs <folder>` multi-turn CLI runner with `--trace`
  - `fixtures/hand_agent.yaml` and `fixtures/docs/` plus unit/CLI tests in `tests/test_agent_schema.py`, `tests/test_tools.py`, `tests/test_agent_runtime.py`, and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, and `forge chat` end-to-end checks passed cleanly.
