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

### Phase 3 — Intake & Groundability Guard (`phase-3-intake-groundability`)
- **Status**: Complete
- **Deliverables**:
  - `forge.intake`: `IntakeInterviewer` (`prompts/intake_questions.v1.md`, `prompts/intake_auto_answer.v1.md`, `prompts/intake_synthesize.v1.md`) supporting `interactive`, `--answers`, and `--auto` (with every auto answer recorded under `assumptions`), strict `<= 5` question cap enforced in code, and `RequirementsDocument` (`requirements.json` + `requirements.md`)
  - `forge.groundability`: `GroundabilityGuard` (`prompts/groundability_decompose.v1.md`, `prompts/groundability_verify.v1.md`) decomposing topics/capabilities, computing deterministic coverage (`key_term_overlap` + retrieval score), verifying returned quotes as real chunk substrings (downgrading unverified/common-knowledge/absent topics to `UNSUPPORTED`), moving gaps to `must_refuse`, narrowing scope on `PARTIAL`, and supporting `--strict`
  - CLI command `forge intake --brief ... --docs ... [--auto|--answers] [--strict] --out ...`
  - Tests in `tests/test_intake_groundability.py` and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, and `forge intake` CLI tests passed cleanly.

### Phase 4 — Agent Generator (`phase-4-generator`)
- **Status**: Complete
- **Deliverables**:
  - `forge.generator.heuristics`: `derive_retrieval_settings()` computing `chunk_size`, `overlap`, `top_k`, `hybrid_alpha`, `rerank`, `query_rewrite`, and `min_score` from `CorpusProfile` stats (`loaded_docs`, `avg_section_len`, `tabular_share`, `language_profile`) with logged rationale
  - `forge.generator.tool_gate`: deterministic `gate_tools()` attaching only tools backed by active `REQ-xxx` capabilities and supported by corpus/brief (e.g. tabular doc for `csv_lookup`, explicit live external need for `http_api`), logging every accepted/rejected candidate in `generation_log.json`
  - `forge.generator.synthesizer`: `AgentGenerator` generating each `SystemPromptSections` field via dedicated `prompts/generate_*.v1.md` templates, enforcing quoted-evidence grounding and honest refusal of gaps/exclusions, building output contract, recording prompt hashes, and running a 3-question `AgentRuntime` smoke test
  - CLI command `forge generate --requirements ... --docs ... --out ...`
  - Tests in `tests/test_generator.py` (8-case tool-selection matrix, heuristics, determinism/cache, smoke run) and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, and `forge generate` CLI checks passed cleanly.
