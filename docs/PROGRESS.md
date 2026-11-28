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

### Phase 5 — Eval Generator (`phase-5-eval-generator`)
- **Status**: Complete
- **Deliverables**:
  - `prompts/evalgen_*.v1.md`: committed generation, solvability, and OOS-check templates
  - `forge.evals`: `EvalCase` / `EvalSet` models, Gate 1 (`verify_case_key_facts` normalized/fuzzy check), Gate 2 (`check_solvability` from gold chunks only), Gate 3 (`check_oos_unanswerable`), Gate 4 (`dedupe_by_embedding`), Gate 5 (`stratified_split` deterministic ~2/3 dev, ~1/3 held_out), and `EvalGenerator` respecting quotas (>= 36 cases) and recording eval-writer model/family vs agent model
  - CLI command `forge evals generate --definition ... --requirements ... --docs ... --out ...` writing `eval_set.json` and `eval_set_stats.md`
  - Tests in `tests/test_evals.py` and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, and `forge evals generate` CLI checks passed cleanly.

### Phase 6 — Judge & Canary Self-Test (`phase-6-judge`)
- **Status**: Complete
- **Deliverables**:
  - `docs/JUDGE.md` and `prompts/judge_*.v1.md` (`judge_extract_claims.v1.md`, `judge_verify_claims.v1.md`, `judge_correctness.v1.md`)
  - `forge.judge.deterministic`: refusal matching, retrieved-chunk citation validity, canonicalized key-fact recall (numbers, units, currency, dates, fuzzy text), numeric hallucination guard, Wilson 95% confidence intervals, and length-vs-pass Pearson correlation
  - `forge.judge.scorer`: `Judge` combining deterministic checks with 2-pass order-shuffled claim-level groundedness and ambiguous correctness verification (falling back deterministically with `uncertain=True` on disagreement), self-preference warnings, and 100% pass-rate suspicion flag
  - `forge.judge.selftest`: 8-type canary suite (`gold_answer`, `numerically_corrupted`, `hallucinated_claim`, `wrong_document`, `padded_verbose`, `terse_correct`, `correct_refusal`, `wrongful_refusal`) writing `judge_reliability.json`
  - CLI commands `forge evals run` (writing `results.json` + `scorecard.md`) and `forge judge-selftest`
  - Tests in `tests/test_judge.py` and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, `forge evals run`, and `forge judge-selftest` passed cleanly.

### Phase 7 — Critic-Optimizer Loop (`phase-7-optimizer`)
- **Status**: Complete
- **Deliverables**:
  - `prompts/optimizer_classify_tie.v1.md` and `prompts/optimizer_propose_patches.v1.md`
  - `forge.optimizer.classifier`: deterministic-first failure classifier (`retrieval_miss`, `hallucination`, `wrong_refusal` incl. `missed_refusal` subtype, `prompt_defect`) with LLM tie-breaker
  - `forge.optimizer.patches` & `forge.optimizer.guards`: typed `RetrievalPatch` (re-indexing on `chunk_size`/`overlap`) and `PromptPatch` (touching only named sections), byte-level `verify_diff_guard()`, and n-gram/overlap `check_patch_for_leak()`
  - `forge.optimizer.loop`: `OptimizerLoop` (max 3 iterations, dev-only iterations, category-regression rollback, 2-consecutive no-improvement & 100%-dev early stop, held-out scored only at baseline and end, per-iteration `iterations/iter_k/` artifacts and `scorecard.json`)
  - CLI command `forge optimize --run <run_dir>`
  - Tests in `tests/test_optimizer.py` and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, and `forge optimize` CLI tests passed cleanly.

### Phase 8 — End-to-End Resumable Pipeline & Report (`phase-8-pipeline-report`)
- **Status**: Complete
- **Deliverables**:
  - `prompts/report_diagnosis.v1.md`
  - `forge.report.lint`: `collect_allowed_numbers()` + `lint_narrative_numbers()` verifying every number in LLM failure diagnosis prose exists in real result files and cites valid case IDs
  - `forge.report.builder`: `build_report()` writing `report.md` (run summary, requirements & groundability flags up front, tool gate audit, per-iteration dev scorecard + final held-out Wilson intervals, failure counts, "What Still Fails and Why", judge reliability + same-model warning, suspicion rule, limitations) and `scorecard.csv`
  - `forge.pipeline`: `ForgePipeline` orchestrating `ingest -> intake -> groundability -> generate -> evals -> judge_selftest -> optimize -> report` with disk checkpoints (`pipeline_state.json`), `--resume`, `--fast`, live stage timings, and partial `report.md` on mid-run failure
  - CLI command `forge run --brief ... --docs ... --out ... [--auto|--answers] [--max-iters] [--strict] [--fast] [--resume]`
  - Tests in `tests/test_pipeline_report.py` and `tests/test_smoke.py`
- **Verification**:
  - `pytest -q`, `ruff check .`, `forge run`, and `forge chat` on generated `definition.yaml` passed cleanly.

### Phase 10 — Hardening, Chaos Tests, Rehearsal & Documentation (`phase-10-hardening`)
- **Status**: Complete
- **Deliverables**:
  - Hardened `forge.ingest.loaders`: large-file byte/char caps with warnings, UTF-16/Latin-1/CP1252 fallback, odd CSV delimiter sniffing (`;`, `\t`, `|`, `,`), empty/scanned PDF/unsupported file warnings
  - Hardened `forge.llm.providers`: transient timeout/5xx exponential backoff retries and clear actionable `ConnectError` guidance for Ollama
  - Hardened `forge.evals.generator`: graceful eval quota scaling on very short corpora with `scaled_quotas_note`
  - Added `--time-budget-min` graceful degradation across `ForgePipeline` + `OptimizerLoop` and recorded degradation notes in `report.md`
  - Added rehearsal sets under `fixtures/rehearsal/it_helpdesk/` and `fixtures/rehearsal/unsupported_scope/`
  - Added `tests/test_domain_isolation.py` (static domain-term check over `src/` and `prompts/`) and `tests/test_hardening.py` (chaos, robustness, time budget, rehearsal tests)
  - Wrote top-level `README.md` with install, model setup, quickstart, architecture, design decisions, and honest limitations
- **Verification**:
  - `pytest -q` (88 passed) and `ruff check .` passed cleanly. Note: Phase 9 live-model demo corpora remain on branch `phase-9-demos` pending an active Ollama instance.

### Phase 11g — Switchable Gemini Provider Profile (`phase-11g-gemini-provider`)
- **Status**: Complete
- **Deliverables**:
  - Verified official Gemini API documentation (`https://ai.google.dev/gemini-api/docs/{openai,models,choosing-a-model,embeddings,rate-limits}`) in `docs/GEMINI.md`
  - Added `forge.gemini.toml` (`gemini-3.5-flash-lite` agent, `gemini-4-hydrogen` judge/eval_writer, `gemini-embedding-001`) selectable via `--profile gemini` across all CLI commands or `FORGE_PROFILE=gemini`, keeping `forge.toml` (Ollama) as default
  - Added `FORGE_GEMINI_API_KEY=` placeholder in `.env.example`; key read strictly from env / gitignored `.env`, never serialized into TOML, logs, caches, or run artifacts
  - Implemented `GeminiProvider`, exponential backoff with jitter on `429`/`5xx`, RPM throttle, concurrency cap, per-run `max_calls_per_run` budget, actionable key/quota error mapping, BM25 fallback on empty embed model, per-role provider routing (`endpoints.role_providers` / `FORGE_<ROLE>_PROVIDER`), hosted-provider privacy notice, and profile/provider reporting in `pipeline_state.json`, `AgentDefinition`, and `report.md`
  - Registered `live` pytest marker in `pyproject.toml` and added mocked/security/`@pytest.mark.live` tests in `tests/test_gemini_provider.py` + updated `README.md` ("Using Gemini")
- **Verification**:
  - `pytest -q` and `ruff check .` passed cleanly.

### Phase 9b — Retrieval Scale, Optimizer Guards, Matcher & Resume Safety (`phase-9b-guards`)
- **Status**: Complete
- **Deliverables**:
  - Normalized `min_score` (`[0.0, 1.0]`) onto `HybridRetriever.score_scale()` / `normalize_score()` (`raw_score_max = 1 / (rrf_k + 1)`), applied the identical normalized gate in `AgentRuntime._retrieve()`, and validated/clamped `RetrievalPatch.min_score` at proposal time
  - Added cheap 3–5 query retrieval-only & contract schema precheck gate (`_precheck_candidate`, `IterationRecord.rejected_precheck`) skipping full dev eval on destructive patches and retrying one alternative fallback operator in the same iteration
  - Enforced strict failure-category -> operator mapping (`validate_patch_operator_mapping`: `RetrievalPatch` restricted to `retrieval_miss`; `prompt_defect`/`contract_failure` -> `PromptPatch`; `hallucination` -> `grounding_rules`)
  - Normalized snake-case underscores and common abbreviations (`moq`, `min`, `max`, `qty`, `pct`, `avg`, `std`, `est`, `approx`) in `canonicalize_text()` while keeping numeric mismatch guards intact
  - Added per-stage role provider/model config hashes in `PipelineState` (`role_hashes`, `stage_role_hashes`, `stage_models`), refusing `--resume` by default on model changes with `PipelineResumeError` and supporting `--allow-model-change` invalidation + downstream re-execution
  - Built `report.md` Section 1 headers from `usage.jsonl` actuals (with `model_change_history` and Section 7 same-provider/same-family warnings) and added top-level + verified per-case `provenance` on `EvalSet`
- **Verification**:
  - `pytest -q`, `ruff check .`, domain isolation grep, and repo secret scan all passed cleanly (`tests/test_phase9b_guards.py`).
