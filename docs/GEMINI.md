# Google Gemini API Integration Reference

Details verified against Google's official Gemini API documentation (fetched 2026-11-27).

## Official Documentation Links

- OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Models catalog: https://ai.google.dev/gemini-api/docs/models
- Choosing a model: https://ai.google.dev/gemini-api/docs/choosing-a-model
- Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Rate limits & usage tiers: https://ai.google.dev/gemini-api/docs/rate-limits
- Live project rate limits dashboard: https://aistudio.google.com/rate-limit

## Endpoint & Protocol

- **OpenAI-compatible Base URL**: `https://generativelanguage.googleapis.com/v1beta/openai/`
  - Chat completions: `POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
  - Embeddings: `POST https://generativelanguage.googleapis.com/v1beta/openai/embeddings`
  - Auth header: `Authorization: Bearer $FORGE_GEMINI_API_KEY`
- **Structured Outputs / JSON Mode**: Supports `response_format={"type": "json_object"}` (as well as strict JSON schema parsing on compatible models) on `/v1beta/openai/chat/completions`.

## Active Verified Model IDs

- **Small / Fast Model (Agent default in `forge.gemini.toml`)**: `gemini-3.5-flash-lite` — low-latency, high-throughput model optimized for fast agent execution.
- **Judge / Eval Writer Default (`forge.gemini.toml`)**: `gemini-3.1-flash-lite` — fast, distinct model avoiding preview `503` spikes and tight daily caps while keeping judge/eval writer separate from the agent model (larger reasoning options such as `gemini-4-hydrogen` / `gemini-4-argon` can be selected via `FORGE_JUDGE_MODEL`).
- **Embeddings Model**: `gemini-embedding-001` (text-only embeddings; `gemini-embedding-2` is also available for multimodal embeddings). Setting `models.embed = ""` in TOML cleanly falls back to BM25-only retrieval without calling an embeddings endpoint.
- *Note*: `gemini-2.5-flash` is marked deprecated in the official models documentation (API access limited to existing users, routing to newer alternatives effective December 1, 2026).

## Rate Limits & Free Tier Behavior

- Limits are enforced **per Google Cloud project** (not per API key) across **RPM** (requests per minute), **TPM** (input tokens per minute), and **RPD** (requests per day; resets midnight Pacific Time).
- Exceeding any dimension returns HTTP `429 RESOURCE_EXHAUSTED`.
- Google updates per-model Free Tier quotas dynamically; inspect active quotas at https://aistudio.google.com/rate-limit.
- To avoid exhausting Free Tier quotas while allowing full 36-case runs, `forge.gemini.toml` defaults to:
  - `requests_per_minute = 14.0`
  - `max_concurrency = 1`
  - `max_http_retries = 5` (exponential backoff with jitter, honoring `Retry-After`)
  - `max_calls_per_run = 1000`
- **Privacy Note**: On the Gemini API Free Tier, prompts and document context sent to the API may be used by Google to improve its products; ForgeAI prints an explicit hosted-provider notice whenever Gemini or another hosted provider is active.

## Using ForgeAI with Gemini

```bash
cp .env.example .env
# Edit .env and set: FORGE_GEMINI_API_KEY=<your-key>

forge run --profile gemini --brief path/to/brief.txt --docs path/to/docs --out runs/gemini_test --auto --fast
```

Or mix local and hosted providers per role (e.g., local Ollama `qwen2.5:3b-instruct` agent + Gemini `gemini-4-hydrogen` judge):

```bash
FORGE_JUDGE_PROVIDER=gemini FORGE_EVAL_WRITER_PROVIDER=gemini forge run --brief brief.txt --docs docs --out runs/mixed --auto
```
