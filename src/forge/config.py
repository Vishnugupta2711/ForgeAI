"""Configuration loading and profile management for ForgeAI."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["ollama", "openai_compat", "gemini"]
HOSTED_PROVIDERS: frozenset[str] = frozenset({"openai_compat", "gemini"})


class ModelsConfig(BaseModel):
    """Model names for each factory role."""

    agent: str = "qwen2.5:3b-instruct"
    judge: str = "llama3.1:8b"
    eval_writer: str = "llama3.1:8b"
    embed: str = "nomic-embed-text"


class EndpointsConfig(BaseModel):
    """LLM provider endpoint and per-role provider settings."""

    provider: ProviderName = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    openai_compat_base_url: str = ""
    openai_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    role_providers: dict[str, ProviderName] = Field(default_factory=dict)
    timeout_seconds: float = 120.0


class RateLimitsConfig(BaseModel):
    """Client-side throttling, concurrency, and retry controls."""

    requests_per_minute: float | None = None
    max_concurrency: int = Field(default=2, ge=1)
    max_http_retries: int = Field(default=3, ge=0)
    backoff_seconds: float = Field(default=0.5, ge=0.0)


class GenerationConfig(BaseModel):
    """Default inference parameters."""

    temperature: float = 0.0
    seed: int = 42
    max_retries: int = 2


class BudgetsConfig(BaseModel):
    """Factory execution budgets and stop thresholds."""

    max_iterations: int = 3
    max_intake_questions: int = 5
    max_tokens_per_run: int = 250000
    max_calls_per_run: int | None = None
    max_latency_seconds_per_run: float = 900.0
    min_improvement_delta: float = 0.02


class CacheConfig(BaseModel):
    """Disk cache settings for LLM and embedding calls."""

    enabled: bool = True
    dir: str = ".cache/llm"


class JudgeConfig(BaseModel):
    """Thresholds for deterministic-first judge claim verification."""

    claim_support_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    claim_unsupport_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class ForgeConfig(BaseModel):
    """Top-level ForgeAI configuration loaded from forge.toml / profiles and env vars."""

    profile: str = "default"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    def provider_for_role(self, role: str) -> ProviderName:
        """Resolve provider for a given role ('agent', 'judge', 'eval_writer', 'embed')."""
        return self.endpoints.role_providers.get(role, self.endpoints.provider)

    def providers_by_role(self) -> dict[str, str]:
        """Return resolved provider mapping for all standard factory roles."""
        return {
            role: self.provider_for_role(role)
            for role in ("agent", "judge", "eval_writer", "embed")
        }

    def hosted_providers_used(self) -> list[str]:
        """Return sorted hosted providers active across configured roles."""
        active: set[str] = set()
        for role in ("agent", "judge", "eval_writer", "embed"):
            prov = self.provider_for_role(role)
            if prov in HOSTED_PROVIDERS:
                if role == "embed" and not self.models.embed.strip():
                    continue
                active.add(prov)
        return sorted(active)

    def privacy_notice(self) -> str | None:
        """Return a one-line privacy notice when any hosted provider is active."""
        hosted = self.hosted_providers_used()
        if not hosted:
            return None
        joined = ", ".join(hosted)
        return (
            f"Privacy notice: Hosted provider ({joined}) active — "
            "document chunks and prompts are sent externally to that provider."
        )


def get_gemini_api_key(root: Path | None = None) -> str:
    """Read FORGE_GEMINI_API_KEY strictly from environment / .env (never from TOML)."""
    base = find_project_root(root)
    _load_dotenv(base / ".env")
    return os.environ.get("FORGE_GEMINI_API_KEY", "").strip()


def _load_dotenv(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ if unset."""
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from start (or cwd) to locate forge.toml or pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "forge.toml").is_file() or (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def load_prompt(name: str, root: Path | None = None) -> str:
    """Load a versioned prompt markdown file from prompts/."""
    base = find_project_root(root)
    prompt_path = base / "prompts" / name
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _normalize_profile_name(profile: str | None) -> str:
    cleaned = (profile or "").strip().lower()
    if not cleaned or cleaned == "ollama":
        return "default"
    return cleaned


def load_config(
    config_path: Path | None = None,
    *,
    profile: str | None = None,
) -> ForgeConfig:
    """Load configuration from forge.toml or forge.<profile>.toml + .env overrides."""
    root = find_project_root(config_path.parent if config_path else None)
    _load_dotenv(root / ".env")

    resolved_profile = _normalize_profile_name(profile or os.environ.get("FORGE_PROFILE"))

    if config_path is not None:
        target = config_path
    elif resolved_profile == "default":
        target = root / "forge.toml"
    else:
        target = root / f"forge.{resolved_profile}.toml"
        if not target.is_file():
            raise FileNotFoundError(
                f"Unknown ForgeAI profile '{resolved_profile}': expected {target}"
            )

    data: dict = {}
    if target.is_file():
        with target.open("rb") as fh:
            data = tomllib.load(fh)

    # Strip any accidental API key if someone puts one in TOML; keys live strictly in env
    endpoints_data = data.get("endpoints")
    if isinstance(endpoints_data, dict):
        endpoints_data.pop("gemini_api_key", None)

    cfg = ForgeConfig.model_validate(data)
    cfg.profile = resolved_profile

    valid_providers: set[str] = {"ollama", "openai_compat", "gemini"}
    if provider := os.environ.get("FORGE_PROVIDER"):
        if provider in valid_providers:
            cfg.endpoints.provider = provider  # type: ignore[assignment]
    if ollama_url := os.environ.get("FORGE_OLLAMA_BASE_URL"):
        cfg.endpoints.ollama_base_url = ollama_url
    if openai_url := os.environ.get("FORGE_OPENAI_COMPAT_BASE_URL"):
        cfg.endpoints.openai_compat_base_url = openai_url
    if openai_key := os.environ.get("FORGE_OPENAI_API_KEY"):
        cfg.endpoints.openai_api_key = openai_key
    if gemini_url := os.environ.get("FORGE_GEMINI_BASE_URL"):
        cfg.endpoints.gemini_base_url = gemini_url

    role_env_map = {
        "agent": "FORGE_AGENT_PROVIDER",
        "judge": "FORGE_JUDGE_PROVIDER",
        "eval_writer": "FORGE_EVAL_WRITER_PROVIDER",
        "embed": "FORGE_EMBED_PROVIDER",
    }
    for role_name, env_var in role_env_map.items():
        if val := os.environ.get(env_var, "").strip().lower():
            if val in valid_providers:
                cfg.endpoints.role_providers[role_name] = val  # type: ignore[assignment]

    if agent_model := os.environ.get("FORGE_AGENT_MODEL", "").strip():
        cfg.models.agent = agent_model
    if judge_model := os.environ.get("FORGE_JUDGE_MODEL", "").strip():
        cfg.models.judge = judge_model
    if eval_model := os.environ.get("FORGE_EVAL_WRITER_MODEL", "").strip():
        cfg.models.eval_writer = eval_model
    if "FORGE_EMBED_MODEL" in os.environ:
        cfg.models.embed = os.environ["FORGE_EMBED_MODEL"].strip()

    return cfg
