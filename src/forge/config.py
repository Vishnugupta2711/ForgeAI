"""Configuration loading for ForgeAI."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    """Model names for each factory role."""

    agent: str = "qwen2.5:3b-instruct"
    judge: str = "llama3.1:8b"
    eval_writer: str = "llama3.1:8b"
    embed: str = "nomic-embed-text"


class EndpointsConfig(BaseModel):
    """LLM provider endpoint settings."""

    provider: Literal["ollama", "openai_compat"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    openai_compat_base_url: str = ""
    openai_api_key: str = ""
    timeout_seconds: float = 120.0


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
    max_latency_seconds_per_run: float = 900.0
    min_improvement_delta: float = 0.02


class CacheConfig(BaseModel):
    """Disk cache settings for LLM and embedding calls."""

    enabled: bool = True
    dir: str = ".cache/llm"


class ForgeConfig(BaseModel):
    """Top-level ForgeAI configuration loaded from forge.toml and env vars."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


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


def load_config(config_path: Path | None = None) -> ForgeConfig:
    """Load configuration from forge.toml with optional .env and env overrides."""
    root = find_project_root(config_path.parent if config_path else None)
    _load_dotenv(root / ".env")

    target = config_path or (root / "forge.toml")
    data: dict = {}
    if target.is_file():
        with target.open("rb") as fh:
            data = tomllib.load(fh)

    cfg = ForgeConfig.model_validate(data)

    if provider := os.environ.get("FORGE_PROVIDER"):
        if provider in ("ollama", "openai_compat"):
            cfg.endpoints.provider = provider  # type: ignore[assignment]
    if ollama_url := os.environ.get("FORGE_OLLAMA_BASE_URL"):
        cfg.endpoints.ollama_base_url = ollama_url
    if openai_url := os.environ.get("FORGE_OPENAI_COMPAT_BASE_URL"):
        cfg.endpoints.openai_compat_base_url = openai_url
    if openai_key := os.environ.get("FORGE_OPENAI_API_KEY"):
        cfg.endpoints.openai_api_key = openai_key

    return cfg
