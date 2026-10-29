"""应用配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = "auto"
    openai_api_key: str | None = field(default=None, repr=False)
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    host: str = "127.0.0.1"
    port: int = 8000
    docs_dir: Path = Path("data")
    top_k: int = 4

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        project_root = root or Path.cwd()
        provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        if provider not in {"auto", "openai", "ollama", "demo"}:
            raise ValueError(
                "LLM_PROVIDER 必须是 auto、openai、ollama 或 demo 之一。"
            )

        docs_path = Path(os.getenv("DOCS_DIR", "data"))
        if not docs_path.is_absolute():
            docs_path = project_root / docs_path

        try:
            port = int(os.getenv("APP_PORT", "8000"))
            top_k = int(os.getenv("TOP_K", "4"))
        except ValueError as exc:
            raise ValueError("APP_PORT 和 TOP_K 必须是整数。") from exc
        if not 1 <= top_k <= 10:
            raise ValueError("TOP_K 必须在 1 到 10 之间。")

        return cls(
            provider=provider,
            openai_api_key=_optional("OPENAI_API_KEY"),
            openai_base_url=_optional("OPENAI_BASE_URL"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip(),
            host=os.getenv("APP_HOST", "127.0.0.1").strip(),
            port=port,
            docs_dir=docs_path.resolve(),
            top_k=top_k,
        )

