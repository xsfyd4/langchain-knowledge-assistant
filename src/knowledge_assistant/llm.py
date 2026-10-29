"""LangChain 模型适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings


class ModelConfigurationError(RuntimeError):
    """模型配置不完整时抛出。"""


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    llm: Any | None
    mode: str
    model_name: str
    note: str | None = None


def create_model_runtime(settings: Settings) -> ModelRuntime:
    provider = settings.provider
    if provider == "auto":
        provider = "openai" if settings.openai_api_key else "demo"

    if provider == "demo":
        return ModelRuntime(
            llm=None,
            mode="demo",
            model_name="本地检索演示",
            note="未配置大模型，当前只展示检索结果。",
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ModelConfigurationError(
                "使用 openai 模式时必须配置 OPENAI_API_KEY。"
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "缺少 langchain-openai，请先执行 pip install -r requirements.txt。"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "temperature": 0.1,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ModelRuntime(
            llm=ChatOpenAI(**kwargs),
            mode="openai",
            model_name=settings.openai_model,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ModelConfigurationError(
                "缺少 langchain-ollama，请先执行 pip install -r requirements.txt。"
            ) from exc
        return ModelRuntime(
            llm=ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.1,
            ),
            mode="ollama",
            model_name=settings.ollama_model,
        )

    raise ModelConfigurationError(f"不支持的模型提供方：{provider}")

