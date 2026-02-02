"""Configuration helpers for the supervisor agent."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
import os


@lru_cache(maxsize=1)
def _load_env() -> None:
    """Load environment variables from a local .env file once."""
    # 우선 현재 작업 디렉터리 기준
    for candidate in [Path(".env"), Path(__file__).resolve().parents[2] / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")
    default_model: str = Field("gpt-5-mini", env="OPENAI_MODEL")
    langsmith_api_key: Optional[str] = Field(
        None, env=["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"]
    )
    langsmith_project: Optional[str] = Field(
        "supervisor", env=["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"]
    )
    langsmith_tracing_v2: bool = Field(
        True, env=["LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"]
    )
    execution_mode: str = Field("serial", env="EXECUTION_MODE")

class Config:
        env_file = ".env"  # pragma: no cover - pydantic config style
        extra = "ignore"


def get_settings() -> Settings:
    """Return cached settings instance with friendly error on missing env."""
    _load_env()
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        missing = [err["loc"][0] for err in exc.errors() if err.get("type") == "missing"]
        hint = (
            "OPENAI_API_KEY 가 .env 또는 환경변수에 설정되어 있어야 합니다. "
            "프로젝트 루트의 .env 파일에 OPENAI_API_KEY=... 를 채워주세요."
        )
        if missing:
            raise RuntimeError(f"환경변수 누락: {', '.join(missing)}. {hint}") from exc
        raise


def configure_observability() -> None:
    """Enable LangSmith/LangChain tracing if 키가 설정되어 있으면."""
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true" if settings.langsmith_tracing_v2 else "false")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        if settings.langsmith_project:
            os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
