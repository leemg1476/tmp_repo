"""LLM factory helpers."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from .config import get_settings


@lru_cache(maxsize=1)
def default_llm(model: Optional[str] = None, temperature: float = 0.1) -> ChatOpenAI:
    """Return a cached ChatOpenAI client."""
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.default_model,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,
    )

