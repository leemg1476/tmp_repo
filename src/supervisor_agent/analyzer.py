"""Depth analyzer for incoming queries."""
from __future__ import annotations

import json
from typing import Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from .llm import default_llm
from .types import Depth


DEPTH_GUIDE = """\
Classify the query by required reasoning depth.
- d0: direct facts or trivial answers, no reasoning.
- d1: single-step retrieval or short reasoning.
- d2: small multi-step reasoning.
- d3: medium multi-step with light synthesis.
- d4: complex multi-step with planning and synthesis.
- d5: open-ended research or multi-agent needed.
Respond with a JSON object: {"depth": "d0"}\
"""


def classify_depth(query: str) -> Depth:
    """Call the LLM to classify the query depth."""
    llm = default_llm(temperature=0)
    messages = [
        SystemMessage(content=DEPTH_GUIDE),
        HumanMessage(content=f"Query: {query}"),
    ]
    raw = llm.invoke(messages).content
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
        depth_value = str(data.get("depth", "")).lower()
        return Depth(depth_value)
    except Exception:
        # Heuristic fallback if model fails to return JSON.
        text = (raw or "").lower() if isinstance(raw, str) else ""
        for level in [Depth.D0, Depth.D1, Depth.D2, Depth.D3, Depth.D4, Depth.D5]:
            if level.value in text:
                return level
        return Depth.D3


def quick_answer(query: str) -> str:
    """Produce a concise direct answer for d0 queries."""
    llm = default_llm(temperature=0.2)
    messages = [
        SystemMessage(
            content="Answer the user's query directly and concisely in one paragraph."
        ),
        HumanMessage(content=query),
    ]
    return llm.invoke(messages).content  # type: ignore[return-value]

