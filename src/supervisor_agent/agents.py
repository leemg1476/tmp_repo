"""Lightweight task-specific agents backed by ChatOpenAI."""
from __future__ import annotations

from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import default_llm


AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "research": "You are a research agent. Gather key facts, data points, and concise notes to address the task.",
    "synthesis": "You are a synthesis agent. Combine provided context into a cohesive, concise explanation.",
    "coding": "You are a coding agent. Suggest code or algorithms to satisfy the task, keeping responses concise.",
    "analysis": "You are an analysis agent. Reason carefully and outline clear conclusions for the task.",
}


def run_agent(agent: str, task_description: str, query: str, context: str | None = None) -> str:
    """Run a single agent with the appropriate system prompt."""
    system_prompt = AGENT_SYSTEM_PROMPTS.get(agent, AGENT_SYSTEM_PROMPTS["analysis"])
    llm = default_llm(temperature=0.2)
    human_content = f"User query: {query}\nTask: {task_description}"
    if context:
        human_content += f"\nContext:\n{context}"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    return llm.invoke(messages).content  # type: ignore[return-value]

