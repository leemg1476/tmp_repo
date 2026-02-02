"""Planner that expands a query into a structured plan with tasks."""
from __future__ import annotations

import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import default_llm
from .types import Depth, Plan, Task, TaskStatus


PLAN_PROMPT = """\
You are a planning agent. Create a concise plan for the given user query.
Return JSON with fields:
- rationale (string)
- execution_mode (string: "serial" or "parallel")  # include a short reason inside rationale
- tasks (list of objects with id, description, agent)
Rules:
- 2 to 5 tasks for depth d1-d2, 3 to 7 for d3-d4, up to 8 for d5.
- Prefer explicit verbs in descriptions.
- Assign agent among: "research", "synthesis", "coding", "analysis".
- Keep `id` short (e.g., t1, t2, ...).
Respond ONLY with JSON.\
"""


def create_plan(query: str, depth: Depth) -> Plan:
    """Generate a Plan based on query and depth."""
    llm = default_llm(temperature=0.3)
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=f"Depth: {depth.value}\nQuery: {query}"),
    ]
    raw = llm.invoke(messages).content
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
        tasks_payload = data.get("tasks", [])
        tasks: List[Task] = []
        for idx, task_data in enumerate(tasks_payload, start=1):
            try:
                tasks.append(
                    Task(
                        id=str(task_data.get("id", f"t{idx}")),
                        description=str(task_data.get("description", "")).strip(),
                        agent=str(task_data.get("agent", "analysis")),
                        status=TaskStatus.PENDING,
                    )
                )
            except Exception:
                continue
        rationale = str(data.get("rationale", "Plan generated")).strip()
        execution_mode = str(data.get("execution_mode", "")).strip().lower() or None
        if not tasks:
            tasks = [
                Task(
                    id="t1",
                    description="Analyze the query and draft a response.",
                    agent="analysis",
                )
            ]
        return Plan(
            depth=depth, rationale=rationale, tasks=tasks, execution_mode=execution_mode
        )
    except Exception:
        # Minimal fallback plan
        return Plan(
            depth=depth,
            rationale="Fallback planner",
            tasks=[
                Task(
                    id="t1",
                    description="Research the query and provide an answer.",
                    agent="analysis",
                )
            ],
            execution_mode=None,
        )
