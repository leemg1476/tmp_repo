"""Aggregate task outputs into a final response."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import default_llm
from .types import GraphState


def aggregate_response(state: GraphState) -> GraphState:
    plan = state.get("plan")
    reports = state.get("task_reports", [])
    query = state["query"]
    immediate = state.get("immediate_answer")

    llm = default_llm(temperature=0.2)
    messages = [
        SystemMessage(
            content="Combine the task reports into a single, well-structured answer. "
            "Keep it concise and actionable. Mention uncertainties if present."
        ),
        HumanMessage(
            content=f"User query: {query}\n\n"
            f"Immediate answer (if any): {immediate or '없음'}\n\n"
            f"Task reports:\n" + ("\n".join(reports) if reports else "없음")
        ),
    ]
    final_answer = llm.invoke(messages).content  # type: ignore[return-value]
    return {**state, "final_answer": final_answer, "plan": plan}
