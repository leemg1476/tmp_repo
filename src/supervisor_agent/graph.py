"""LangGraph workflow that wires analyzer, planner, executor, and aggregator."""
from __future__ import annotations

from typing import Iterable

from langgraph.graph import END, StateGraph

from .aggregator import aggregate_response
from .analyzer import classify_depth, quick_answer
from .config import configure_observability, get_settings
from .executor import execute_next_task
from .planner import create_plan
from .types import Depth, GraphState, TaskStatus


def _analyze_depth(state: GraphState) -> GraphState:
    depth = classify_depth(state["query"])
    if depth == Depth.D0:
        # Immediate answer path; no plan needed.
        answer = quick_answer(state["query"])
        return {
            **state,
            "depth": depth,
            "immediate_answer": answer,
        }
    return {**state, "depth": depth, "task_reports": []}


def _plan(state: GraphState) -> GraphState:
    plan = create_plan(state["query"], state["depth"])
    if plan.execution_mode and not state.get("execution_mode"):
        return {**state, "plan": plan, "execution_mode": plan.execution_mode}
    return {**state, "plan": plan}


def _should_plan(state: GraphState) -> str:
    return "aggregate" if state.get("depth") == Depth.D0 else "plan"


def _should_continue(state: GraphState) -> str:
    plan = state.get("plan")
    if not plan:
        return "aggregate"
    pending_exists = any(t.status == TaskStatus.PENDING for t in plan.tasks)
    return "execute" if pending_exists else "aggregate"


def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze", _analyze_depth)
    workflow.add_node("plan", _plan)
    workflow.add_node("execute", execute_next_task)
    workflow.add_node("aggregate", aggregate_response)

    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges(
        "analyze",
        _should_plan,
        {"aggregate": "aggregate", "plan": "plan"},
    )
    workflow.add_edge("plan", "execute")
    workflow.add_conditional_edges(
        "execute",
        _should_continue,
        {"execute": "execute", "aggregate": "aggregate"},
    )
    workflow.add_edge("aggregate", END)
    return workflow.compile()


def run_supervisor(query: str, execution_mode: str | None = None):
    """Convenience helper to run the compiled workflow."""
    configure_observability()
    settings = get_settings()
    app = build_workflow()
    result = app.invoke(
        {
            "query": query,
            "execution_mode": execution_mode or settings.execution_mode,
        }
    )
    # d0: 즉답, d1~d5: aggregator 결과
    if result.get("depth") == Depth.D0:
        return result.get("immediate_answer")
    return result.get("final_answer")
