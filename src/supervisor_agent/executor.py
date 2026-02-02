"""Plan execution logic for the supervisor graph."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

from .search_agent import ask_with_search
from .types import GraphState, Plan, Task, TaskStatus


def _next_pending(plan: Plan) -> Optional[Task]:
    for task in plan.tasks:
        if task.status == TaskStatus.PENDING:
            return task
    return None


def _build_question(state: GraphState, task: Task, context: str) -> str:
    prompt = (
        "모든 웹 검색을 허용한다. 필요한 경우 언제든 검색하라.\n"
        "사용자에게 추가질의를 하지 말고, 검색 후 데이터가 부족하면 너가 알아서 추론해서 응답하라.\n"
        f"User query: {state['query']}\nTask: {task.description}"
    )
    if context:
        prompt += f"\nContext:\n{context}"
    return prompt


def execute_next_task(state: GraphState) -> GraphState:
    """Execute the next pending task and update state."""
    plan: Plan = state["plan"]
    task_reports = list(state.get("task_reports", []))

    mode = str(state.get("execution_mode", "serial")).lower()
    context = "\n\n".join(task_reports) if task_reports else ""

    if mode == "parallel":
        pending_tasks = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
        if not pending_tasks:
            return state
        for task in pending_tasks:
            task.status = TaskStatus.RUNNING

        def _run(t: Task) -> Tuple[Task, str]:
            q = _build_question(state, t, context)
            return t, ask_with_search(q)

        results = []
        try:
            with ThreadPoolExecutor(max_workers=len(pending_tasks)) as pool:
                futures = [pool.submit(_run, t) for t in pending_tasks]
                for fut in as_completed(futures):
                    results.append(fut.result())
        except Exception as exc:  # pragma: no cover - defensive
            for task in pending_tasks:
                if task.status != TaskStatus.COMPLETE:
                    task.status = TaskStatus.FAILED
                    task.result = f"failed: {exc}"
            task_reports.append(f"parallel FAILED: {exc}")
            return {**state, "plan": plan, "task_reports": task_reports, "last_error": str(exc)}

        # Preserve original task order in reports
        result_map = {t.id: res for t, res in results}
        for task in pending_tasks:
            result = result_map.get(task.id, "")
            task.result = result
            task.status = TaskStatus.COMPLETE
            task_reports.append(f"{task.id} [{task.agent}] → {result}")
        return {**state, "plan": plan, "task_reports": task_reports}

    # serial mode (default)
    task = _next_pending(plan)
    if not task:
        return state
    task.status = TaskStatus.RUNNING
    try:
        question = _build_question(state, task, context)
        result = ask_with_search(question)
        task.result = result
        task.status = TaskStatus.COMPLETE
        task_reports.append(f"{task.id} [{task.agent}] → {result}")
        return {**state, "plan": plan, "task_reports": task_reports}
    except Exception as exc:  # pragma: no cover - defensive
        task.status = TaskStatus.FAILED
        task.result = f"failed: {exc}"
        task_reports.append(f"{task.id} FAILED: {exc}")
        return {**state, "plan": plan, "task_reports": task_reports, "last_error": str(exc)}
