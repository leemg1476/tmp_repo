"""Typed structures used across the supervisor agent graph."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class Depth(str, Enum):
    """Depth levels for query analysis."""

    D0 = "d0"
    D1 = "d1"
    D2 = "d2"
    D3 = "d3"
    D4 = "d4"
    D5 = "d5"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Task(BaseModel):
    """Single task definition created by the planner."""

    id: str = Field(..., description="Stable identifier for the task.")
    description: str = Field(..., description="What the task should achieve.")
    agent: str = Field(..., description="Agent label responsible for the task.")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Optional[str] = Field(default=None)


class Plan(BaseModel):
    """Plan consisting of multiple tasks."""

    depth: Depth
    rationale: str
    tasks: List[Task]
    execution_mode: Optional[str] = None


class GraphState(TypedDict, total=False):
    """State persisted through the LangGraph workflow."""

    query: str
    depth: Depth
    immediate_answer: Optional[str]
    plan: Plan
    task_reports: List[str]
    last_error: Optional[str]
    final_answer: Optional[str]
    execution_mode: str
