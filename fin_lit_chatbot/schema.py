from __future__ import annotations

from typing import Any, Literal, TypedDict

Topic = Literal["investment_education", "money_management", "general"]
TaskType = Literal["explain", "compare", "calculate", "assess", "quiz", "follow_up"]


class ChatState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_query: str

    topic: Topic
    task_type: TaskType

    session_financials: dict[str, Any]
    risk_quiz_state: dict[str, Any]

    retrieved_chunks: list[dict[str, Any]]
    retrieved_summary: str

    calculation_result: dict[str, Any]
    assessment_result: dict[str, Any]
    risk_profile_result: str | None

    structured_task_status: str
    structured_task_type: str
    structured_completed_last_turn: bool
    last_completed_structured_task: str | None

    active_query: str
    rephrase_attempts: int
    needs_rephrase: bool

    follow_up_suggestions: list[str]
    response_draft: str
