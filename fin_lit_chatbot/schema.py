from __future__ import annotations

from typing import Any, Literal, TypedDict

Topic = Literal["investment_education", "money_management", "general"]
TaskType = Literal["explain", "compare", "calculate", "assess", "quiz", "follow_up"]
SuggestionType = Literal["content", "tool"]


class FollowUpPayload(TypedDict, total=False):
    text: str
    code: str
    meta: dict[str, Any]
    type: SuggestionType


UserInput = str | FollowUpPayload
FollowUpSuggestion = str | FollowUpPayload


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

    follow_up_suggestions: list[FollowUpSuggestion]
    button_text: str
    button_code: str
    button_meta: dict[str, Any]
    button_code_known: bool
    deterministic_route_used: bool
    preferred_structured_tool: str
    response_draft: str
