from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fin_lit_chatbot.schema import FollowUpPayload

CANONICAL_TOOL_CODES = {
    "quiz.start",
    "tool.calculate.budget",
    "tool.calculate.debt_example",
    "tool.assess.insurance_needs",
}

CODE_ALIASES = {
    "tool.quiz.risk_tolerance": "quiz.start",
}


def canonicalize_payload_code(code: str) -> str:
    normalized = code.strip()
    return CODE_ALIASES.get(normalized, normalized)


def is_known_deterministic_code(code: str) -> bool:
    normalized = canonicalize_payload_code(code)
    return normalized in {
        "topic.start.money_management",
        "topic.start.investment_education",
        "quiz.start",
        "quiz.answer.A",
        "quiz.answer.B",
        "tool.calculate.budget",
        "tool.calculate.debt_example",
        "tool.assess.insurance_needs",
    }


def infer_deterministic_code_from_text(text: str) -> str:
    lower = text.strip().lower()
    if "risk tolerance quiz" in lower or "risk quiz" in lower:
        return "quiz.start"
    if "budget calculation" in lower:
        return "tool.calculate.budget"
    if "debt example calculation" in lower:
        return "tool.calculate.debt_example"
    if "insurance-needs assessment" in lower or "insurance needs assessment" in lower:
        return "tool.assess.insurance_needs"
    return ""


def payload_from_text(
    text: str,
    *,
    code: str | None = None,
    meta: dict[str, Any] | None = None,
    suggestion_type: str | None = None,
) -> FollowUpPayload:
    payload: FollowUpPayload = {"text": text.strip()}
    normalized_code = canonicalize_payload_code(code or "")
    if normalized_code:
        payload["code"] = normalized_code
    if isinstance(meta, dict) and meta:
        payload["meta"] = dict(meta)
    if suggestion_type in {"content", "tool"}:
        payload["type"] = suggestion_type
    return payload


def normalize_message_payload(value: object) -> FollowUpPayload | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return payload_from_text(text)

    if not isinstance(value, dict):
        return None

    text = str(value.get("text", "")).strip()
    code = canonicalize_payload_code(str(value.get("code", "")).strip())
    meta_raw = value.get("meta")
    suggestion_type = value.get("type")

    if not text and code:
        text = code
    if not text:
        return None

    payload = payload_from_text(text, code=code or None, suggestion_type=suggestion_type if isinstance(suggestion_type, str) else None)
    if isinstance(meta_raw, dict) and meta_raw:
        payload["meta"] = dict(meta_raw)
    return payload


def normalize_payload_list(values: Iterable[object]) -> list[FollowUpPayload]:
    normalized: list[FollowUpPayload] = []
    for value in values:
        payload = normalize_message_payload(value)
        if payload:
            normalized.append(payload)
    return normalized


def payload_text(value: object) -> str:
    payload = normalize_message_payload(value)
    return payload["text"] if payload else ""
