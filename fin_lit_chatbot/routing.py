from __future__ import annotations

import json
import re

from fin_lit_chatbot.schema import ChatState

VALID_TOPICS = {"investment_education", "money_management", "general"}
VALID_TASK_TYPES = {"explain", "compare", "calculate", "assess", "quiz", "follow_up"}


def parse_router_json(text: str) -> tuple[str, str] | None:
    raw = text.strip()
    if not raw:
        return None

    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        topic = str(data.get("topic", "")).strip()
        task_type = str(data.get("task_type", "")).strip()
        if topic in VALID_TOPICS and task_type in VALID_TASK_TYPES:
            return topic, task_type

    return None


def fallback_route_intent(state: ChatState) -> ChatState:
    q = state["user_query"].lower()
    broad_money = any(x in q for x in ["money management", "manage money", "personal finance"]) and not any(
        x in q for x in ["budget", "debit", "credit", "loan", "debt", "insurance", "spending"]
    )
    broad_investing = any(x in q for x in ["investing", "investment education", "learn investment", "invest"]) and not any(
        x in q for x in ["stock", "bond", "etf", "crypto", "diversification", "portfolio", "risk"]
    )

    if "risk quiz" in q or "risk tolerance" in q or re.fullmatch(r"[ab]", q.strip()):
        state["topic"] = "money_management"
        state["task_type"] = "quiz"
    elif broad_money:
        state["topic"] = "money_management"
        state["task_type"] = "follow_up"
    elif broad_investing:
        state["topic"] = "investment_education"
        state["task_type"] = "follow_up"
    elif any(x in q for x in ["budget", "income", "expense", "savings", "spending"]):
        state["topic"] = "money_management"
        state["task_type"] = "calculate"
    elif any(x in q for x in ["interest", "repay", "monthly payment"]) and any(
        x in q for x in ["debt", "loan", "credit"]
    ):
        state["topic"] = "money_management"
        state["task_type"] = "calculate"
    elif any(x in q for x in ["insurance", "premium", "coverage"]) and "need" in q:
        state["topic"] = "money_management"
        state["task_type"] = "assess"
    elif any(x in q for x in ["stock", "bond", "crypto", "option", "investment", "diversification"]):
        state["topic"] = "investment_education"
        state["task_type"] = "compare" if ("vs" in q or "compare" in q) else "explain"
    else:
        state["topic"] = "money_management"
        state["task_type"] = "explain"

    return state
