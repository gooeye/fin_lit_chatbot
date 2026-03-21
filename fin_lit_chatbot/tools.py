from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from langchain_core.tools import BaseTool, tool

from fin_lit_chatbot.constants import (
    RISK_MISMATCH_INTERPRETATIONS,
    RISK_PROFILE_BANDS,
    RISK_QUIZ,
    RISK_RESULT_TEMPLATES,
)


def _profile_from_total(total: int) -> tuple[str, str]:
    for band in RISK_PROFILE_BANDS:
        if int(band["min_total"]) <= total <= int(band["max_total"]):
            return str(band["profile"]), str(band["summary"])
    return "moderate", "Balanced willingness and capacity for some investment risk."


def _mismatch_meaning(willingness: int, capacity: int) -> str:
    if willingness > capacity:
        return RISK_MISMATCH_INTERPRETATIONS["willingness_gt_capacity"]
    if capacity > willingness:
        return RISK_MISMATCH_INTERPRETATIONS["capacity_gt_willingness"]
    return RISK_MISMATCH_INTERPRETATIONS["aligned"]


def _score_risk_answers(answers: dict[int, str]) -> dict[str, Any]:
    willingness = 0
    capacity = 0
    for q_num, answer in answers.items():
        val = RISK_QUIZ[int(q_num)]["score"][answer]
        if RISK_QUIZ[int(q_num)]["dimension"] == "willingness":
            willingness += val
        else:
            capacity += val

    total = willingness + capacity
    profile, profile_summary = _profile_from_total(total)
    mismatch = _mismatch_meaning(willingness, capacity)
    template = RISK_RESULT_TEMPLATES.get(profile, {})

    return {
        "profile": profile,
        "profile_summary": profile_summary,
        "willingness_score": willingness,
        "capacity_score": capacity,
        "total_score": total,
        "headline": str(template.get("headline", "Here is your current risk profile.")),
        "profile_explanation": str(template.get("explanation", profile_summary)),
        "mismatch_meaning": mismatch,
        "next_steps": list(template.get("next_steps", []))[:3],
    }


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


def _option_overlap_score(user_tokens: set[str], option_text: str) -> float:
    option_tokens = _tokenize(option_text)
    if not user_tokens or not option_tokens:
        return 0.0
    shared = len(user_tokens & option_tokens)
    return shared / max(1, len(option_tokens))


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in phrases)


def extract_quiz_choice(user_input: str, current_question: int) -> str | None:
    """Infer deterministic A/B choice from direct or natural-language quiz reply."""
    if current_question not in RISK_QUIZ:
        return None

    text = user_input.strip()
    lower = text.lower()

    # Explicit single-letter input.
    explicit = re.fullmatch(r"\s*([ab])\s*", lower)
    if explicit:
        return explicit.group(1).upper()

    # Prefix forms like "A. ...", "B) ...", "a - ...".
    prefix = re.match(r"^\s*([ab])(?:\b|[\).:\-])", lower)
    if prefix:
        return prefix.group(1).upper()

    # Direct phrasing forms.
    if _contains_any(lower, ["option a", "choose a", "pick a", "answer is a", "go with a"]):
        return "A"
    if _contains_any(lower, ["option b", "choose b", "pick b", "answer is b", "go with b"]):
        return "B"

    # Natural-language semantic-ish match via option text overlap.
    q = RISK_QUIZ[current_question]
    option_a = str(q["options"]["A"])
    option_b = str(q["options"]["B"])
    user_tokens = _tokenize(lower)
    score_a = _option_overlap_score(user_tokens, option_a)
    score_b = _option_overlap_score(user_tokens, option_b)

    threshold = 0.25
    if score_a >= threshold or score_b >= threshold:
        if score_a > score_b:
            return "A"
        if score_b > score_a:
            return "B"

    return None


@tool("advance_risk_quiz_tool")
def advance_risk_quiz_tool(user_input: str, current_question: int, answers: dict[str, str] | None = None) -> dict[str, Any]:
    """Advance deterministic risk-quiz state and return either next question or final score."""
    answers = answers or {}
    normalized_answers = {int(k): str(v).upper() for k, v in answers.items()}
    choice = extract_quiz_choice(user_input, current_question)

    if choice and current_question <= len(RISK_QUIZ):
        normalized_answers[current_question] = choice
        current_question += 1

    if current_question <= len(RISK_QUIZ):
        q = RISK_QUIZ[current_question]
        return {
            "status": "ask_question",
            "current_question": current_question,
            "answers": {str(k): v for k, v in normalized_answers.items()},
            "question": q["question"],
            "option_a": q["options"]["A"],
            "option_b": q["options"]["B"],
            "option_a_explanation": q.get("option_explanations", {}).get("A", ""),
            "option_b_explanation": q.get("option_explanations", {}).get("B", ""),
            "followup_tip": q.get("followup_tip", ""),
            "message": (
                f"Risk tolerance quiz — question {current_question}:\n"
                f"{q['question']}\n"
                f"A. {q['options']['A']}\n"
                f"B. {q['options']['B']}\n\n"
                f"Tip: {q.get('followup_tip', '')}\n\n"
                "Reply with A/B, or answer in your own words."
            ),
        }

    score = _score_risk_answers(normalized_answers)
    return {
        "status": "complete",
        "current_question": current_question,
        "answers": {str(k): v for k, v in normalized_answers.items()},
        "message": (
            f"{score['headline']}\n"
            f"Profile: **{score['profile']}** ({score['profile_summary']})\n\n"
            "What these scores mean:\n"
            "- Willingness score reflects your emotional comfort with market ups and downs.\n"
            "- Capacity score reflects how much risk your timeline/goals can practically absorb.\n\n"
            f"Your scores: willingness {score['willingness_score']}/2, "
            f"capacity {score['capacity_score']}/2, total {score['total_score']}/4.\n"
            f"Interpretation: {score['mismatch_meaning']}\n\n"
            f"About your profile: {score['profile_explanation']}\n\n"
            "Educational only, not personalized investment advice."
        ),
        **score,
    }


@tool("calculate_budget_tool")
def calculate_budget_tool(income_monthly: float, fixed_expenses: float, variable_expenses: float) -> str:
    """Calculate monthly budget summary. Requires: income_monthly, fixed_expenses, variable_expenses."""
    income = float(income_monthly)
    fixed = float(fixed_expenses)
    variable = float(variable_expenses)
    expenses_total = fixed + variable
    surplus = income - expenses_total
    savings_rate = surplus / income if income else 0.0
    return (
        "Simple monthly budget summary:\n"
        f"- Income: {income:.2f}\n"
        f"- Total expenses: {expenses_total:.2f}\n"
        f"- Surplus: {surplus:.2f}\n"
        f"- Savings rate: {savings_rate * 100:.1f}%"
    )


@tool("calculate_debt_example_tool")
def calculate_debt_example_tool(balance: float, annual_interest_rate: float, monthly_payment: float) -> str:
    """Simulate debt repayment. Requires: balance, annual_interest_rate, monthly_payment."""
    monthly_rate = (annual_interest_rate / 100) / 12
    months = 0
    total_interest = 0.0
    current = float(balance)

    while current > 0 and months < 600:
        interest = current * monthly_rate
        principal = float(monthly_payment) - interest
        if principal <= 0:
            return "Your payment is too low to reduce this debt at the given interest rate."
        current -= principal
        total_interest += interest
        months += 1

    return (
        "Debt example result:\n"
        f"- Months to repay: {int(months)}\n"
        f"- Total interest: {total_interest:.2f}\n"
        f"- Ending balance: {max(current, 0.0):.2f}"
    )


@tool("assess_insurance_needs_tool")
def assess_insurance_needs_tool(has_dependents: bool = False, has_income_obligations: bool = False) -> str:
    """Return a simple insurance-needs assessment."""
    needs_attention: list[str] = ["medical coverage awareness"]
    if has_dependents:
        needs_attention.append("life coverage awareness")
    if has_income_obligations:
        needs_attention.append("income protection awareness")
    priority = "basic" if len(needs_attention) <= 2 else "elevated"
    return (
        "Basic insurance-needs assessment:\n"
        f"- Priority level: {priority}\n"
        f"- Areas to review: {', '.join(needs_attention)}\n"
        "- Note: This is educational guidance, not policy advice."
    )


BOUND_TOOLS: list[BaseTool] = [
    advance_risk_quiz_tool,
    calculate_budget_tool,
    calculate_debt_example_tool,
    assess_insurance_needs_tool,
]

TOOL_REGISTRY: dict[str, BaseTool] = {t.name: t for t in BOUND_TOOLS}
