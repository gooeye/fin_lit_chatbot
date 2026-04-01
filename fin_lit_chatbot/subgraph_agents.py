from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from fin_lit_chatbot.constants import RISK_QUIZ
from fin_lit_chatbot.payloads import canonicalize_payload_code, payload_from_text
from fin_lit_chatbot.schema import ChatState
from fin_lit_chatbot.tools import (
    TOOL_REGISTRY,
    advance_risk_quiz_tool,
    assess_insurance_needs_tool,
    calculate_budget_tool,
    calculate_debt_example_tool,
)


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _invoke_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return "I can help with a risk quiz, budget, debt example, or insurance checklist."
    try:
        result = tool.invoke(args)
    except Exception:
        return "I need a bit more detail to run that. Please share the required inputs."
    return result if isinstance(result, str) else str(result)


def _recent_chat_history(state: ChatState, limit: int = 8) -> str:
    msgs = state.get("messages", [])[-limit:]
    lines: list[str] = []
    for m in msgs:
        role = str(m.get("role", "unknown"))
        content = str(m.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class RiskQuizAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm_with_tools = llm.bind_tools([advance_risk_quiz_tool])

    def _apply_quiz_result(self, state: ChatState, result: dict[str, Any]) -> dict[str, str | bool]:
        quiz_state = state.setdefault("risk_quiz_state", {"current_question": 1, "answers": {}})
        quiz_state["current_question"] = int(result.get("current_question", 1))
        quiz_state["answers"] = {
            int(k): v for k, v in (result.get("answers", {}) or {}).items()
        }

        profile = result.get("profile")
        if profile:
            state["risk_profile_result"] = str(profile)

        result_status = str(result.get("status", "ask_question"))
        if result_status == "ask_question":
            option_a = str(result.get("option_a", "")).strip()
            option_b = str(result.get("option_b", "")).strip()
            question_number = int(result.get("current_question", quiz_state.get("current_question", 1)))
            if option_a and option_b:
                state["follow_up_suggestions"] = [
                    payload_from_text(
                        f"A. {option_a}",
                        code="quiz.answer.A",
                        meta={"question_number": question_number},
                    ),
                    payload_from_text(
                        f"B. {option_b}",
                        code="quiz.answer.B",
                        meta={"question_number": question_number},
                    ),
                ]

        if result_status == "complete":
            next_steps = result.get("next_steps", [])
            if isinstance(next_steps, list):
                state["follow_up_suggestions"] = [
                    payload_from_text(str(item).strip())
                    for item in next_steps
                    if str(item).strip()
                ][:2]

        message = result.get("message")
        if isinstance(message, str) and message.strip():
            return {
                "message": message,
                "status": "completed" if result_status == "complete" else "in_progress",
                "task": "quiz",
                "tool_called": True,
            }

        return {
            "message": "Let's continue the risk quiz.",
            "status": "in_progress",
            "task": "quiz",
            "tool_called": True,
        }

    def _run_deterministic(self, state: ChatState) -> dict[str, str | bool] | None:
        code = canonicalize_payload_code(str(state.get("button_code", "")).strip())
        if code not in {"quiz.start", "quiz.answer.A", "quiz.answer.B"}:
            return None

        quiz_state = state.setdefault("risk_quiz_state", {"current_question": 1, "answers": {}})
        current_question = int(quiz_state.get("current_question", 1))
        answers = {str(k): v for k, v in quiz_state.get("answers", {}).items()}

        if code == "quiz.start":
            quiz_state["current_question"] = 1
            quiz_state["answers"] = {}
            state["risk_profile_result"] = None
            current_question = 1
            answers = {}
            user_input = "Start risk quiz"
        else:
            if current_question > len(RISK_QUIZ):
                return {
                    "message": "The risk quiz is already complete. Tap Start risk quiz if you want to run it again.",
                    "status": "collecting_inputs",
                    "task": "quiz",
                    "tool_called": False,
                }

            button_meta = state.get("button_meta", {})
            question_number = button_meta.get("question_number") if isinstance(button_meta, dict) else None
            try:
                expected_question = int(question_number) if question_number is not None else 0
            except (TypeError, ValueError):
                expected_question = 0

            if expected_question and expected_question != current_question:
                return {
                    "message": (
                        f"That answer was for quiz question {expected_question}. "
                        f"Please use the latest options for question {current_question}."
                    ),
                    "status": "collecting_inputs",
                    "task": "quiz",
                    "tool_called": False,
                }

            user_input = code.rsplit(".", 1)[-1]

        result = advance_risk_quiz_tool.invoke(
            {
                "user_input": user_input,
                "current_question": current_question,
                "answers": answers,
            }
        )
        if isinstance(result, dict):
            return self._apply_quiz_result(state, result)

        return {
            "message": "Would you like to start the risk quiz?",
            "status": "collecting_inputs",
            "task": "quiz",
            "tool_called": False,
        }

    def run(self, state: ChatState) -> dict[str, str | bool]:
        deterministic = self._run_deterministic(state)
        if deterministic is not None:
            return deterministic

        history = _recent_chat_history(state)
        prompt = (
            "You are the risk quiz agent.\n"
            "If user wants to start/continue risk quiz, call advance_risk_quiz_tool.\n"
            "If user is unclear, ask one short clarifying question instead of calling a tool.\n"
            f"User query: {state.get('user_query', '')}\n"
            f"Recent chat history:\n{history}\n"
            f"risk_quiz_state: {json.dumps(state.get('risk_quiz_state', {}))}\n"
        )
        reply = self.llm_with_tools.invoke(prompt)
        tool_calls = getattr(reply, "tool_calls", None) or []
        if tool_calls:
            first = tool_calls[0]
            tool_name = first.get("name", "")
            args = dict(first.get("args", {}) or {})
            if isinstance(tool_name, str) and isinstance(args, dict):
                quiz_state = state.setdefault("risk_quiz_state", {"current_question": 1, "answers": {}})
                args.setdefault("user_input", state.get("user_query", ""))
                args.setdefault("current_question", int(quiz_state.get("current_question", 1)))
                answers = {str(k): v for k, v in quiz_state.get("answers", {}).items()}
                args.setdefault("answers", answers)
                result = advance_risk_quiz_tool.invoke(args)
                if isinstance(result, dict):
                    return self._apply_quiz_result(state, result)

        return {
            "message": _as_text(getattr(reply, "content", "")).strip() or "Would you like to start the risk quiz?",
            "status": "collecting_inputs",
            "task": "quiz",
            "tool_called": False,
        }


class BudgetDebtAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm_with_tools = llm.bind_tools([calculate_budget_tool, calculate_debt_example_tool])

    def _run_deterministic(self, state: ChatState) -> dict[str, str | bool] | None:
        preferred_tool = str(state.get("preferred_structured_tool", "")).strip()
        if preferred_tool not in {"calculate_budget_tool", "calculate_debt_example_tool"}:
            return None

        financials = state.setdefault("session_financials", {})
        required_fields = {
            "calculate_budget_tool": ["income_monthly", "fixed_expenses", "variable_expenses"],
            "calculate_debt_example_tool": ["balance", "annual_interest_rate", "monthly_payment"],
        }[preferred_tool]

        if all(financials.get(field) is not None for field in required_fields):
            args = {field: float(financials[field]) for field in required_fields}
            return {
                "message": _invoke_tool_call(preferred_tool, args),
                "status": "completed",
                "task": "calculate",
                "tool_called": True,
            }

        if preferred_tool == "calculate_budget_tool":
            message = "To run a budget calculation, share your monthly income, fixed expenses, and variable expenses."
        else:
            message = "To run a debt example calculation, share the balance, annual interest rate, and monthly payment."

        return {
            "message": message,
            "status": "collecting_inputs",
            "task": "calculate",
            "tool_called": False,
        }

    def run(self, state: ChatState) -> dict[str, str | bool]:
        deterministic = self._run_deterministic(state)
        if deterministic is not None:
            return deterministic

        history = _recent_chat_history(state)
        prompt = (
            "You are the budget/debt agent.\n"
            "Use tools only when required inputs are available.\n"
            "If inputs are missing, ask a concise follow-up question and do NOT call a tool yet.\n"
            "Required inputs:\n"
            "- calculate_budget_tool: income_monthly, fixed_expenses, variable_expenses\n"
            "- calculate_debt_example_tool: balance, annual_interest_rate, monthly_payment\n"
            "Routing:\n"
            "- If user asks about debt repayment/interest/loan simulation, choose debt tool when ready.\n"
            "- Otherwise choose budget tool when ready.\n"
            f"User query: {state.get('user_query', '')}\n"
            f"Recent chat history:\n{history}\n"
            f"session_financials: {json.dumps(state.get('session_financials', {}))}\n"
        )
        reply = self.llm_with_tools.invoke(prompt)
        tool_calls = getattr(reply, "tool_calls", None) or []
        if tool_calls:
            first = tool_calls[0]
            tool_name = first.get("name", "")
            args = dict(first.get("args", {}) or {})
            if isinstance(tool_name, str) and isinstance(args, dict):
                financials = state.setdefault("session_financials", {})
                if tool_name == "calculate_budget_tool":
                    for key in ["income_monthly", "fixed_expenses", "variable_expenses"]:
                        if key in args and args[key] is not None:
                            financials[key] = float(args[key])
                if tool_name == "calculate_debt_example_tool":
                    for key in ["balance", "annual_interest_rate", "monthly_payment"]:
                        if key in args and args[key] is not None:
                            financials[key] = float(args[key])
                return {
                    "message": _invoke_tool_call(tool_name, args),
                    "status": "completed",
                    "task": "calculate",
                    "tool_called": True,
                }

        return {
            "message": _as_text(getattr(reply, "content", "")).strip()
            or "Please share the missing numbers so I can run the calculation.",
            "status": "collecting_inputs",
            "task": "calculate",
            "tool_called": False,
        }


class InsuranceCheckAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm_with_tools = llm.bind_tools([assess_insurance_needs_tool])

    def _run_deterministic(self, state: ChatState) -> dict[str, str | bool] | None:
        preferred_tool = str(state.get("preferred_structured_tool", "")).strip()
        if preferred_tool != "assess_insurance_needs_tool":
            return None

        return {
            "message": (
                "To run the insurance needs assessment, tell me whether you have dependents "
                "and whether you have major income obligations."
            ),
            "status": "collecting_inputs",
            "task": "assess",
            "tool_called": False,
        }

    def run(self, state: ChatState) -> dict[str, str | bool]:
        deterministic = self._run_deterministic(state)
        if deterministic is not None:
            return deterministic

        history = _recent_chat_history(state)
        prompt = (
            "You are the insurance-check agent.\n"
            "Call assess_insurance_needs_tool when enough context is available.\n"
            "Otherwise ask one concise clarifying question.\n"
            f"User query: {state.get('user_query', '')}\n"
            f"Recent chat history:\n{history}\n"
        )
        reply = self.llm_with_tools.invoke(prompt)
        tool_calls = getattr(reply, "tool_calls", None) or []
        if tool_calls:
            first = tool_calls[0]
            tool_name = first.get("name", "")
            args = first.get("args", {})
            if isinstance(tool_name, str) and isinstance(args, dict):
                return {
                    "message": _invoke_tool_call(tool_name, args),
                    "status": "completed",
                    "task": "assess",
                    "tool_called": True,
                }
        return {
            "message": _as_text(getattr(reply, "content", "")).strip()
            or "Do you have dependents or major income obligations?",
            "status": "collecting_inputs",
            "task": "assess",
            "tool_called": False,
        }


class StructuredToolsAgent:
    """Orchestrator for specialist structured-tool subgraph agents."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.risk_quiz_agent = RiskQuizAgent(llm)
        self.budget_debt_agent = BudgetDebtAgent(llm)
        self.insurance_check_agent = InsuranceCheckAgent(llm)

    def run(self, state: ChatState) -> dict[str, str | bool]:
        task = state.get("task_type")
        q = state.get("user_query", "").lower()
        preferred_tool = str(state.get("preferred_structured_tool", "")).strip()

        if preferred_tool == "advance_risk_quiz_tool":
            return self.risk_quiz_agent.run(state)

        if preferred_tool == "assess_insurance_needs_tool":
            return self.insurance_check_agent.run(state)

        if preferred_tool in {"calculate_budget_tool", "calculate_debt_example_tool"}:
            return self.budget_debt_agent.run(state)

        if task == "quiz" or "risk quiz" in q or "risk tolerance" in q:
            return self.risk_quiz_agent.run(state)

        if task == "assess" or "insurance" in q:
            return self.insurance_check_agent.run(state)

        if task == "calculate":
            return self.budget_debt_agent.run(state)

        return self.budget_debt_agent.run(state)
