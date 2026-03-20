from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

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

    def run(self, state: ChatState) -> dict[str, str | bool]:
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
                    quiz_state["current_question"] = int(result.get("current_question", 1))
                    quiz_state["answers"] = {
                        int(k): v for k, v in (result.get("answers", {}) or {}).items()
                    }
                    profile = result.get("profile")
                    if profile:
                        state["risk_profile_result"] = str(profile)
                    if str(result.get("status", "")) == "complete":
                        next_steps = result.get("next_steps", [])
                        if isinstance(next_steps, list):
                            cleaned = [str(x).strip() for x in next_steps if str(x).strip()]
                            state["follow_up_suggestions"] = cleaned[:2]
                    message = result.get("message")
                    status = str(result.get("status", "ask_question"))
                    if isinstance(message, str) and message.strip():
                        return {
                            "message": message,
                            "status": "completed" if status == "complete" else "in_progress",
                            "task": "quiz",
                            "tool_called": True,
                        }
                    return {
                        "message": "Let’s continue the risk quiz.",
                        "status": "in_progress",
                        "task": "quiz",
                        "tool_called": True,
                    }
        return {
            "message": _as_text(getattr(reply, "content", "")).strip() or "Would you like to start the risk quiz?",
            "status": "collecting_inputs",
            "task": "quiz",
            "tool_called": False,
        }


class BudgetDebtAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm_with_tools = llm.bind_tools([calculate_budget_tool, calculate_debt_example_tool])

    def run(self, state: ChatState) -> dict[str, str | bool]:
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
                if tool_name == "calculate_budget_tool":
                    financials = state.setdefault("session_financials", {})
                    for key in ["income_monthly", "fixed_expenses", "variable_expenses"]:
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

    def run(self, state: ChatState) -> dict[str, str | bool]:
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

        if task == "quiz" or "risk quiz" in q or "risk tolerance" in q:
            return self.risk_quiz_agent.run(state)

        if task == "assess" or "insurance" in q:
            return self.insurance_check_agent.run(state)

        if task == "calculate":
            return self.budget_debt_agent.run(state)

        # default structured-tools fallback
        return self.budget_debt_agent.run(state)
