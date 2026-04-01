from __future__ import annotations

import unittest

from fin_lit_chatbot.engine import FinLitBot
from fin_lit_chatbot.payloads import canonicalize_payload_code, normalize_message_payload
from fin_lit_chatbot.subgraph_agents import BudgetDebtAgent, RiskQuizAgent


class DummyLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _prompt):
        raise AssertionError("LLM fallback should not be used in deterministic tests.")


class StructuredPayloadTests(unittest.TestCase):
    def test_payload_normalization_canonicalizes_codes(self) -> None:
        payload = normalize_message_payload(
            {
                "text": "Take the risk tolerance quiz",
                "code": "tool.quiz.risk_tolerance",
                "meta": {"source": "button"},
            }
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["code"], "quiz.start")
        self.assertEqual(payload["meta"]["source"], "button")
        self.assertEqual(canonicalize_payload_code("tool.quiz.risk_tolerance"), "quiz.start")

    def test_deterministic_preprocessor_routes_known_codes(self) -> None:
        bot = FinLitBot.__new__(FinLitBot)
        bot._status_callback = None

        state = {
            "button_code": "tool.quiz.risk_tolerance",
            "button_meta": {},
        }
        updated = FinLitBot.deterministic_preprocessor(bot, state)

        self.assertEqual(updated["button_code"], "quiz.start")
        self.assertTrue(updated["button_code_known"])
        self.assertTrue(updated["deterministic_route_used"])
        self.assertEqual(updated["task_type"], "quiz")
        self.assertEqual(updated["preferred_structured_tool"], "advance_risk_quiz_tool")

    def test_risk_quiz_buttons_stay_deterministic(self) -> None:
        agent = RiskQuizAgent(DummyLLM())
        state = {
            "user_query": "Start risk quiz",
            "button_code": "quiz.start",
            "button_meta": {},
            "risk_quiz_state": {"current_question": 1, "answers": {}},
        }

        start_result = agent.run(state)

        self.assertEqual(start_result["status"], "in_progress")
        start_suggestions = state["follow_up_suggestions"]
        self.assertEqual([item["code"] for item in start_suggestions], ["quiz.answer.A", "quiz.answer.B"])
        self.assertEqual(start_suggestions[0]["meta"]["question_number"], 1)

        state["user_query"] = start_suggestions[0]["text"]
        state["button_code"] = "quiz.answer.A"
        state["button_meta"] = {"question_number": 1}

        answer_result = agent.run(state)

        self.assertEqual(answer_result["status"], "in_progress")
        self.assertEqual(state["risk_quiz_state"]["answers"][1], "A")
        self.assertEqual(state["risk_quiz_state"]["current_question"], 2)
        answer_suggestions = state["follow_up_suggestions"]
        self.assertEqual(answer_suggestions[0]["meta"]["question_number"], 2)

    def test_budget_button_asks_for_missing_inputs_then_runs(self) -> None:
        agent = BudgetDebtAgent(DummyLLM())
        state = {
            "user_query": "Run budget calculation",
            "preferred_structured_tool": "calculate_budget_tool",
            "session_financials": {},
        }

        missing_result = agent.run(state)
        self.assertEqual(missing_result["status"], "collecting_inputs")
        self.assertIn("monthly income", str(missing_result["message"]))

        state["session_financials"] = {
            "income_monthly": 4000,
            "fixed_expenses": 1500,
            "variable_expenses": 1000,
        }
        completed_result = agent.run(state)

        self.assertEqual(completed_result["status"], "completed")
        self.assertIn("Simple monthly budget summary", str(completed_result["message"]))


if __name__ == "__main__":
    unittest.main()
