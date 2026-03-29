from __future__ import annotations

import json
from collections.abc import Callable

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from fin_lit_chatbot.config import Settings
from fin_lit_chatbot.constants import RISK_QUIZ
from fin_lit_chatbot.rag import RagService
from fin_lit_chatbot.routing import fallback_route_intent, parse_router_json
from fin_lit_chatbot.schema import ChatState
from fin_lit_chatbot.subgraph_agents import StructuredToolsAgent
from fin_lit_chatbot.tools import extract_quiz_choice


class FinLitBot:
    def __init__(self) -> None:
        self.settings = Settings()
        self.rag = RagService(self.settings)
        self._fast_llm = self._build_chat_model(self.settings.fast_model, temperature=0.1)
        self._smart_llm = self._build_chat_model(self.settings.smart_model, temperature=0.2)
        self._structured_tools_agent = StructuredToolsAgent(self._fast_llm)
        self._status_callback: Callable[[str], None] | None = None
        self._token_callback: Callable[[str], None] | None = None
        self._graph = self._build_graph()

    def _invoke_config(self, state: ChatState, channel: str) -> dict:
        return {
            "tags": ["finlit", channel, "langgraph"],
            "metadata": {
                "channel": channel,
                "topic": state.get("topic", "unknown"),
                "task_type": state.get("task_type", "unknown"),
            },
        }

    def _build_chat_model(self, model: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            default_headers=self.settings.openrouter_headers(),
        )

    def _build_graph(self):
        builder = StateGraph(ChatState)
        builder.add_node("ingest_input", self.ingest_input)
        builder.add_node("load_session_context", self.load_session_context)
        builder.add_node("intent_topic_router", self.intent_topic_router)
        builder.add_node("follow_up_question", self.follow_up_question)
        builder.add_node("structured_handover", self.structured_handover)
        builder.add_node("query_rephraser", self.query_rephraser)

        builder.add_node("investment_knowledge", self.investment_knowledge)
        builder.add_node("money_management_knowledge", self.money_management_knowledge)
        builder.add_node("structured_tools", self.structured_tools)

        builder.add_node("response_composer", self.response_composer)

        builder.add_edge(START, "ingest_input")
        builder.add_edge("ingest_input", "load_session_context")
        builder.add_edge("load_session_context", "intent_topic_router")
        builder.add_conditional_edges(
            "intent_topic_router",
            self.route_from_router,
            {
                "follow_up_subgraph": "follow_up_question",
                "investment_knowledge_subgraph": "investment_knowledge",
                "money_management_knowledge_subgraph": "money_management_knowledge",
                "structured_tools_subgraph": "structured_tools",
            },
        )
        builder.add_edge("follow_up_question", "response_composer")
        builder.add_conditional_edges(
            "investment_knowledge",
            self.route_after_knowledge,
            {
                "query_rephraser": "query_rephraser",
                "response_composer": "response_composer",
            },
        )
        builder.add_conditional_edges(
            "money_management_knowledge",
            self.route_after_knowledge,
            {
                "query_rephraser": "query_rephraser",
                "response_composer": "response_composer",
            },
        )
        builder.add_conditional_edges(
            "query_rephraser",
            self.route_from_rephraser,
            {
                "investment_knowledge_subgraph": "investment_knowledge",
                "money_management_knowledge_subgraph": "money_management_knowledge",
            },
        )
        builder.add_edge("structured_tools", "structured_handover")
        builder.add_edge("structured_handover", "response_composer")
        builder.add_edge("response_composer", END)

        return builder.compile()

    def respond(self, state: ChatState, user_query: str, channel: str = "streamlit") -> ChatState:
        input_state = self._build_input_state(state, user_query)

        result: ChatState = self._graph.invoke(
            input_state,
            config=self._invoke_config(input_state, channel),
        )

        messages = list(result.get("messages", []))
        messages.append({"role": "assistant", "content": result.get("response_draft", "")})
        result["messages"] = messages
        return result

    def respond_live(
        self,
        state: ChatState,
        user_query: str,
        on_status: Callable[[str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        channel: str = "streamlit",
    ) -> ChatState:
        input_state = self._build_input_state(state, user_query)
        self._status_callback = on_status
        self._token_callback = on_token

        if self._status_callback:
            self._status_callback("Starting...")

        try:
            result: ChatState = self._graph.invoke(
                input_state,
                config=self._invoke_config(input_state, channel),
            )
        finally:
            self._status_callback = None
            self._token_callback = None

        messages = list(result.get("messages", []))
        messages.append({"role": "assistant", "content": result.get("response_draft", "")})
        result["messages"] = messages
        return result

    def respond_with_progress(self, state: ChatState, user_query: str, channel: str = "streamlit"):
        input_state = self._build_input_state(state, user_query)
        final_state: ChatState = input_state

        status_map = {
            "ingest_input": "Reading your message...",
            "load_session_context": "Loading session context...",
            "intent_topic_router": "Thinking about what you said...",
            "follow_up_question": "Preparing follow-up options...",
            "query_rephraser": "Trying an alternate search query...",
            "investment_knowledge": "Researching investment documents...",
            "money_management_knowledge": "Researching money-management documents...",
            "structured_tools": "Running calculation/assessment tools...",
            "structured_handover": "Finalizing tool results...",
            "response_composer": "Composing response...",
        }

        yield {"type": "status", "message": "Starting..."}

        for event in self._graph.stream(
            input_state,
            stream_mode="updates",
            config=self._invoke_config(input_state, channel),
        ):
            if not isinstance(event, dict):
                continue

            for node_name, update in event.items():
                if node_name in status_map:
                    yield {"type": "status", "message": status_map[node_name]}
                if isinstance(update, dict):
                    final_state = {**final_state, **update}

        messages = list(final_state.get("messages", []))
        messages.append({"role": "assistant", "content": final_state.get("response_draft", "")})
        final_state["messages"] = messages
        yield {"type": "final", "state": final_state}

    def _build_input_state(self, state: ChatState, user_query: str) -> ChatState:
        messages = list(state.get("messages", []))
        messages.append({"role": "user", "content": user_query})
        return {
            **state,
            "messages": messages,
            "response_draft": "",
            "follow_up_suggestions": [],
            "structured_task_status": "idle",
            "structured_task_type": "",
            "active_query": user_query,
            "rephrase_attempts": 0,
            "needs_rephrase": False,
        }

    def _emit_status(self, message: str) -> None:
        if self._status_callback:
            self._status_callback(message)

    def _emit_token(self, token: str) -> None:
        if self._token_callback and token:
            self._token_callback(token)

    def _chunk_to_text(self, chunk: object) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out: list[str] = []
            for item in content:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        out.append(text)
            return "".join(out)
        return ""

    # Nodes
    def ingest_input(self, state: ChatState) -> ChatState:
        self._emit_status("Reading your question...")
        state["user_query"] = state["messages"][-1]["content"].strip()
        return state

    def load_session_context(self, state: ChatState) -> ChatState:
        self._emit_status("Loading session context...")
        state.setdefault("session_financials", {})
        state.setdefault("risk_quiz_state", {"current_question": 1, "answers": {}})
        state.setdefault("follow_up_suggestions", [])
        state.setdefault("structured_task_status", "idle")
        state.setdefault("structured_task_type", "")
        state.setdefault("structured_completed_last_turn", False)
        state.setdefault("last_completed_structured_task", None)
        state.setdefault("active_query", state.get("user_query", ""))
        state.setdefault("rephrase_attempts", 0)
        state.setdefault("needs_rephrase", False)
        return state

    def intent_topic_router(self, state: ChatState) -> ChatState:
        self._emit_status("Routing to the right path...")
        query = state["user_query"].strip()

        quiz_state = state.get("risk_quiz_state", {})
        current_question = int(quiz_state.get("current_question", 1))
        answers = quiz_state.get("answers", {})
        quiz_ongoing = (current_question <= len(RISK_QUIZ)) and (
            bool(answers) or state.get("task_type") == "quiz"
        )

        # Quiz-aware routing: if quiz is already in progress, prioritize continuation checks.
        if quiz_ongoing:
            if extract_quiz_choice(query, current_question) is not None:
                state["topic"] = "money_management"
                state["task_type"] = "quiz"
                return state

            continue_prompt = (
                "You are a quiz-intent checker. A risk quiz is currently in progress.\n"
                "Return STRICT JSON only with key continue_quiz as true or false.\n"
                "Set continue_quiz=true only if user intends to continue/answer the quiz;\n"
                "set false if user asks a different question/topic.\n\n"
                f"User query: {query}"
            )
            continue_reply = self._fast_llm.invoke(continue_prompt)
            continue_text = continue_reply.content if isinstance(continue_reply.content, str) else str(continue_reply.content)

            candidates = [continue_text.strip()]
            start = continue_text.find("{")
            end = continue_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidates.append(continue_text[start : end + 1])

            for candidate in candidates:
                try:
                    data = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("continue_quiz") is True:
                    state["topic"] = "money_management"
                    state["task_type"] = "quiz"
                    return state

        prompt = (
            "You are an intent router for a financial literacy chatbot.\n"
            "Classify the user query into STRICT JSON with keys topic and task_type only.\n"
            "No markdown. No explanation.\n\n"
            "Allowed topic values: investment_education, money_management, general\n"
            "Allowed task_type values: explain, compare, calculate, assess, quiz, follow_up\n\n"
            "Routing policy:\n"
            "- If the query is broad/ambiguous and needs clarification before teaching, use task_type=follow_up.\n"
            "- Use topic=investment_education for stocks, bonds, diversification, risk/return product learning.\n"
            "- Use topic=money_management for budgeting, spending, credit/debit, loans, debt, insurance.\n"
            "- Use task_type=quiz for risk tolerance quiz interactions.\n"
            "- Use task_type=calculate for budget/debt math style requests.\n"
            "- Use task_type=assess for insurance-needs assessment style requests.\n"
            "- Use task_type=compare for explicit comparisons (e.g., X vs Y).\n"
            "- Otherwise use explain/follow_up.\n\n"
            "Examples:\n"
            "User: I want to learn money management\n"
            "Output: {\"topic\":\"money_management\",\"task_type\":\"follow_up\"}\n"
            "User: Teach me investing\n"
            "Output: {\"topic\":\"investment_education\",\"task_type\":\"follow_up\"}\n"
            "User: What is diversification?\n"
            "Output: {\"topic\":\"investment_education\",\"task_type\":\"explain\"}\n"
            "User: Credit cards vs debit cards\n"
            "Output: {\"topic\":\"money_management\",\"task_type\":\"compare\"}\n\n"
            f"User query: {query}"
        )

        reply = self._fast_llm.invoke(prompt)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        parsed = parse_router_json(text)
        if parsed:
            topic, task_type = parsed
            state["topic"] = topic
            state["task_type"] = task_type
            return state

        return fallback_route_intent(state)

    def route_from_router(self, state: ChatState) -> str:
        task_type = state.get("task_type", "follow_up")
        topic = state.get("topic", "general")

        if task_type == "follow_up":
            return "follow_up_subgraph"
        if task_type in {"quiz", "calculate", "assess"}:
            return "structured_tools_subgraph"
        if topic == "investment_education":
            return "investment_knowledge_subgraph"
        return "money_management_knowledge_subgraph"

    def route_after_knowledge(self, state: ChatState) -> str:
        if state.get("needs_rephrase"):
            return "query_rephraser"
        return "response_composer"

    def route_from_rephraser(self, state: ChatState) -> str:
        topic = state.get("topic", "money_management")
        if topic == "investment_education":
            return "investment_knowledge_subgraph"
        return "money_management_knowledge_subgraph"

    def follow_up_question(self, state: ChatState) -> ChatState:
        self._emit_status("Preparing follow-up options...")
        topic = state.get("topic", "general")
        query = state.get("user_query", "")

        question, suggestions = self._generate_follow_up_payload(
            state=state,
            topic=topic,
            user_query=query,
            mode="follow_up",
            max_suggestions=5,
        )
        state["response_draft"] = question or "Could you share a bit more detail so I can help precisely?"
        state["follow_up_suggestions"] = suggestions[:5]
        return state

    def structured_handover(self, state: ChatState) -> ChatState:
        self._emit_status("Finalizing tool results...")
        status = state.get("structured_task_status", "idle")
        task = state.get("structured_task_type", "")

        if status == "completed":
            state["structured_completed_last_turn"] = True
            state["last_completed_structured_task"] = task or state.get("task_type")

            done_label = {
                "calculate": "calculation",
                "quiz": "risk quiz",
                "assess": "insurance assessment",
            }.get(task, "task")

            prefix = f"Done — we’ve completed the {done_label}."
            body = state.get("response_draft", "").strip()
            state["response_draft"] = f"{prefix}\n\n{body}" if body else prefix
            return state

        state["structured_completed_last_turn"] = False
        return state

    def query_rephraser(self, state: ChatState) -> ChatState:
        self._emit_status("Trying an alternate search query...")
        attempts = int(state.get("rephrase_attempts", 0))
        if attempts >= 2:
            state["needs_rephrase"] = False
            return state

        prompt = (
            "You rewrite financial education search queries for retrieval.\n"
            "Return ONLY one short query string, no quotes, no markdown.\n"
            "Keep meaning the same, but use alternate phrasing and synonyms.\n"
            "Focus on concepts likely to appear in educational documents.\n\n"
            f"Topic: {state.get('topic', 'money_management')}\n"
            f"Original user question: {state.get('user_query', '')}\n"
            f"Last attempted retrieval query: {state.get('active_query', state.get('user_query', ''))}\n"
            f"Attempt number: {attempts + 1}"
        )
        reply = self._fast_llm.invoke(prompt)
        rewritten = reply.content if isinstance(reply.content, str) else str(reply.content)
        rewritten = rewritten.strip().strip('"')
        if not rewritten:
            rewritten = state.get("user_query", "")

        state["active_query"] = rewritten
        state["rephrase_attempts"] = attempts + 1
        state["needs_rephrase"] = False
        return state

    def _docs_support_question(self, question: str, docs: list[dict[str, object]]) -> bool:
        if not docs:
            return False

        excerpts = []
        for d in docs[:3]:
            excerpts.append(str(d.get("chunk_text", ""))[:500])
        prompt = (
            "Decide if the provided excerpts are sufficient to answer the user question.\n"
            "Return STRICT JSON only: {\"supported\": true|false}.\n"
            "Set supported=true only if excerpts are clearly relevant and enough for a basic explanation.\n\n"
            f"User question: {question}\n\n"
            f"Excerpts:\n{chr(10).join(excerpts)}"
        )
        try:
            reply = self._fast_llm.invoke(prompt)
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            candidates = [text.strip()]
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start : end + 1])

            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(parsed, dict) and isinstance(parsed.get("supported"), bool):
                    return bool(parsed.get("supported"))
        except Exception:
            pass

        # Fallback: if we have any retrieved docs, accept.
        return True

    def _fallback_general_answer(self, question: str, topic: str) -> str:
        prompt = (
            "You are a beginner-friendly financial literacy tutor.\n"
            "Provide a concise, educational high-level answer from general financial literacy knowledge.\n"
            "Do not provide personalized investment advice.\n"
            "If uncertain, be transparent about limits.\n\n"
            f"Topic: {topic}\n"
            f"User question: {question}"
        )
        reply = self._smart_llm.invoke(prompt)
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return (
            "I couldn't find strong supporting documents after trying alternate retrieval phrasing, "
            "so here's a general educational explanation:\n\n"
            f"{content}"
        )

    def _generate_follow_up_payload(
        self,
        state: ChatState,
        topic: str,
        user_query: str,
        mode: str,
        max_suggestions: int,
    ) -> tuple[str, list[str]]:
        recent_messages = state.get("messages", [])[-10:]
        history_lines: list[str] = []
        for msg in recent_messages:
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", "")).strip()
            if content:
                history_lines.append(f"{role}: {content}")
        user_history = "\n".join(history_lines)

        last_assistant_message = ""
        for msg in reversed(state.get("messages", [])):
            if str(msg.get("role", "")) == "assistant":
                last_assistant_message = str(msg.get("content", "")).strip()
                break

        response_draft = str(state.get("response_draft", "")).strip()
        structured_task_status = str(state.get("structured_task_status", "idle"))
        last_completed_task = str(state.get("last_completed_structured_task", ""))

        retrieved_brief = []
        for d in state.get("retrieved_chunks", [])[:5]:
            retrieved_brief.append(
                {
                    "title": d.get("title", "untitled"),
                    "source": d.get("source", "unknown"),
                    "topic": d.get("topic", topic),
                }
            )

        mode_instructions = {
            "follow_up": "User request is broad. Ask a narrowing question and provide 2-5 focused options.",
            "post_explain": "A knowledge explanation just happened. Provide exactly 2 sensible next-learning options.",
            "post_tool_completion": "A tool/calculation/assessment just completed. Provide exactly 2 next-step options that build on that result.",
        }

        prompt = (
            "You are generating follow-up UX prompts for a financial literacy chatbot.\n"
            "Return STRICT JSON only with this shape:\n"
            "{\"question\": string, \"suggestions\": [{\"text\": string, \"type\": \"content\"|\"tool\"}, ...]}\n"
            "No markdown. No extra text.\n\n"
            "Constraints:\n"
            "- question: one concise beginner-friendly question\n"
            f"- suggestions: 1 to {max_suggestions} short clickable user replies\n"
            "- for mode=follow_up, provide 2 to 5 suggestions\n"
            "- for mode=post_explain or mode=post_tool_completion, provide exactly 2 suggestions\n"
            "- Suggestions must be context-aware and clearly connected to the latest assistant response and recent conversation.\n"
            "- Each suggestion MUST include type='content' or type='tool'.\n"
            "- At most ONE suggestion may have type='tool'.\n"
            "- For tool type, only suggest existing tools: risk tolerance quiz, budget calculation, debt example calculation, insurance-needs assessment.\n"
            "- Avoid overlap/redundancy across suggestions.\n"
            "- Use learning phrasing (e.g., 'Learn about ...') rather than operational product actions.\n"
            "- suggestions must be concrete and immediately actionable\n"
            "- if topic is money_management, suggestions should prioritize:\n"
            "  budgeting/spending, credit-vs-debit, loans/borrowing, managing debt, insurance basics\n"
            "- include tool-oriented options when relevant: risk tolerance quiz, budget calculation, debt example calculation, insurance-needs assessment\n"
            "- do not suggest unsupported capabilities\n"
            "- avoid personalized investment advice\n\n"
            "Few-shot examples:\n"
            "Input:\n"
            "mode=follow_up\n"
            "topic=money_management\n"
            "user=I want to learn money management\n"
            "Output:\n"
            "{\"question\":\"Great choice—what part of money management should we start with?\",\"suggestions\":[{\"text\":\"Learn about budgeting and spending\",\"type\":\"content\"},{\"text\":\"Learn about credit cards vs debit cards\",\"type\":\"content\"},{\"text\":\"Learn about loans and borrowing\",\"type\":\"content\"},{\"text\":\"Learn about managing debt\",\"type\":\"content\"},{\"text\":\"Learn about insurance basics\",\"type\":\"content\"}]}\n\n"
            "Input:\n"
            "mode=follow_up\n"
            "topic=investment_education\n"
            "user=Teach me investing\n"
            "Output:\n"
            "{\"question\":\"Nice—what do you want to focus on first in investment education?\",\"suggestions\":[{\"text\":\"Learn about portfolio balancing\",\"type\":\"content\"},{\"text\":\"Learn about financial products\",\"type\":\"content\"},{\"text\":\"Learn about risk vs return basics\",\"type\":\"content\"}]}\n\n"
            "Input:\n"
            "mode=post_explain\n"
            "topic=money_management\n"
            "user=What is an emergency fund?\n"
            "Output:\n"
            "{\"question\":\"Want to go one step deeper?\",\"suggestions\":[{\"text\":\"Learn about emergency fund sizing\",\"type\":\"content\"},{\"text\":\"Run budget calculation\",\"type\":\"tool\"}]}\n\n"
            "Input:\n"
            "mode=post_tool_completion\n"
            "topic=money_management\n"
            "user=cool\n"
            "Output:\n"
            "{\"question\":\"Nice—what do you want to do next?\",\"suggestions\":[{\"text\":\"Learn about how loan interest works\",\"type\":\"content\"},{\"text\":\"Run debt example calculation\",\"type\":\"tool\"}]}\n\n"
            f"Now generate output for:\nmode={mode}\ntopic={topic}\n"
            f"mode_instruction={mode_instructions.get(mode, '')}\n"
            f"user={user_query}\n"
            f"recent_user_history={user_history}\n"
            f"last_assistant_message={last_assistant_message}\n"
            f"latest_response_draft={response_draft}\n"
            f"structured_task_status={structured_task_status}\n"
            f"last_completed_structured_task={last_completed_task}\n"
            f"retrieved_documents_brief={json.dumps(retrieved_brief)}"
        )

        reply = self._fast_llm.invoke(prompt)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        parsed = self._parse_follow_up_json(text)
        if not parsed:
            repair_prompt = (
                "Convert the following content into STRICT JSON with shape "
                '{"question": string, "suggestions": [{"text": string, "type": "content"|"tool"}, ...]}. '
                "No markdown. No extra text.\n\n"
                f"Content:\n{text}"
            )
            repaired = self._fast_llm.invoke(repair_prompt)
            repaired_text = repaired.content if isinstance(repaired.content, str) else str(repaired.content)
            parsed = self._parse_follow_up_json(repaired_text)

        if parsed:
            question, suggestions = parsed
            normalized = self._normalize_follow_up_suggestions(suggestions)
            deduped = [item["text"] for item in normalized]

            if question.strip():
                if mode == "follow_up":
                    return question.strip(), deduped[:max_suggestions]
                return question.strip(), deduped[:2]

        fallback_question = "Could you share what you want to do next so I can guide you better?"
        return fallback_question, []

    def _normalize_follow_up_suggestions(self, suggestions: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        tool_count = 0

        for item in suggestions:
            text = str(item.get("text", "")).strip()
            s_type = str(item.get("type", "content")).strip().lower()
            if not text:
                continue
            if s_type not in {"content", "tool"}:
                s_type = "content"

            key = text.lower()
            if key in seen:
                continue
            seen.add(key)

            if s_type == "tool":
                if not self._is_supported_tool_suggestion(text):
                    continue
                tool_count += 1
                if tool_count > 1:
                    continue

            cleaned.append({"text": text, "type": s_type})

        return cleaned

    def _is_supported_tool_suggestion(self, text: str) -> bool:
        lower = text.lower()
        return any(
            x in lower
            for x in [
                "risk tolerance quiz",
                "budget calculation",
                "debt example calculation",
                "insurance-needs assessment",
                "insurance needs assessment",
            ]
        )

    def _parse_follow_up_json(self, text: str) -> tuple[str, list[dict[str, str]]] | None:
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

            question = str(data.get("question", "")).strip()
            suggestions_raw = data.get("suggestions", [])
            if not isinstance(suggestions_raw, list):
                continue

            suggestions: list[dict[str, str]] = []
            for item in suggestions_raw:
                if isinstance(item, dict):
                    text_val = str(item.get("text", "")).strip()
                    type_val = str(item.get("type", "content")).strip().lower()
                    if text_val:
                        suggestions.append({"text": text_val, "type": type_val})
                else:
                    # Backward compatibility if model returns string list.
                    s = str(item).strip()
                    if s:
                        suggestions.append({"text": s, "type": "content"})

            if question and suggestions:
                return question, suggestions

        return None

    def _grounded_answer(self, state: ChatState, topic: str, docs: list[dict[str, object]]) -> str:
        context = "\n\n".join(
            (
                f"Research document {i}:\n"
                f"- title: {d.get('title', 'untitled')}\n"
                f"- topic: {d.get('topic', topic)}\n"
                f"- source: {d.get('source', 'unknown')}\n"
                f"- excerpt: {d.get('chunk_text', '')}"
            )
            for i, d in enumerate(docs[:3], start=1)
        )
        state["retrieved_summary"] = context

        prompt = (
            "You are a beginner-friendly financial literacy tutor. "
            "Use only the provided research documents. Keep it practical and clear. "
            "Do not provide personalized investment advice.\n"
            "Important grounding rules:\n"
            "- Do NOT claim ownership of resources from sources (avoid words like 'our guides', 'we provide', 'you can download from us').\n"
            "- Some sources will mention external resources like guides for young adults or guides in various languages. Since we have no access to that, do not bring it up."
            "- If the source mentions resources, present it as source information, not as chatbot-owned offerings.\n"
            "- Do not invent availability of tools, files, links, or downloads beyond what the user explicitly asked for and what context states.\n\n"
            f"User question: {state['user_query']}\n\n"
            f"Research documents:\n{context}\n\n"
            "Write a direct answer in plain language."
        )
        self._emit_status("Drafting answer...")
        if self._token_callback:
            text_out = ""
            for chunk in self._smart_llm.stream(prompt):
                token = self._chunk_to_text(chunk)
                if token:
                    text_out += token
                    self._emit_token(token)
            if text_out.strip():
                return text_out

        reply = self._smart_llm.invoke(prompt)
        return reply.content if isinstance(reply.content, str) else str(reply.content)

    def investment_knowledge(self, state: ChatState) -> ChatState:
        self._emit_status("Researching investment documents...")
        topic = "investment_education"
        active_query = str(state.get("active_query", state.get("user_query", "")))
        docs = self.rag.retrieve_docs(active_query, topic=topic)
        state["retrieved_chunks"] = docs

        supported = self._docs_support_question(str(state.get("user_query", "")), docs)
        if docs and supported:
            state["needs_rephrase"] = False
            state["response_draft"] = self._grounded_answer(state, topic=topic, docs=docs)
            return state

        attempts = int(state.get("rephrase_attempts", 0))
        if attempts < 2:
            state["needs_rephrase"] = True
            state["response_draft"] = ""
            return state

        state["needs_rephrase"] = False
        state["response_draft"] = self._fallback_general_answer(
            question=str(state.get("user_query", "")),
            topic=topic,
        )
        return state

    def money_management_knowledge(self, state: ChatState) -> ChatState:
        self._emit_status("Researching money-management documents...")
        topic = "money_management"
        active_query = str(state.get("active_query", state.get("user_query", "")))
        docs = self.rag.retrieve_docs(active_query, topic=topic)
        state["retrieved_chunks"] = docs

        supported = self._docs_support_question(str(state.get("user_query", "")), docs)
        if docs and supported:
            state["needs_rephrase"] = False
            state["response_draft"] = self._grounded_answer(state, topic=topic, docs=docs)
            return state

        attempts = int(state.get("rephrase_attempts", 0))
        if attempts < 2:
            state["needs_rephrase"] = True
            state["response_draft"] = ""
            return state

        state["needs_rephrase"] = False
        state["response_draft"] = self._fallback_general_answer(
            question=str(state.get("user_query", "")),
            topic=topic,
        )
        return state

    def structured_tools(self, state: ChatState) -> ChatState:
        self._emit_status("Running calculation/assessment tools...")
        result = self._structured_tools_agent.run(state)
        state["response_draft"] = str(result.get("message", "")).strip()
        state["structured_task_status"] = str(result.get("status", "idle"))
        state["structured_task_type"] = str(result.get("task", state.get("task_type", "")))
        return state

    def response_composer(self, state: ChatState) -> ChatState:
        self._emit_status("Composing response...")
        body = state.get("response_draft", "").strip()

        if state.get("task_type") == "explain" and not state.get("follow_up_suggestions"):
            question, suggestions = self._generate_follow_up_payload(
                state=state,
                topic=state.get("topic", "general"),
                user_query=state.get("user_query", ""),
                mode="post_explain",
                max_suggestions=2,
            )
            state["follow_up_suggestions"] = suggestions[:2]
            if question:
                body += f"\n\n{question}"

        if state.get("structured_task_status") == "completed" and not state.get("follow_up_suggestions"):
            question, suggestions = self._generate_follow_up_payload(
                state=state,
                topic=state.get("topic", "general"),
                user_query=state.get("user_query", ""),
                mode="post_tool_completion",
                max_suggestions=2,
            )
            state["follow_up_suggestions"] = suggestions[:2]
            if question:
                body += f"\n\n{question}"

        follow_ups = state.get("follow_up_suggestions", [])
        if follow_ups:
            bullets = "\n".join(f"- {x}" for x in follow_ups)
            body += f"\n\nSuggested follow-ups:\n{bullets}"
        state["response_draft"] = body
        return state
