from __future__ import annotations

import streamlit as st

from fin_lit_chatbot.engine import ChatState, FinLitBot

st.set_page_config(page_title="FinLit Tutor", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 980px;}
    .small-note {color: #6b7280; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False
if "disclaimer_exited" not in st.session_state:
    st.session_state.disclaimer_exited = False

if not st.session_state.disclaimer_accepted and not st.session_state.disclaimer_exited:
    st.title("Disclaimer")
    st.info(
        "Before we begin:\n"
        "This chatbot provides general financial information only and does not constitute financial advice. "
        "We are not regulated by the Monetary Authority of Singapore (MAS) as a financial adviser. "
        "The information provided does not consider your personal financial situation or objectives.\n\n"
        "Do you acknowledge and wish to continue?"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, continue", use_container_width=True, type="primary"):
            st.session_state.disclaimer_accepted = True
            st.session_state.disclaimer_exited = False
            st.rerun()
    with col2:
        if st.button("Exit", use_container_width=True):
            st.session_state.disclaimer_accepted = False
            st.session_state.disclaimer_exited = True
            st.rerun()

    st.stop()

if st.session_state.disclaimer_exited:
    st.warning("You chose to exit. Refresh the page if you want to continue later.")
    st.stop()

if "bot" not in st.session_state:
    st.session_state.bot = FinLitBot()
if "state" not in st.session_state:
    st.session_state.state = ChatState(
        follow_up_suggestions=[
            "I want to start with money management",
            "I want to start with investment education",
        ],
        messages=[
            {
                "role": "assistant",
                "content": (
                    "Hi! I’m your financial literacy tutor.\n\n"
                    "We can start with either:\n"
                    "- **Money management**\n"
                    "- **Investment education**\n\n"
                    "Tell me which one you want to start with."
                ),
            }
        ]
    )

bot: FinLitBot = st.session_state.bot
state: ChatState = st.session_state.state

st.title("💬 Financial Literacy Tutor")
st.caption("Beginner-friendly finance education with LangGraph + Chroma + OpenRouter")

with st.sidebar:
    st.subheader("Quick actions")
    if st.button("Start risk quiz", use_container_width=True):
        st.session_state.queued_user_prompt = "Start risk quiz"
    if st.button("Build a simple budget", use_container_width=True):
        st.session_state.queued_user_prompt = "Help me make a budget"
    if st.button("Stocks vs bonds", use_container_width=True):
        st.session_state.queued_user_prompt = "What is the difference between stocks and bonds?"

    st.divider()
    st.markdown("### Session snapshot")
    st.json(
        {
            "topic": state.get("topic"),
            "task_type": state.get("task_type"),
            "risk_profile_result": state.get("risk_profile_result"),
            "session_financials": state.get("session_financials", {}),
        }
    )

for msg in state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

suggestions = state.get("follow_up_suggestions", [])
messages = state.get("messages", [])
if suggestions and messages and messages[-1].get("role") == "assistant":
    st.markdown("#### Suggested replies")
    cols = st.columns(min(3, len(suggestions)))
    for i, suggestion in enumerate(suggestions):
        col = cols[i % len(cols)]
        with col:
            if st.button(suggestion, use_container_width=True, key=f"followup_{len(messages)}_{i}_{suggestion}"):
                st.session_state.queued_user_prompt = suggestion

prefill = st.session_state.pop("queued_user_prompt", None)
user_prompt = st.chat_input("Ask about investing, budgeting, debt, cards, or insurance")
if prefill and not user_prompt:
    user_prompt = prefill

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.empty()
        response_box = st.empty()
        streamed = {"text": ""}

        def on_status(msg: str) -> None:
            status_box.info(msg)

        def on_token(token: str) -> None:
            streamed["text"] += token
            response_box.markdown(streamed["text"] + "▌")

        new_state = bot.respond_live(
            state,
            user_prompt,
            on_status=on_status,
            on_token=on_token,
        )

        status_box.empty()
        response_box.markdown(new_state.get("response_draft", ""))

    st.session_state.state = new_state
    st.rerun()

st.markdown('<p class="small-note">For educational purposes only. Not personalized financial advice.</p>', unsafe_allow_html=True)
