from __future__ import annotations

import logging
import secrets
from typing import Final

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from fin_lit_chatbot.config import Settings
from fin_lit_chatbot.engine import ChatState, FinLitBot

logger = logging.getLogger(__name__)

FOLLOW_UP_CALLBACK_PREFIX: Final[str] = "fu:"
CONSENT_CALLBACK_PREFIX: Final[str] = "consent:"
MAX_TELEGRAM_MESSAGE_LEN: Final[int] = 4096


def _initial_state() -> ChatState:
    return ChatState(
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
                    "- Money management\n"
                    "- Investment education\n\n"
                    "Tell me which one you want to start with."
                ),
            }
        ],
    )


def _strip_embedded_suggestions(text: str) -> str:
    marker = "\n\nSuggested follow-ups:\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text.strip()


def _chunk_text(text: str, chunk_size: int = MAX_TELEGRAM_MESSAGE_LEN) -> list[str]:
    raw = text.strip()
    if not raw:
        return ["I can help with money management or investment education. What would you like to explore?"]

    chunks: list[str] = []
    current = ""
    for paragraph in raw.split("\n"):
        next_line = paragraph if not current else f"{current}\n{paragraph}"
        if len(next_line) <= chunk_size:
            current = next_line
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > chunk_size:
            chunks.append(paragraph[:chunk_size])
            paragraph = paragraph[chunk_size:]
        current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _follow_up_markup(callback_items: list[tuple[str, str]]) -> InlineKeyboardMarkup | None:
    if not callback_items:
        return None

    keyboard = [
        [InlineKeyboardButton(text=suggestion[:60], callback_data=f"{FOLLOW_UP_CALLBACK_PREFIX}{callback_key}")]
        for suggestion, callback_key in callback_items
    ]
    return InlineKeyboardMarkup(keyboard)


def _consent_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(text="OK", callback_data=f"{CONSENT_CALLBACK_PREFIX}ok"),
                InlineKeyboardButton(text="Reject", callback_data=f"{CONSENT_CALLBACK_PREFIX}reject"),
            ]
        ]
    )


def _build_follow_up_callback_items(
    context: ContextTypes.DEFAULT_TYPE,
    suggestions: list[str],
) -> list[tuple[str, str]]:
    follow_up_map = context.chat_data.get("follow_up_map")
    if not isinstance(follow_up_map, dict):
        follow_up_map = {}
        context.chat_data["follow_up_map"] = follow_up_map

    callback_items: list[tuple[str, str]] = []
    for suggestion in suggestions:
        cleaned = suggestion.strip()
        if not cleaned:
            continue
        callback_key = secrets.token_urlsafe(6)
        follow_up_map[callback_key] = cleaned
        callback_items.append((cleaned, callback_key))

    return callback_items


async def _send_intro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = context.chat_data.get("state")
    if state is None:
        state = _initial_state()
        context.chat_data["state"] = state

    intro = str(state.get("messages", [{}])[-1].get("content", ""))
    suggestions = list(state.get("follow_up_suggestions", []))
    callback_items = _build_follow_up_callback_items(context, suggestions)
    markup = _follow_up_markup(callback_items)

    if update.callback_query and update.callback_query.message:
        await _reply_text(update.callback_query.message, intro, reply_markup=markup)
    elif update.message:
        await _reply_text(update.message, intro, reply_markup=markup)


async def _reply_text(
    message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    try:
        await message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await message.reply_text(text, reply_markup=reply_markup)


async def _send_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: ChatState,
) -> None:
    response_text = _strip_embedded_suggestions(str(state.get("response_draft", "")))
    suggestions = list(state.get("follow_up_suggestions", []))
    callback_items = _build_follow_up_callback_items(context, suggestions)
    markup = _follow_up_markup(callback_items)
    chunks = _chunk_text(response_text)

    for i, chunk in enumerate(chunks):
        reply_markup = markup if i == len(chunks) - 1 else None
        if update.callback_query:
            await _reply_text(update.callback_query.message, chunk, reply_markup=reply_markup)
        elif update.message:
            await _reply_text(update.message, chunk, reply_markup=reply_markup)


async def _process_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_prompt: str,
) -> None:
    bot = context.application.bot_data["finlit_bot"]
    state = context.chat_data.get("state")
    if state is None:
        state = _initial_state()

    new_state = bot.respond(state, user_prompt, channel="telegram")
    context.chat_data["state"] = new_state
    await _send_response(update, context, new_state)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data.clear()
    context.chat_data["state"] = _initial_state()
    context.chat_data["follow_up_map"] = {}
    context.chat_data["disclaimer_pending"] = True
    context.chat_data["disclaimer_accepted"] = False

    disclaimer = (
        "This chatbot provides general financial information only and does not constitute financial advice. "
        "The information does not consider your personal financial situation or objectives."
    )

    if update.message:
        await _reply_text(update.message, disclaimer, reply_markup=_consent_markup())


async def handle_disclaimer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()
    if not query.data.startswith(CONSENT_CALLBACK_PREFIX):
        return

    action = query.data.removeprefix(CONSENT_CALLBACK_PREFIX)
    if action == "ok":
        context.chat_data["disclaimer_pending"] = False
        context.chat_data["disclaimer_accepted"] = True
        if query.message:
            await _reply_text(query.message, "Thanks for confirming.")
        await _send_intro(update, context)
        return

    context.chat_data["disclaimer_pending"] = False
    context.chat_data["disclaimer_accepted"] = False
    if query.message:
        await _reply_text(query.message, "Understood. I won’t proceed. Send /start anytime if you change your mind.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Send any finance question, or tap a suggested follow-up button. Use /start to reset the session."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    if context.chat_data.get("disclaimer_pending", False):
        await _reply_text(update.message, "Please tap OK or Reject on the disclaimer above first.")
        return

    if not context.chat_data.get("disclaimer_accepted", False):
        await _reply_text(update.message, "Please send /start when you are ready.")
        return

    await _process_prompt(update, context, update.message.text.strip())


async def handle_follow_up_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()
    if not query.data.startswith(FOLLOW_UP_CALLBACK_PREFIX):
        return

    callback_key = query.data.removeprefix(FOLLOW_UP_CALLBACK_PREFIX)
    follow_up_map = context.chat_data.get("follow_up_map")
    selected = follow_up_map.get(callback_key) if isinstance(follow_up_map, dict) else None

    if not selected:
        await query.answer("That option is no longer available. Please pick a newer option.", show_alert=True)
        return

    if query.message:
        await _reply_text(query.message, f"Selected option: {selected}")

    await _process_prompt(update, context, selected)


def main() -> None:
    settings = Settings()
    token = settings.telegram_bot_token
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment variables.")

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    app = Application.builder().token(token).build()
    app.bot_data["finlit_bot"] = FinLitBot()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(handle_disclaimer_callback, pattern=r"^consent:(ok|reject)$"))
    app.add_handler(CallbackQueryHandler(handle_follow_up_callback, pattern=r"^fu:[A-Za-z0-9_-]+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Starting Telegram bot polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
