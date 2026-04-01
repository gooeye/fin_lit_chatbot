from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from typing import Final

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
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
from fin_lit_chatbot.payloads import (
    normalize_message_payload,
    normalize_payload_list,
    payload_from_text,
    payload_text,
)
from fin_lit_chatbot.schema import UserInput

logger = logging.getLogger(__name__)

FOLLOW_UP_CALLBACK_PREFIX: Final[str] = "fu:"
CONSENT_CALLBACK_PREFIX: Final[str] = "consent:"
MAX_TELEGRAM_MESSAGE_LEN: Final[int] = 4096
PROGRESS_SPINNER_FRAMES: Final[tuple[str, ...]] = ("⠋", "⠙", "⠸", "⠴", "⠦", "⠇")


def _initial_state() -> ChatState:
    return ChatState(
        follow_up_suggestions=[
            payload_from_text(
                "I want to start with money management",
                code="topic.start.money_management",
            ),
            payload_from_text(
                "I want to start with investment education",
                code="topic.start.investment_education",
            ),
        ],
        messages=[
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm your financial literacy tutor.\n\n"
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
    suggestions: list[object],
) -> list[tuple[str, str]]:
    follow_up_map = context.chat_data.get("follow_up_map")
    if not isinstance(follow_up_map, dict):
        follow_up_map = {}
        context.chat_data["follow_up_map"] = follow_up_map

    callback_items: list[tuple[str, str]] = []
    for suggestion in normalize_payload_list(suggestions):
        cleaned = payload_text(suggestion)
        if not cleaned:
            continue
        callback_key = secrets.token_urlsafe(6)
        follow_up_map[callback_key] = suggestion
        callback_items.append((cleaned, callback_key))

    return callback_items


async def _send_intro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = context.chat_data.get("state")
    if state is None:
        state = _initial_state()
        context.chat_data["state"] = state

    intro = str(state.get("messages", [{}])[-1].get("content", ""))
    suggestions = normalize_payload_list(state.get("follow_up_suggestions", []))
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
) -> object | None:
    try:
        return await message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        return await message.reply_text(text, reply_markup=reply_markup)


def _progress_text(status: str, frame_index: int = 0) -> str:
    clean_status = status.strip() or "Thinking..."
    frame = PROGRESS_SPINNER_FRAMES[frame_index % len(PROGRESS_SPINNER_FRAMES)]
    return f"{frame} {clean_status}"


async def _send_typing_action(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
        pass


def _collect_progress_events(
    bot: FinLitBot,
    state: ChatState,
    user_input: UserInput,
    loop: asyncio.AbstractEventLoop,
    progress_queue: asyncio.Queue,
) -> None:
    final_state: ChatState = state
    try:
        for event in bot.respond_with_progress(state, user_input, channel="telegram"):
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("type", "")).strip()
            if event_type == "status":
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    ("status", str(event.get("message", ""))),
                )
                continue

            if event_type == "final":
                maybe_state = event.get("state")
                if isinstance(maybe_state, dict):
                    final_state = maybe_state

        loop.call_soon_threadsafe(progress_queue.put_nowait, ("final", final_state))
    except Exception as exc:
        loop.call_soon_threadsafe(progress_queue.put_nowait, ("error", exc))


async def _animate_progress_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    progress_message,
    progress_state: dict[str, str],
    stop_event: asyncio.Event,
    *,
    initial_frame_index: int = 0,
    initial_text: str = "",
) -> None:
    frame_index = initial_frame_index
    tick_count = 0
    last_text = initial_text

    while not stop_event.is_set():
        current_text = _progress_text(progress_state.get("status", ""), frame_index)
        if current_text != last_text:
            await _update_progress_message(progress_message, current_text)
            last_text = current_text

        if tick_count % 8 == 0:
            await _send_typing_action(context, chat_id)

        frame_index += 1
        tick_count += 1

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.3)
        except asyncio.TimeoutError:
            continue


async def _update_progress_message(progress_message, text: str) -> None:
    if progress_message is None:
        return
    try:
        await progress_message.edit_text(text)
    except Exception:
        pass


async def _delete_progress_message(progress_message) -> None:
    if progress_message is None:
        return
    try:
        await progress_message.delete()
    except Exception:
        pass


async def _send_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: ChatState,
) -> None:
    response_text = _strip_embedded_suggestions(str(state.get("response_draft", "")))
    suggestions = normalize_payload_list(state.get("follow_up_suggestions", []))
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
    user_input: UserInput,
) -> None:
    bot = context.application.bot_data["finlit_bot"]
    state = context.chat_data.get("state")
    if state is None:
        state = _initial_state()

    reply_target = update.callback_query.message if update.callback_query else update.message
    progress_message = None
    stop_progress_event = asyncio.Event()
    progress_task: asyncio.Task | None = None
    progress_state = {"status": "Starting..."}
    final_state = state
    progress_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    if reply_target is not None:
        await _send_typing_action(context, reply_target.chat_id)
        initial_progress_text = _progress_text(progress_state["status"], 0)
        progress_message = await _reply_text(reply_target, initial_progress_text)
        progress_task = asyncio.create_task(
            _animate_progress_message(
                context,
                reply_target.chat_id,
                progress_message,
                progress_state,
                stop_progress_event,
                initial_frame_index=1,
                initial_text=initial_progress_text,
            )
        )

    worker_task = asyncio.create_task(
        asyncio.to_thread(
            _collect_progress_events,
            bot,
            state,
            user_input,
            loop,
            progress_queue,
        )
    )

    try:
        while True:
            event_type, payload = await progress_queue.get()
            if event_type == "status":
                progress_state["status"] = str(payload)
                continue

            if event_type == "final":
                if isinstance(payload, dict):
                    final_state = payload
                break

            if event_type == "error":
                if isinstance(payload, Exception):
                    raise payload
                raise RuntimeError("Unexpected error while generating Telegram response.")

        await worker_task
    finally:
        stop_progress_event.set()
        if progress_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task
        await _delete_progress_message(progress_message)

    context.chat_data["state"] = final_state
    await _send_response(update, context, final_state)


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
        await _reply_text(query.message, "Understood. I won't proceed. Send /start anytime if you change your mind.")


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

    selected_payload = normalize_message_payload(selected) or payload_from_text(str(selected).strip())
    selected_text = payload_text(selected_payload)

    if query.message:
        await _reply_text(query.message, f"Selected option: {selected_text}")

    await _process_prompt(update, context, selected_payload)


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
