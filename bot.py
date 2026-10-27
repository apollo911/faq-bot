"""Telegram FAQ bot: entry point and message handlers."""

import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from knowledge_base import KnowledgeBaseError, load_knowledge_base
from llm import FaqAssistant

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
TELEGRAM_MESSAGE_LIMIT = 4096

GREETING = (
    "Здравствуйте! Я ИИ-консультант компании. "
    "Отвечаю на вопросы о доставке, оплате, гарантии, возврате и работе магазина. "
    "Напишите свой вопрос."
)
TEXT_ONLY_REPLY = "Пожалуйста, напишите вопрос текстом — я пока понимаю только текстовые сообщения."

router = Router()


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split text into Telegram-sized chunks, preferring line breaks as split points."""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(GREETING)


@router.message(F.text)
async def handle_question(message: Message, assistant: FaqAssistant, knowledge_dir: Path) -> None:
    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            # Read on every question so new files in knowledge/ work without a restart.
            knowledge_base = await asyncio.to_thread(load_knowledge_base, knowledge_dir)
        except KnowledgeBaseError:
            logger.exception("Could not load the knowledge base")
            reply = assistant.fallback_message
        else:
            reply = await assistant.answer(message.text, knowledge_base)

    for chunk in split_message(reply):
        await message.answer(chunk)


@router.message()
async def handle_non_text(message: Message) -> None:
    await message.answer(TEXT_ONLY_REPLY)


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Set {name} in the .env file (see .env.example)")
    return value


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # HTTP client logs include request URLs; keep them quiet so nothing sensitive leaks into logs.
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    telegram_token = get_required_env("TELEGRAM_BOT_TOKEN")
    get_required_env("ANTHROPIC_API_KEY")  # read by AsyncAnthropic() itself
    model = get_required_env("LLM_MODEL")
    manager_contact = get_required_env("MANAGER_CONTACT")

    # Fail at startup, not on the first customer question, if knowledge/ is empty.
    load_knowledge_base(KNOWLEDGE_DIR)

    dispatcher = Dispatcher(
        assistant=FaqAssistant(AsyncAnthropic(), model, manager_contact),
        knowledge_dir=KNOWLEDGE_DIR,
    )
    dispatcher.include_router(router)

    bot = Bot(token=telegram_token)
    logger.info("Bot started, model %s", model)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
