from unittest.mock import AsyncMock, MagicMock

import bot
from bot import TEXT_ONLY_REPLY, handle_non_text, handle_question, handle_start, split_message

FALLBACK = "Свяжитесь с менеджером"


def make_message(text="Сколько стоит доставка?"):
    message = MagicMock()
    message.text = text
    message.chat.id = 42
    message.bot = AsyncMock()
    message.answer = AsyncMock()
    return message


def make_assistant(answer="Доставка стоит 490 рублей."):
    assistant = MagicMock()
    assistant.answer = AsyncMock(return_value=answer)
    assistant.fallback_message = FALLBACK
    return assistant


async def test_start_sends_greeting():
    message = make_message("/start")

    await handle_start(message)

    message.answer.assert_awaited_once_with(bot.GREETING)


async def test_question_is_answered_from_knowledge_base(tmp_path):
    (tmp_path / "faq.txt").write_text("Доставка 490 рублей.", encoding="utf-8")
    message = make_message()
    assistant = make_assistant()

    await handle_question(message, assistant=assistant, knowledge_dir=tmp_path)

    question, knowledge = assistant.answer.await_args.args
    assert question == "Сколько стоит доставка?"
    assert "Доставка 490 рублей." in knowledge
    message.answer.assert_awaited_once_with("Доставка стоит 490 рублей.")


async def test_missing_knowledge_base_sends_fallback(tmp_path):
    message = make_message()
    assistant = make_assistant()

    await handle_question(message, assistant=assistant, knowledge_dir=tmp_path / "missing")

    assistant.answer.assert_not_awaited()
    message.answer.assert_awaited_once_with(FALLBACK)


async def test_long_answer_is_split(tmp_path):
    (tmp_path / "faq.txt").write_text("Ответ", encoding="utf-8")
    message = make_message()
    assistant = make_assistant(answer="а" * 5000)

    await handle_question(message, assistant=assistant, knowledge_dir=tmp_path)

    sent = [call.args[0] for call in message.answer.await_args_list]
    assert len(sent) == 2
    assert "".join(sent) == "а" * 5000


async def test_non_text_message_asks_for_text():
    message = make_message(text=None)

    await handle_non_text(message)

    message.answer.assert_awaited_once_with(TEXT_ONLY_REPLY)


def test_split_message_keeps_short_text():
    assert split_message("Короткий ответ") == ["Короткий ответ"]


def test_split_message_prefers_line_breaks():
    text = "первая строка\nвторая строка"

    assert split_message(text, limit=20) == ["первая строка", "вторая строка"]


def test_split_message_respects_limit():
    chunks = split_message("x" * 25, limit=10)

    assert chunks == ["x" * 10, "x" * 10, "x" * 5]
