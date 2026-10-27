from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx2
import pytest

from llm import FaqAssistant

MANAGER = "@manager"
KNOWLEDGE = '<document name="faq.txt">\nДоставка 490 рублей.\n</document>'


def make_response(text="Доставка стоит 490 рублей.", stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def make_assistant(create):
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return FaqAssistant(client, "claude-haiku-5-5", MANAGER)


async def test_returns_text_from_claude():
    assistant = make_assistant(AsyncMock(return_value=make_response()))

    answer = await assistant.answer("Сколько стоит доставка?", KNOWLEDGE)

    assert answer == "Доставка стоит 490 рублей."


async def test_request_contains_rules_knowledge_and_question():
    create = AsyncMock(return_value=make_response())
    assistant = make_assistant(create)

    await assistant.answer("Сколько стоит доставка?", KNOWLEDGE)

    request = create.await_args.kwargs
    rules, knowledge = request["system"]
    assert request["model"] == "claude-haiku-5-5"
    assert "Never invent prices" in rules["text"]
    assert MANAGER in rules["text"]
    assert KNOWLEDGE in knowledge["text"]
    assert request["messages"] == [{"role": "user", "content": "Сколько стоит доставка?"}]


async def test_knowledge_base_is_cached():
    create = AsyncMock(return_value=make_response())
    assistant = make_assistant(create)

    await assistant.answer("Вопрос", KNOWLEDGE)

    rules, knowledge = create.await_args.kwargs["system"]
    assert knowledge["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in rules


async def test_refusal_returns_fallback():
    assistant = make_assistant(AsyncMock(return_value=make_response(text="", stop_reason="refusal")))

    answer = await assistant.answer("Вопрос", KNOWLEDGE)

    assert answer == assistant.fallback_message
    assert MANAGER in answer


async def test_empty_answer_returns_fallback():
    assistant = make_assistant(AsyncMock(return_value=make_response(text="  ")))

    answer = await assistant.answer("Вопрос", KNOWLEDGE)

    assert answer == assistant.fallback_message


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APIConnectionError(request=REQUEST),
        anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=REQUEST), body=None),
        anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=REQUEST), body=None),
        anthropic.InternalServerError("server error", response=httpx2.Response(500, request=REQUEST), body=None),
    ],
)
async def test_api_errors_return_fallback(error):
    assistant = make_assistant(AsyncMock(side_effect=error))

    answer = await assistant.answer("Вопрос", KNOWLEDGE)

    assert answer == assistant.fallback_message
