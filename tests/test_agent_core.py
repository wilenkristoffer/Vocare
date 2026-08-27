from __future__ import annotations

import uuid

import pytest
from google import genai

from vocare.agent.core import NO_CONTEXT_NOTE, Agent, _format_context, _format_history
from vocare.config import Settings
from vocare.rag.models import RetrievedChunk


class StubMCPClient:
    def function_declarations(self) -> list:
        return []

    async def call_tool(self, name: str, arguments: dict) -> str:
        raise RuntimeError("boom")


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_api_key="test-key")


def test_format_context_below_threshold_returns_fallback_note(settings: Settings) -> None:
    hits = [RetrievedChunk(title="x", content="y", source="s", similarity=0.1)]
    assert _format_context(hits, settings.rag_min_similarity) == NO_CONTEXT_NOTE


def test_format_context_includes_hits_above_threshold(settings: Settings) -> None:
    hits = [RetrievedChunk(title="Doc A", content="useful info", source="s", similarity=0.9)]
    result = _format_context(hits, settings.rag_min_similarity)
    assert "Doc A" in result
    assert "useful info" in result


def test_format_context_filters_out_low_similarity_hits(settings: Settings) -> None:
    hits = [
        RetrievedChunk(title="Good", content="keep me", source="s", similarity=0.9),
        RetrievedChunk(title="Bad", content="drop me", source="s", similarity=0.05),
    ]
    result = _format_context(hits, settings.rag_min_similarity)
    assert "keep me" in result
    assert "drop me" not in result


def test_format_history_empty_when_no_good_hits(settings: Settings) -> None:
    hits = [RetrievedChunk(title="past", content="irrelevant", source="s", similarity=0.1)]
    assert _format_history(hits, settings.rag_min_similarity) == ""


def test_format_history_includes_good_hits(settings: Settings) -> None:
    hits = [RetrievedChunk(title="past", content="remembered detail", source="s", similarity=0.9)]
    result = _format_history(hits, settings.rag_min_similarity)
    assert "remembered detail" in result


async def test_execute_tool_wraps_exceptions_as_text_not_a_raise(settings: Settings) -> None:
    client = genai.Client(api_key=settings.gemini_api_key)
    agent = Agent(client, settings, StubMCPClient(), uuid.uuid4(), mode="text")
    result = await agent._execute_tool("some_tool", {"a": 1})
    assert "error calling tool" in result
    assert "boom" in result


async def test_agent_works_with_no_mcp_client_tool_calling_disabled(settings: Settings) -> None:
    """VOCARE_ENABLE_TOOLS=false path: the agent must construct and respond to
    tool-call attempts gracefully with mcp_client=None, not just when a real
    (or stub) MCP client is present."""
    client = genai.Client(api_key=settings.gemini_api_key)
    agent = Agent(client, settings, None, uuid.uuid4(), mode="text")
    assert agent.mcp_client is None
    result = await agent._execute_tool("some_tool", {"a": 1})
    assert "tool calling is disabled" in result
