"""Spawns the real local MCP server subprocess (stdio) and exercises the full
discovery -> Gemini-schema-conversion -> call round trip. No Postgres or Gemini
API key needed - only kb_search touches those, and it isn't called here - so
this runs in normal CI, not just the docker-compose integration lane.
"""

from __future__ import annotations

from vocare.agent.mcp_client import MCPToolClient, optional_mcp_client
from vocare.config import Settings


async def test_discovers_expected_tools() -> None:
    async with MCPToolClient() as client:
        names = {decl.name for decl in client.function_declarations()}
    assert names == {
        "calculate",
        "get_current_time",
        "device_status",
        "device_control",
        "list_devices",
        "kb_search",
    }


async def test_calculate_tool_round_trip() -> None:
    async with MCPToolClient() as client:
        result = await client.call_tool("calculate", {"expression": "6 * 7"})
    assert "42" in result


async def test_device_tools_round_trip() -> None:
    async with MCPToolClient() as client:
        listed = await client.call_tool("list_devices", {})
        assert "autodose-01" in listed

        paused = await client.call_tool(
            "device_control", {"device_id": "autodose-01", "action": "pause"}
        )
        assert "paused" in paused

        status = await client.call_tool("device_status", {"device_id": "autodose-01"})
        assert "paused" in status


async def test_unknown_device_reports_error_text_not_a_crash() -> None:
    async with MCPToolClient() as client:
        result = await client.call_tool("device_status", {"device_id": "no-such-device"})
    assert "error" in result.lower() or "unknown" in result.lower()


async def test_optional_mcp_client_yields_none_when_tools_disabled() -> None:
    settings = Settings(gemini_api_key="test-key", vocare_enable_tools=False)
    async with optional_mcp_client(settings) as client:
        assert client is None


async def test_optional_mcp_client_connects_when_tools_enabled() -> None:
    settings = Settings(gemini_api_key="test-key", vocare_enable_tools=True)
    async with optional_mcp_client(settings) as client:
        assert client is not None
        assert "calculate" in {decl.name for decl in client.function_declarations()}
