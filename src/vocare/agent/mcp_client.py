from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vocare.config import Settings
from vocare.logging_config import get_logger

logger = get_logger(__name__)

_JSON_TYPE_TO_GEMINI = {
    "object": types.Type.OBJECT,
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "array": types.Type.ARRAY,
}


def json_schema_to_gemini_schema(schema: dict[str, Any]) -> types.Schema:
    """Convert an MCP tool's JSON-Schema `inputSchema` into a Gemini `types.Schema`.

    Only handles the subset both sides actually use for simple tool
    parameters (object/string/number/integer/boolean/array, properties,
    required, items, description) - MCP tool schemas here are hand-written
    and simple, this isn't a general JSON-Schema-to-Gemini-Schema converter.
    """
    json_type = schema.get("type", "object")
    gemini_type = _JSON_TYPE_TO_GEMINI.get(json_type, types.Type.STRING)

    kwargs: dict[str, Any] = {"type": gemini_type}
    if "description" in schema:
        kwargs["description"] = schema["description"]
    if json_type == "object" and "properties" in schema:
        kwargs["properties"] = {
            name: json_schema_to_gemini_schema(prop)
            for name, prop in schema["properties"].items()
        }
        if schema.get("required"):
            kwargs["required"] = schema["required"]
    if json_type == "array" and "items" in schema:
        kwargs["items"] = json_schema_to_gemini_schema(schema["items"])
    return types.Schema(**kwargs)


class MCPToolClient:
    """Spawns the local MCP tool server (stdio transport) and exposes its tools
    as Gemini FunctionDeclarations, plus a way to actually call them.

    This is the adapter described in plan.md: agent -> MCP tool discovery ->
    Gemini function-calling -> tool execution -> result back to the model.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._mcp_tools: list[Any] = []

    async def connect(self) -> None:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "vocare.mcp_server.server"],
        )
        read, write = await self._stack.enter_async_context(stdio_client(server_params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        tools_response = await session.list_tools()
        self._mcp_tools = tools_response.tools
        logger.info("mcp_tools_discovered", tools=[t.name for t in self._mcp_tools])

    async def aclose(self) -> None:
        await self._stack.aclose()

    def function_declarations(self) -> list[types.FunctionDeclaration]:
        return [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=json_schema_to_gemini_schema(tool.input_schema),
            )
            for tool in self._mcp_tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._session is not None, "call connect() first"
        result = await self._session.call_tool(name, arguments)
        text_parts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(text_parts) if text_parts else str(result.content)

    async def __aenter__(self) -> MCPToolClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


@asynccontextmanager
async def optional_mcp_client(settings: Settings) -> AsyncIterator[MCPToolClient | None]:
    """Yields a connected MCPToolClient, or None when tool-calling is turned
    off (VOCARE_ENABLE_TOOLS=false) - callers pass whatever they get straight
    to Agent, which treats None as "no tools for this session," not an error.
    """
    if not settings.vocare_enable_tools:
        yield None
        return
    async with MCPToolClient() as client:
        yield client
