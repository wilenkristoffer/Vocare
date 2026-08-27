"""Local MCP tool server for Vocare.

Run as a subprocess over stdio - the agent (agent/mcp_client.py) spawns this
module with `python -m vocare.mcp_server.server` and talks MCP over
stdin/stdout. There is no network listener here; this is the "local tool
server" MCP pattern, not a hosted MCP service.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from vocare.config import get_settings
from vocare.gemini_client import embed_query, make_client
from vocare.mcp_server import tools_state
from vocare.rag import store
from vocare.rag.db import session_scope

# mcp>=2.0 renamed FastMCP -> MCPServer; same decorator-based API otherwise.
mcp = MCPServer("vocare-tools")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "12 * (3 + 4)". Supports
    + - * / ** % // and parentheses only."""
    try:
        result = tools_state.calculate(expression)
    except tools_state.CalculationError as exc:
        return f"error: {exc}"
    return str(result)


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date/time in an IANA timezone (e.g. "UTC", "Europe/Stockholm")."""
    try:
        return tools_state.get_current_time(timezone)
    except ValueError as exc:
        return f"error: {exc}"


@mcp.tool()
def device_status(device_id: str) -> str:
    """Get the status of a mock AutoDose unit, e.g. "autodose-01"."""
    try:
        status = tools_state.device_status(device_id)
    except KeyError as exc:
        return f"error: {exc}"
    return str(status)


@mcp.tool()
def device_control(device_id: str, action: str) -> str:
    """Pause or resume a mock AutoDose unit. action must be "pause" or "resume"."""
    try:
        status = tools_state.device_control(device_id, action)
    except (KeyError, ValueError) as exc:
        return f"error: {exc}"
    return str(status)


@mcp.tool()
def list_devices() -> str:
    """List all known mock AutoDose units and their current status."""
    return str(tools_state.list_devices())


@mcp.tool()
async def kb_search(query: str, top_k: int = 3) -> str:
    """Explicitly search the AutoDose support knowledge base for a topic.

    The assistant also gets relevant knowledge-base passages injected
    automatically for every turn (see agent/core.py) - this tool exists so the
    model can *choose* to search again with a more specific/reformulated query
    when the automatically-injected context isn't enough, which is a more
    honest demonstration of deliberate tool use than relying on auto-injection
    alone.
    """
    settings = get_settings()
    client = make_client(settings)
    query_embedding = await embed_query(client, settings, query)
    async with session_scope(settings) as session:
        results = await store.search_knowledge(session, query, query_embedding, top_k)
    if not results:
        return "no matching knowledge base entries found"
    return "\n\n".join(
        f"[{r.title}] (similarity={r.similarity:.2f})\n{r.content}" for r in results
    )


if __name__ == "__main__":
    mcp.run()
