from __future__ import annotations

import uuid

from google import genai
from google.genai import types

from vocare.agent.mcp_client import MCPToolClient
from vocare.agent.prompts import SYSTEM_PROMPT
from vocare.config import Settings
from vocare.gemini_client import embed_query
from vocare.logging_config import get_logger
from vocare.rag import store
from vocare.rag.db import session_scope
from vocare.rag.models import RetrievedChunk

logger = get_logger(__name__)

MAX_TOOL_HOPS = 4
NO_CONTEXT_NOTE = (
    "(No knowledge-base passage cleared the similarity threshold for this question - "
    "say so plainly if you don't otherwise know the answer, per the escalation policy.)"
)


def _format_context(kb_hits: list[RetrievedChunk], min_similarity: float) -> str:
    good_hits = [h for h in kb_hits if h.similarity >= min_similarity]
    if not good_hits:
        return NO_CONTEXT_NOTE
    lines = ["Relevant knowledge-base passages:"]
    for hit in good_hits:
        lines.append(f"- [{hit.title}] (similarity={hit.similarity:.2f}): {hit.content}")
    return "\n".join(lines)


def _format_history(history_hits: list[RetrievedChunk], min_similarity: float) -> str:
    good_hits = [h for h in history_hits if h.similarity >= min_similarity]
    if not good_hits:
        return ""
    lines = ["Relevant snippets from earlier conversations (different session):"]
    for hit in good_hits:
        lines.append(f"- {hit.content}")
    return "\n".join(lines)


class Agent:
    """Owns one chat session: RAG context injection + Gemini tool-call loop.

    Used by both text mode and voice mode (voice_session.py) for the actual
    "decide what to say / which tool to call" logic - voice mode only differs
    in how audio gets turned into a user_text turn and how the reply gets
    spoken, not in the reasoning loop itself.
    """

    def __init__(
        self,
        client: genai.Client,
        settings: Settings,
        mcp_client: MCPToolClient | None,
        conversation_session_id: uuid.UUID,
        mode: str,
    ) -> None:
        self.client = client
        self.settings = settings
        self.mcp_client = mcp_client
        self.conversation_session_id = conversation_session_id
        self.mode = mode
        self.chat = client.aio.chats.create(
            model=settings.vocare_text_model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=(
                    [types.Tool(function_declarations=mcp_client.function_declarations())]
                    if mcp_client is not None
                    else None
                ),
            ),
        )

    async def _execute_tool(self, name: str, args: dict) -> str:
        if self.mcp_client is None:
            # Shouldn't happen - no tools were declared to the model when
            # mcp_client is None - but surface it plainly rather than crash
            # if the model somehow still emits a function call.
            return (
                f"error calling tool {name}: tool calling is disabled (VOCARE_ENABLE_TOOLS=false)"
            )
        try:
            return await self.mcp_client.call_tool(name, args)
        except Exception as exc:  # noqa: BLE001 - tool failures must become model-visible text
            logger.warning("tool_call_failed", tool=name, args=args, error=str(exc))
            return f"error calling tool {name}: {exc}"

    async def respond(self, user_text: str) -> str:
        query_embedding = await embed_query(self.client, self.settings, user_text)

        async with session_scope(self.settings) as session:
            kb_hits = await store.search_knowledge(
                session, user_text, query_embedding, self.settings.rag_top_k
            )
            history_hits = await store.search_conversation_history(
                session,
                query_embedding,
                self.settings.rag_top_k,
                exclude_session_id=self.conversation_session_id,
            )

        context_block = _format_context(kb_hits, self.settings.rag_min_similarity)
        history_block = _format_history(history_hits, self.settings.rag_min_similarity)
        augmented = f"{context_block}\n\n{history_block}\n\nUser question: {user_text}".strip()

        response = await self.chat.send_message(augmented)
        for _ in range(MAX_TOOL_HOPS):
            calls = response.function_calls
            if not calls:
                break
            response_parts = []
            for call in calls:
                tool_name = call.name or "unknown_tool"
                logger.info("tool_call", tool=tool_name, args=dict(call.args or {}))
                result = await self._execute_tool(tool_name, dict(call.args or {}))
                response_parts.append(
                    types.Part.from_function_response(name=tool_name, response={"result": result})
                )
            response = await self.chat.send_message(response_parts)

        final_text = response.text or "(no response text)"
        assistant_embedding = await embed_query(self.client, self.settings, final_text)
        if await self._is_duplicate_question(query_embedding):
            logger.debug("skipped_duplicate_exchange", question=user_text)
        else:
            await self._persist_exchange(user_text, final_text, query_embedding, assistant_embedding)
        return final_text

    async def _is_duplicate_question(self, question_embedding: list[float]) -> bool:
        async with session_scope(self.settings) as session:
            return await store.find_similar_past_question(
                session, question_embedding, self.settings.rag_dedup_similarity
            )

    async def _persist_exchange(
        self,
        question: str,
        answer: str,
        question_embedding: list[float],
        answer_embedding: list[float],
    ) -> None:
        async with session_scope(self.settings) as session:
            await store.add_conversation_exchange(
                session,
                session_id=self.conversation_session_id,
                question=question,
                answer=answer,
                question_embedding=question_embedding,
                answer_embedding=answer_embedding,
            )
            await session.commit()


async def start_session(
    client: genai.Client, settings: Settings, mcp_client: MCPToolClient | None, mode: str
) -> Agent:
    async with session_scope(settings) as session:
        session_id = await store.create_conversation_session(session, mode)
        await session.commit()
    return Agent(client, settings, mcp_client, session_id, mode)
