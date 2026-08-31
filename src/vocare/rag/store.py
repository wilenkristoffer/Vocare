from __future__ import annotations

import re
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vocare.rag.models import (
    ConversationExchange,
    ConversationSession,
    KnowledgeChunk,
    RetrievedChunk,
)

# Similarity score assigned to a full-text-only match (see search_knowledge) -
# high enough to clear the agent's rag_min_similarity fallback threshold.
LEXICAL_MATCH_SIMILARITY = 0.9

_WORD_RE = re.compile(r"\w+")


def _identifier_terms(query_text: str) -> str | None:
    """Extract alphanumeric-identifier-looking tokens (e.g. "E02") from the
    query and OR them together for to_tsquery, e.g.
    "What does E02 mean?" -> "E02".

    Deliberately narrow (only tokens containing a digit), not "OR every word
    in the query": an early version OR'd all words, which matched so many
    chunks on ordinary words like "error" or "code" that they drowned out
    genuinely well-ranked vector hits, all tied at the same flat lexical-match
    score. Specific codes/identifiers are exactly the case dense embeddings
    get wrong (see search_knowledge's docstring) and rare enough in a query
    that OR-matching just those is safe. Extracting tokens ourselves (rather
    than passing raw query text into to_tsquery) also sidesteps tsquery
    operator-syntax injection from user input.
    """
    words = _WORD_RE.findall(query_text)
    identifier_like = [w for w in words if any(ch.isdigit() for ch in w)]
    return " | ".join(identifier_like) if identifier_like else None


async def add_knowledge_chunk(
    session: AsyncSession, *, source: str, title: str, content: str, embedding: list[float]
) -> None:
    session.add(KnowledgeChunk(source=source, title=title, content=content, embedding=embedding))


async def clear_knowledge_base(session: AsyncSession) -> None:
    await session.execute(delete(KnowledgeChunk))


async def search_knowledge(
    session: AsyncSession, query_text: str, query_embedding: list[float], top_k: int
) -> list[RetrievedChunk]:
    """Hybrid search: vector similarity plus a plain full-text match.

    Dense embeddings are weak at telling apart short alphanumeric identifiers
    (e.g. "E02" vs "E03") when everything around them ("error code", "AutoDose
    unit", troubleshooting language) is nearly identical - in practice the
    right chunk for "what does E02 mean?" can rank outside the top-10 purely
    on cosine similarity. A full-text match on the literal query terms is a
    strong relevance signal that catches exactly this case, so results from
    both are merged rather than relying on vector similarity alone.
    """
    distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
    vector_stmt = (
        select(KnowledgeChunk, distance.label("distance")).order_by(distance).limit(top_k)
    )
    vector_result = await session.execute(vector_stmt)
    hits: dict[uuid.UUID, RetrievedChunk] = {
        chunk.id: RetrievedChunk(
            title=chunk.title,
            content=chunk.content,
            source=chunk.source,
            similarity=1.0 - float(dist),
        )
        for chunk, dist in vector_result.all()
    }

    or_terms = _identifier_terms(query_text)
    if or_terms is not None:
        tsquery = func.to_tsquery("english", or_terms)
        tsvector = func.to_tsvector("english", KnowledgeChunk.title + " " + KnowledgeChunk.content)
        lexical_stmt = (
            select(KnowledgeChunk)
            .where(tsvector.op("@@")(tsquery))
            .order_by(func.ts_rank(tsvector, tsquery).desc())
            .limit(top_k)
        )
        lexical_result = await session.execute(lexical_stmt)
        for chunk in lexical_result.scalars().all():
            existing = hits.get(chunk.id)
            similarity = max(existing.similarity if existing else 0.0, LEXICAL_MATCH_SIMILARITY)
            hits[chunk.id] = RetrievedChunk(
                title=chunk.title, content=chunk.content, source=chunk.source, similarity=similarity
            )

    ranked = sorted(hits.values(), key=lambda hit: hit.similarity, reverse=True)
    return ranked[:top_k]


async def create_conversation_session(session: AsyncSession, mode: str) -> uuid.UUID:
    record = ConversationSession(mode=mode)
    session.add(record)
    await session.flush()
    return record.id


async def add_conversation_exchange(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    question: str,
    answer: str,
    question_embedding: list[float],
    answer_embedding: list[float],
) -> None:
    session.add(
        ConversationExchange(
            session_id=session_id,
            question=question,
            answer=answer,
            question_embedding=question_embedding,
            answer_embedding=answer_embedding,
        )
    )


def _exchange_hit(exchange: ConversationExchange, similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        title="past conversation",
        content=f"Q: {exchange.question}\nA: {exchange.answer}",
        source=str(exchange.session_id),
        similarity=similarity,
    )


async def search_conversation_history(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
    exclude_session_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant question+answer pairs from *past* sessions - this is the
    'RAG over previous conversations' piece: continuity across separate runs of the
    app, not just within the current session's own message history (which the model
    already sees in full via normal chat context).

    Searches both sides of each pair - the query can match an old question's
    phrasing or an old answer's phrasing - and always returns the full pair as one
    hit, keyed by exchange id, so the model never sees an answer without the
    question it belongs to (or vice versa).
    """
    hits: dict[uuid.UUID, RetrievedChunk] = {}
    for column in (ConversationExchange.question_embedding, ConversationExchange.answer_embedding):
        distance = column.cosine_distance(query_embedding)
        stmt = select(ConversationExchange, distance.label("distance"))
        if exclude_session_id is not None:
            stmt = stmt.where(ConversationExchange.session_id != exclude_session_id)
        stmt = stmt.order_by(distance).limit(top_k)
        result = await session.execute(stmt)
        for exchange, dist in result.all():
            similarity = 1.0 - float(dist)
            existing = hits.get(exchange.id)
            if existing is None or similarity > existing.similarity:
                hits[exchange.id] = _exchange_hit(exchange, similarity)

    ranked = sorted(hits.values(), key=lambda hit: hit.similarity, reverse=True)
    return ranked[:top_k]


async def find_similar_past_question(
    session: AsyncSession, question_embedding: list[float], min_similarity: float
) -> bool:
    """True if an existing exchange's question is a near-duplicate of this one.

    Checked across *all* sessions, not just past ones - a repeat within the same
    long-running session is just as much a duplicate as one from an old session.
    Used to avoid filling conversation_exchanges with endless near-identical rows
    (e.g. the same FAQ-style question asked many times).
    """
    distance = ConversationExchange.question_embedding.cosine_distance(question_embedding)
    stmt = select(distance.label("distance")).order_by(distance).limit(1)
    result = await session.execute(stmt)
    closest = result.scalar_one_or_none()
    return closest is not None and (1.0 - float(closest)) >= min_similarity
