from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class KnowledgeChunk(Base):
    """One retrievable unit from the local knowledge base (knowledge_base/*.md)."""

    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationSession(Base):
    """One run of the assistant (one CLI invocation of `vocare chat` / `vocare voice`)."""

    __tablename__ = "conversation_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mode: Mapped[str] = mapped_column(Text, nullable=False)  # "text" | "voice"
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    exchanges: Mapped[list[ConversationExchange]] = relationship(back_populates="session")


class ConversationExchange(Base):
    """One question+answer pair from a session, each side embedded so past
    conversations are retrievable either by a new question resembling an old
    question, or by a new question resembling an old answer's wording."""

    __tablename__ = "conversation_exchanges"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversation_sessions.id"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    answer_embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ConversationSession] = relationship(back_populates="exchanges")


class RetrievedChunk:
    """Plain result object returned by store search functions (not an ORM model)."""

    def __init__(self, title: str, content: str, source: str, similarity: float) -> None:
        self.title = title
        self.content = content
        self.source = source
        self.similarity = similarity
