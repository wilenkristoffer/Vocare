"""conversation_exchanges: pair question+answer in one row, drop conversation_turns

Revision ID: 0002_conversation_exchanges
Revises: 0001_init
Create Date: 2026-08-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0002_conversation_exchanges"
down_revision = "0001_init"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.drop_index("ix_conversation_turns_embedding", table_name="conversation_turns")
    op.drop_table("conversation_turns")

    op.create_table(
        "conversation_exchanges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversation_sessions.id"),
            nullable=False,
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("question_embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("answer_embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index(
        "ix_conversation_exchanges_question_embedding",
        "conversation_exchanges",
        ["question_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"question_embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_conversation_exchanges_answer_embedding",
        "conversation_exchanges",
        ["answer_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"answer_embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_exchanges_answer_embedding", table_name="conversation_exchanges")
    op.drop_index(
        "ix_conversation_exchanges_question_embedding", table_name="conversation_exchanges"
    )
    op.drop_table("conversation_exchanges")

    op.create_table(
        "conversation_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversation_sessions.id"),
            nullable=False,
        ),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_conversation_turns_embedding",
        "conversation_turns",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
