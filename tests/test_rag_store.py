from __future__ import annotations

import os

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vocare.rag import store
from vocare.rag.models import Base

pytestmark = pytest.mark.integration

# Deliberately a DIFFERENT database than the app's default ("vocare") - this
# fixture create_all()s and drop_all()s the schema, which must never run
# against the same database a real `vocare ingest`/`vocare chat` run uses.
# (Earlier version of this fixture used the app's own DB and its drop_all()
# wiped out a real ingested knowledge base - see the incident this comment
# is protecting against.)
_TEST_DB_NAME = "vocare_test"
_APP_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://vocare:vocare@localhost:5432/vocare"
)
_BASE_URL, _, _APP_DB_NAME = _APP_DATABASE_URL.rpartition("/")
DATABASE_URL = f"{_BASE_URL}/{_TEST_DB_NAME}"

assert _APP_DB_NAME != _TEST_DB_NAME, "test database must not be the app database"


def _unit_vector(dim: int, hot_index: int) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    vec[hot_index] = 1.0
    return vec.tolist()


async def _ensure_test_database_exists() -> None:
    admin_engine = create_async_engine(_APP_DATABASE_URL, isolation_level="AUTOCOMMIT")
    async with admin_engine.connect() as conn:
        # _TEST_DB_NAME is a hardcoded module constant, not user input - safe to inline.
        exists = (
            await conn.exec_driver_sql(
                f"SELECT 1 FROM pg_database WHERE datname = '{_TEST_DB_NAME}'"
            )
        ).scalar()
        if not exists:
            await conn.exec_driver_sql(f'CREATE DATABASE "{_TEST_DB_NAME}"')
    await admin_engine.dispose()


@pytest.fixture
async def db_session():
    await _ensure_test_database_exists()
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_search_knowledge_orders_by_similarity(db_session) -> None:
    await store.add_knowledge_chunk(
        db_session, source="a.md", title="Close match", content="c1", embedding=_unit_vector(768, 0)
    )
    await store.add_knowledge_chunk(
        db_session, source="b.md", title="Far match", content="c2", embedding=_unit_vector(768, 1)
    )
    await db_session.commit()

    results = await store.search_knowledge(
        db_session, "zzz_no_lexical_match_zzz", _unit_vector(768, 0), top_k=2
    )

    assert len(results) == 2
    assert results[0].title == "Close match"
    assert results[0].similarity > results[1].similarity


async def test_search_knowledge_lexical_fallback_finds_poorly_ranked_exact_term(db_session) -> None:
    """Regression test for the real failure this hybrid search fixes: a query
    for a specific error code where the *correct* chunk is a poor vector match
    (embeddings struggle to tell "E02" apart from other, similarly-worded
    error-code entries) but contains the literal term "E02" nowhere else."""
    await store.add_knowledge_chunk(
        db_session,
        source="codes.md",
        title="E02",
        content="E02 means the dispense count mismatched.",
        embedding=_unit_vector(768, 500),  # deliberately far from the query embedding
    )
    for i in range(5):
        await store.add_knowledge_chunk(
            db_session,
            source="other.md",
            title=f"Unrelated topic {i}",
            content=f"Some unrelated troubleshooting content number {i}.",
            embedding=_unit_vector(768, i),  # close to the query embedding
        )
    await db_session.commit()

    results = await store.search_knowledge(
        db_session, "What does E02 mean?", _unit_vector(768, 0), top_k=3
    )

    assert any(r.title == "E02" for r in results)


async def test_conversation_history_excludes_current_session(db_session) -> None:
    session_a = await store.create_conversation_session(db_session, mode="text")
    session_b = await store.create_conversation_session(db_session, mode="text")
    await store.add_conversation_turn(
        db_session,
        session_id=session_a,
        role="user",
        content="from session A",
        embedding=_unit_vector(768, 0),
    )
    await store.add_conversation_turn(
        db_session,
        session_id=session_b,
        role="user",
        content="from session B",
        embedding=_unit_vector(768, 0),
    )
    await db_session.commit()

    results = await store.search_conversation_history(
        db_session, _unit_vector(768, 0), top_k=5, exclude_session_id=session_a
    )

    contents = [r.content for r in results]
    assert "from session B" in contents
    assert "from session A" not in contents
