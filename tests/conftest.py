from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Integration tests need a real Postgres+pgvector (docker compose up db) and
    are skipped unless explicitly opted into, so `pytest` works out of the box
    with no infrastructure for the pure-logic test suite."""
    if os.environ.get("RUN_INTEGRATION"):
        return
    skip_integration = pytest.mark.skip(
        reason="set RUN_INTEGRATION=1 to run (requires `docker compose up db`)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
