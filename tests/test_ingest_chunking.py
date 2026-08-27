from __future__ import annotations

from vocare.rag.ingest import chunk_markdown

SAMPLE_DOC = """# Widget Overview

## What it is

A widget is a small useful thing.

## How it works

It works by widgeting.

## Troubleshooting

If it doesn't widget, check the batteries.
"""

SAMPLE_DOC_NO_SECTIONS = """# Simple Doc

Just one paragraph, no sub-sections at all.
"""


def test_chunk_markdown_splits_on_h2() -> None:
    chunks = chunk_markdown(SAMPLE_DOC)
    assert len(chunks) == 3
    titles = [c.title for c in chunks]
    assert titles == [
        "Widget Overview - What it is",
        "Widget Overview - How it works",
        "Widget Overview - Troubleshooting",
    ]


def test_chunk_markdown_content_excludes_headings() -> None:
    chunks = chunk_markdown(SAMPLE_DOC)
    assert "widgeting" in chunks[1].content
    assert "##" not in chunks[1].content
    assert "#" not in chunks[1].content


def test_chunk_markdown_single_chunk_when_no_h2() -> None:
    chunks = chunk_markdown(SAMPLE_DOC_NO_SECTIONS)
    assert len(chunks) == 1
    assert chunks[0].title == "Simple Doc"
    assert "one paragraph" in chunks[0].content


def test_chunk_markdown_empty_sections_are_dropped() -> None:
    doc = "# Title\n\n## Empty section\n\n## Real section\n\nsome content\n"
    chunks = chunk_markdown(doc)
    assert len(chunks) == 1
    assert chunks[0].title == "Title - Real section"
