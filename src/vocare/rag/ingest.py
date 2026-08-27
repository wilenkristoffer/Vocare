from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vocare.config import KNOWLEDGE_BASE_DIR, Settings, get_settings
from vocare.gemini_client import embed_texts, make_client
from vocare.logging_config import get_logger
from vocare.rag import store
from vocare.rag.db import session_scope

logger = get_logger(__name__)

_H1 = re.compile(r"^#\s+(.*)$")
_H2 = re.compile(r"^##\s+(.*)$")


@dataclass
class Chunk:
    title: str
    content: str


def chunk_markdown(text: str) -> list[Chunk]:
    """Split a KB doc into retrievable chunks.

    Pure/synchronous and embedding-free on purpose, so chunking logic is unit
    testable without a Gemini API key or a database.

    Rule: split on H2 (##) sections; each section's chunk title is
    "<H1 title> - <H2 title>". A doc with no H2 headings becomes a single chunk
    titled after its H1 (or "untitled" if it has none).
    """
    lines = text.splitlines()
    doc_title = "untitled"
    for line in lines:
        match = _H1.match(line.strip())
        if match:
            doc_title = match.group(1).strip()
            break

    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in lines:
        h2 = _H2.match(line.strip())
        if h2:
            if current_title is not None or current_lines:
                sections.append((current_title or doc_title, current_lines))
            current_title = h2.group(1).strip()
            current_lines = []
        elif _H1.match(line.strip()):
            continue
        else:
            current_lines.append(line)
    sections.append((current_title or doc_title, current_lines))

    chunks: list[Chunk] = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        full_title = doc_title if title == doc_title else f"{doc_title} - {title}"
        chunks.append(Chunk(title=full_title, content=body))
    return chunks


def load_chunks_from_dir(directory: Path) -> list[tuple[str, Chunk]]:
    """Returns (source_filename, chunk) pairs for every .md file in directory."""
    results: list[tuple[str, Chunk]] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_markdown(text):
            results.append((path.name, chunk))
    return results


async def run_ingest(settings: Settings | None = None, directory: Path | None = None) -> int:
    """Wipes and rebuilds the knowledge_chunks table from knowledge_base/*.md.

    Simple full-rebuild strategy is fine at this scale (a few dozen chunks) -
    no incremental-update logic needed.
    """
    settings = settings or get_settings()
    directory = directory or KNOWLEDGE_BASE_DIR
    pairs = load_chunks_from_dir(directory)
    if not pairs:
        logger.warning("no_markdown_files_found", directory=str(directory))
        return 0

    client = make_client(settings)
    texts = [chunk.content for _, chunk in pairs]
    embeddings = await embed_texts(client, settings, texts, task_type="RETRIEVAL_DOCUMENT")

    async with session_scope(settings) as session:
        await store.clear_knowledge_base(session)
        for (source, chunk), embedding in zip(pairs, embeddings, strict=True):
            await store.add_knowledge_chunk(
                session,
                source=source,
                title=chunk.title,
                content=chunk.content,
                embedding=embedding,
            )
        await session.commit()

    logger.info("ingested_chunks", count=len(pairs))
    return len(pairs)
