from __future__ import annotations

import numpy as np
from google import genai
from google.genai import types

from vocare.config import Settings


def make_client(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.require_api_key())


async def embed_texts(
    client: genai.Client,
    settings: Settings,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed a batch of texts with the Gemini embedding model, L2-normalized.

    Normalization is done manually: gemini-embedding-001 only returns unit-length
    vectors at its native 3072 dims, so at a smaller output_dimensionality (we use
    768, to keep the vector column small) we normalize ourselves before storing.
    pgvector's cosine-distance operator doesn't strictly require this, but it keeps
    "similarity score" interpretable as a true cosine similarity in [-1, 1].

    Async (uses client.aio) so it doesn't block the event loop alongside the
    async DB/MCP calls elsewhere in the agent loop.
    """
    if not texts:
        return []
    response = await client.aio.models.embed_content(
        model=settings.vocare_embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.vocare_embedding_dim,
        ),
    )
    vectors: list[list[float]] = []
    for embedding in response.embeddings or []:
        values = np.array(embedding.values, dtype=np.float32)
        norm = np.linalg.norm(values)
        if norm > 0:
            values = values / norm
        vectors.append(values.tolist())
    return vectors


async def embed_query(client: genai.Client, settings: Settings, text: str) -> list[float]:
    vectors = await embed_texts(client, settings, [text], task_type="RETRIEVAL_QUERY")
    return vectors[0] if vectors else []
