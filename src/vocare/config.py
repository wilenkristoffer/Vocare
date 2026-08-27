from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PACKAGE_ROOT / "knowledge_base"


class Settings(BaseSettings):
    """App configuration, loaded from environment / .env.

    Model IDs are configurable because Gemini model names change often -
    verify current IDs at https://ai.google.dev/gemini-api/docs/models
    if a model call starts failing with a "not found" error.
    """

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    gemini_api_key: str = ""

    database_url: str = "postgresql+asyncpg://vocare:vocare@localhost:5432/vocare"

    vocare_text_model: str = "gemini-2.5-flash"
    vocare_live_model: str = "gemini-2.5-flash-native-audio-latest"
    vocare_embedding_model: str = "gemini-embedding-001"
    vocare_embedding_dim: int = 768

    vocare_log_level: str = "INFO"
    vocare_log_file: str = "vocare.log"

    vocare_voice_reply: bool = False
    """In `vocare voice`, whether the reply is spoken back (Live API TTS) or just
    printed as text. Off by default: it's the faster iteration loop (no extra
    Live API round trip) and the text reply is always printed either way -
    this only controls whether it's *also* spoken."""

    vocare_enable_tools: bool = True
    """Whether the agent spawns the local MCP tool server and can call tools
    (calculator, clock, kb_search, mock device control). Core chat/voice with
    RAG works with this off - tool-calling/MCP is a demonstrable capability
    layered on top, not something the app depends on to function."""

    rag_top_k: int = 5
    rag_min_similarity: float = 0.55
    """Below this cosine-similarity score, retrieval is treated as 'no good match'
    for the confidence/fallback logic - see agent/core.py."""

    def require_api_key(self) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in "
                "(get a key at https://aistudio.google.com/apikey)."
            )
        return self.gemini_api_key


def get_settings() -> Settings:
    return Settings()
