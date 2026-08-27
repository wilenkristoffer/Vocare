# Vocare

A standalone assistant you run locally: talk to it through your mic, or just type, and it
answers grounded in a small knowledge base via RAG, with a couple of local tools it can call
through a real MCP server when a question needs more than retrieval. Replies show up as text
by default; spoken replies are an opt-in setting, not the default.

It's set up around a fictional pharmacy-robotics product, the "Meridian AutoDose" tablet
dispenser, mostly because error codes, troubleshooting steps, and maintenance schedules give
RAG and tool-calling something structured and non-trivial to work with (see
`src/vocare/knowledge_base/`). Swap in your own docs and it works the same way for any domain.

See [plan.md](plan.md) for the architecture and design reasoning in more depth.

## Architecture

```
   Terminal (mic / text)
        |
        v
+----------------------------+
|        Vocare app          |
|  +-----------------------+ |
|  |   Agent core          | |<---- Gemini API (text) / Gemini Live API (voice synth)
|  |  (tool-call loop)     | |      via google-genai SDK
|  +--+-----------------+--+ |
|     |                 |    |
|  RAG retrieval    MCP client |--spawns (stdio)--> Local MCP tool server
|     |                      |                      (calculator, clock,
|     v                      |                       kb-lookup, mock device tool)
|  +------------+            |
|  |  Postgres  |<-----------+   (KB embeddings, conversation history)
|  | + pgvector |  (Dockerized)
|  +------------+
+----------------------------+
```

One Python process (the CLI) plus one Postgres container. No web frontend, no separate
backend service, no network MCP service.

## Setup

Prerequisites: Python 3.11+, Docker, a Gemini API key
([aistudio.google.com/apikey](https://aistudio.google.com/apikey) - the free tier is enough to
build and try this out).

```bash
# 1. Start Postgres (with pgvector)
docker compose up -d db

# 2. Install
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# 3. Configure
copy .env.example .env        # then fill in GEMINI_API_KEY

# 4. Create the schema
alembic upgrade head

# 5. Build the knowledge-base index
vocare ingest

# 6. Talk to it
vocare chat     # text mode
vocare voice    # voice mode (push-to-talk: Enter to start/stop speaking)
```

## Project layout

```
src/vocare/
  cli.py              entry point (chat / voice / ingest)
  config.py           Settings (env vars, model IDs)
  gemini_client.py    Gemini API wrapper (text + embeddings)
  voice_session.py    voice mode: STT via generateContent, TTS via Live API
  audio/io.py         mic capture + playback (sounddevice)
  agent/
    core.py           the tool-call loop + RAG context injection + escalation
    mcp_client.py      MCP stdio client -> Gemini FunctionDeclaration adapter
    prompts.py         system prompt / escalation policy
  mcp_server/
    server.py           local MCP tool server (stdio) - calculate, get_current_time,
                         device_status/control, list_devices, kb_search
    tools_state.py       pure tool logic (unit tested independently of MCP plumbing)
  rag/
    models.py / db.py / store.py   Postgres + pgvector schema and queries
    ingest.py                       markdown -> chunks -> embeddings -> Postgres
  knowledge_base/*.md   the small support knowledge base
migrations/              Alembic migrations
tests/                   pytest suite (integration tests need `docker compose up db`)
```

## A few things worth knowing before reading the code

- **Voice mode doesn't run one continuous Gemini Live session with in-session tool calling.**
  It transcribes the recorded utterance via Gemini's standard multimodal `generateContent`
  (which understands audio input directly), runs the *same* `Agent.respond()` text mode uses
  for RAG + tool-calling + the reply, and only uses the Live API for streaming the reply back
  as speech. One agent implementation for both modes, and each Gemini capability used is
  independently simple to reason about. See `voice_session.py` for the full rationale. A
  documented stretch goal is collapsing this into one continuous live session for lower latency.
- **Spoken replies are opt-in**: `vocare voice` always prints the reply as text; set
  `VOCARE_VOICE_REPLY=true` in `.env` to also have it spoken back via the Live API. Default is
  text-only - one fewer network round trip per turn while iterating.
- **RAG is hybrid, not pure vector search**: dense embeddings turn out to be bad at telling
  apart short codes like "E02" vs "E03" when everything around them reads almost identically,
  so knowledge-base search also does a targeted full-text match on identifier-like query terms
  and merges it with the vector results (see `rag/store.py`).
- **RAG confidence threshold**: retrieved knowledge-base/history passages below
  `rag_min_similarity` (default 0.55, see `config.py`) are dropped rather than injected, and the
  model is explicitly told when nothing cleared the bar - this is the "say you don't know
  rather than confabulate" behavior from the escalation policy.
- **MCP tools are genuinely local, and entirely optional**: `agent/mcp_client.py` spawns
  `python -m vocare.mcp_server.server` as a child process and talks MCP over stdio - there's no
  network service, matching the "standalone" goal while still exercising real MCP
  discovery/tool-calling. Set `VOCARE_ENABLE_TOOLS=false` and the app never spawns that
  subprocess at all - it's a demonstrable capability layered on top of RAG chat/voice, not
  something the core app depends on.
- **Model IDs are configurable** (`.env` / `config.py`) because Gemini model names change often.
  If a call fails with a "model not found"-style error, check
  [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for the
  current Flash/Live model IDs and update `.env`.

## Testing

```bash
pytest                                  # pure-logic unit tests, no infra needed
RUN_INTEGRATION=1 pytest -m integration # needs `docker compose up -d db`
ruff check .
mypy src
```

CI (`.github/workflows/ci.yml`) runs all of the above against a Postgres+pgvector service
container. No Gemini API key is needed in CI - tests use synthetic embedding vectors and a
dummy API key for object construction, not live model calls.

## License

MIT - see [LICENSE](LICENSE).
