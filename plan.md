# Vocare - Standalone Voice AI Assistant

## Concept

A standalone, local-first assistant: launch one Python process and talk to it through your
mic, or just type - it answers grounded in a small knowledge base via RAG, with a couple of
local tools it can call (via MCP) when a question needs more than retrieval. Replies default
to text; speaking the reply back (via the Live API) is an opt-in setting, not the default.
No browser, no extra services required to run it beyond a local Postgres container (Gemini API
is the one external dependency).

Text mode is the simpler, always-available path (same agent core, no audio I/O) - good for
scripted testing and for trying it out without a mic. Voice mode adds speech input and,
optionally, a spoken reply on top of the same agent.

**Name**: *Vocare* (Latin: "to call").

## Why this shape

- **Standalone** = one application to run: a Python CLI (`vocare` / `python -m vocare`) plus a
  local Postgres instance (via `docker compose up`). No browser, no separate frontend service,
  no external backend to stand up. "Standalone" here means "one thing to `git clone` and run,"
  not "zero dependencies" - a local Postgres container is a fine dependency to have, and
  `pgvector` is a genuinely good fit for the RAG store rather than something bolted on to check
  a box.
- **Voice is a core input mode, text is the default output**: the Gemini Live API does the
  speech side of things, but the assistant doesn't assume you want to be talked at - a reply is
  text on screen unless you explicitly turn on spoken replies. Voice input and spoken output are
  independent choices.
- **Tool-calling / MCP is an optional layer, not a dependency**: small, local tools (a
  calculator, a clock/timezone lookup, an explicit knowledge-base search, a mock device
  status/control tool) demonstrate the real mechanism - agent -> MCP tool discovery -> Gemini
  function-calling -> tool execution -> result back to the model - without pretending there's a
  real backend system behind them. `VOCARE_ENABLE_TOOLS=false` turns it off entirely: no MCP
  subprocess gets spawned, and the agent just answers from RAG. Plain knowledge-base chat/voice
  never depends on tool-calling working.

## Why this is feasible with Gemini (research notes)

- **Voice**: [Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api) is a stateful
  WebSocket session with native bidirectional audio - speech in, speech out, barge-in
  (interrupt), input/output transcription, and function calling *during* the live session.
  - Audio format is fixed and simple: input is raw 16-bit PCM, 16kHz, little-endian, mono
    (sent as `audio/pcm;rate=16000` blobs via `session.send_realtime_input`); output is raw
    16-bit PCM, 24kHz. The SDK will resample input if you send a different rate, but 16kHz/16-bit/mono
    capture is the target. Standard local pattern: `sounddevice` (or `pyaudio`) for mic capture
    and playback, `CHUNK_SIZE` around 512-1024 samples. This maps cleanly onto a terminal app -
    no browser Web Audio API needed.
  - Default session length is ~10 minutes (session resumption exists for longer conversations;
    not needed for an MVP).
  - Sources: [Live API capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities),
    [Get started with Live API via WebSockets](https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket),
    [gemini-live-api-examples](https://github.com/google-gemini/gemini-live-api-examples).
- **Text + tool calling**: the [`google-genai` Python SDK](https://github.com/googleapis/python-genai)
  supports structured `FunctionDeclaration`s; the model returns a function-call part, we execute
  it, and feed the result back. Same loop for text mode and inside the Live session.
- **RAG**: no managed "RAG service" needed at this scale. Gemini embeddings + Postgres with the
  `pgvector` extension for storage/similarity search over a few dozen short knowledge-base docs
  (and later, conversation history) is a well-trodden, production-realistic setup without being
  overkill for the data size.
- **MCP, run locally**: MCP's **stdio transport** is exactly the "local tool server" pattern -
  the client (our app) spawns the MCP server as a child process and talks JSON-RPC over
  stdin/stdout. No network service, no separate container - fits "standalone" perfectly. Use the
  official [`mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk) for the tool
  server, and a small client-side adapter that discovers its tools and turns their schemas into
  Gemini `FunctionDeclaration`s.
- **Cost**: Gemini Flash-tier free tier (~1,500 requests/day) is enough to build and try this out
  without a paid key. [Free tier limits reference](https://yingtu.ai/en/blog/gemini-api-free-tier).

## Architecture

```
   Terminal / CLI
        │  mic audio (sounddevice) + text input
        ▼
┌───────────────────────────┐
│        Vocare app          │
│  ┌───────────────────────┐ │
│  │   Agent core           │ │◄──── Gemini Live API (voice) / Gemini API (text)
│  │  (tool-call loop)      │ │      via google-genai SDK
│  └──┬──────────────┬─────┘ │
│     │              │        │
│  RAG retrieval   MCP client │──spawns (stdio)──► Local MCP tool server
│     │                       │                    (calculator, clock,
│     ▼                       │                     kb-lookup, mock device tool)
│  ┌────────────┐             │
│  │  Postgres  │◄────────────┘  (KB embeddings, conversation history)
│  │ + pgvector │  (Dockerized)
│  └────────────┘
└───────────────────────────┘
```

Single process, single repo, one Postgres container. No web frontend, no network MCP service -
the MCP tool server is a local child process over stdio, not a container.

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Best Gemini SDK support, comfortable ecosystem for this kind of app |
| App shape | CLI entry point (`python -m vocare` / installable console script) | Literal "standalone application" |
| Terminal UI | Plain stdout to start; `rich`/`textual` as a polish pass (status indicators, live transcript) | Cheap way to make the demo look good without a browser |
| LLM | Gemini (`google-genai` SDK) - Live API for voice, same/adjacent model for text mode | One vendor, covers voice + text + function calling |
| Audio I/O | `sounddevice` (mic capture + playback), 16-bit PCM, 16kHz in / 24kHz out | Matches Live API's native format exactly, avoids resampling code |
| Agent orchestration | Hand-rolled tool-call loop (no LangChain/LlamaIndex) | Small enough to own end-to-end; keeps the mechanics visible instead of hidden behind a framework |
| RAG store | PostgreSQL + `pgvector`, run via Docker Compose | Real, production-style RAG backend; one DB for KB embeddings + conversation-history embeddings |
| DB access | SQLAlchemy 2.0 (async) + Alembic migrations | Proper schema management even at this scale |
| Embeddings | Gemini embedding model | Same vendor/key as the rest |
| MCP | Official `mcp` Python SDK, stdio transport, spawned as a child process by the app | Real MCP discovery/tool-calling without needing a network service |
| Testing | pytest + pytest-asyncio; Gemini calls mocked in unit tests; integration tests against a real Postgres (CI service container) | |
| Lint/type | ruff + mypy | |
| CI/CD | GitHub Actions: lint -> type-check -> test (with a Postgres+pgvector service container) | Voice I/O can't run in CI (no mic), so CI covers text-mode + RAG + MCP tool logic |
| Packaging | `pyproject.toml`, installable via `pip install .`; `docker-compose.yml` provides Postgres | |
| Monitoring | Structured logs (`structlog`) to a local file | Right-sized for a local CLI app rather than force-fitting server ops tooling |

## Small local knowledge base + tools

- KB: a couple dozen short markdown docs for the "Meridian AutoDose" pharmacy-robot support
  domain (equipment troubleshooting, error codes, maintenance, FAQs) - a believable, structured
  domain that gives RAG and retrieval-confidence something real to work with.
- Tools (via MCP):
  1. `kb_search` - explicit knowledge-base search (separate from the automatic RAG context
     injection, so the model can *choose* to search again with a different query when the
     automatically-injected context isn't enough)
  2. small deterministic tools (`calculate`, `get_current_time`) that make tool-use
     successes/failures obvious in a demo
  3. a mock device status/control tool (fits the robotics theme) that reads/writes an
     in-memory state, without pretending to integrate with real hardware

## Escalation logic

The agent has a clear, narrated fallback when it doesn't know something or a tool fails - e.g.
"I don't have information on that in my knowledge base" rather than confabulating. Confidence
thresholds on retrieval and explicit tool-failure handling back this up; anything safety-related
or outside the equipment-support scope gets flagged rather than guessed at.

## Phased roadmap

0. **Scaffold**: `pyproject.toml`, package layout, `docker-compose.yml` (Postgres + pgvector), CI skeleton, `.env.example` for the Gemini API key + DB URL
1. **Text mode MVP**: CLI text chat loop, Gemini text calls, basic conversation loop (no persistence yet)
2. **RAG v1**: write the synthetic KB, ingestion script, Postgres/pgvector schema + migrations, retrieval + citations, wire into text mode
3. **MCP tools**: local MCP server (stdio) with the tools above, client-side adapter, wired into the agent's tool set
4. **Voice mode**: Gemini Live API session, `sounddevice` mic capture/playback, push-to-talk, same agent/RAG/tools as text mode, spoken reply as an opt-in setting
5. **Conversation memory RAG**: persist past sessions in Postgres, embed + retrieve relevant history across runs
6. **Escalation/fallback logic**: confidence thresholds, tool-failure handling, "I don't know" behavior, tested with edge-case prompts
7. **Polish**: `rich`/`textual` terminal UI, better logging, packaging as an installable CLI
8. **Testing + CI**: pytest suite (text mode, RAG, MCP tools mocked/real), GitHub Actions pipeline
9. **Docs**: README with architecture diagram, install/run instructions, LICENSE

## Open decisions to revisit

- Exact Gemini model IDs to pin (check current Flash/Live model names at implementation time - they rev often)
- Push-to-talk vs. voice-activity-detection for turn-taking in voice mode (push-to-talk is simpler and more reliable for v1; VAD is a nice stretch)
- Whether to add a minimal optional GUI later (e.g. a system-tray or simple desktop window) - not needed for the core story, terminal is enough

## Stretch goals (only after core phases are solid)

- Collapse voice mode into one continuous Live session with in-session tool calling, for lower latency
- Containerize the app itself too (not just Postgres), so the whole thing is `docker compose up` with zero local Python setup
- Voice-activity-detection turn-taking instead of push-to-talk
- Multi-language voice support (Live API supports 24 languages)
- A lightweight "LLM-as-judge" eval script over a fixed set of test conversations, tracked in CI
