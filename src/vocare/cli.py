from __future__ import annotations

import argparse
import asyncio

from vocare.agent.core import start_session
from vocare.agent.mcp_client import optional_mcp_client
from vocare.config import get_settings
from vocare.gemini_client import make_client
from vocare.logging_config import configure_logging, get_logger
from vocare.rag.ingest import run_ingest

logger = get_logger(__name__)


async def _run_chat() -> None:
    settings = get_settings()
    client = make_client(settings)
    async with optional_mcp_client(settings) as mcp_client:
        agent = await start_session(client, settings, mcp_client, mode="text")
        tools_note = "" if settings.vocare_enable_tools else " (tools disabled)"
        print(f"Vocare text mode{tools_note}. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_text = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break
            reply = await agent.respond(user_text)
            print(f"assistant: {reply}\n")


async def _run_voice() -> None:
    from vocare.voice_session import run_voice_turn  # audio deps only needed here

    settings = get_settings()
    client = make_client(settings)
    async with optional_mcp_client(settings) as mcp_client:
        agent = await start_session(client, settings, mcp_client, mode="voice")
        reply_mode = "spoken + text" if settings.vocare_voice_reply else "text only"
        tools_note = "" if settings.vocare_enable_tools else ", tools disabled"
        print(
            f"Vocare voice mode (replies: {reply_mode}{tools_note} - see .env to change). "
            "Press Enter to talk, press Enter again to stop.\n"
            "Type 'q' and press Enter (instead of recording) to quit.\n"
        )
        while True:
            try:
                keep_going = await run_voice_turn(client, settings, agent)
            except KeyboardInterrupt:
                print()
                break
            if not keep_going:
                break


async def _run_ingest() -> None:
    settings = get_settings()
    count = await run_ingest(settings)
    print(f"Ingested {count} knowledge-base chunks into Postgres.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vocare", description="Standalone voice AI assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat", help="Text chat mode")
    subparsers.add_parser("voice", help="Voice mode (push-to-talk)")
    subparsers.add_parser("ingest", help="(Re)build the knowledge-base index in Postgres")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    try:
        if args.command == "chat":
            asyncio.run(_run_chat())
        elif args.command == "voice":
            asyncio.run(_run_voice())
        elif args.command == "ingest":
            asyncio.run(_run_ingest())
    except RuntimeError as exc:
        logger.error("startup_error", error=str(exc))
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
