"""Voice mode: audio in/out around the same Agent used by text mode.

Design choice (see plan.md "open decisions"): rather than running a single
continuous Gemini Live session that does transcription, reasoning, tool
calls, AND speech synthesis all inside one socket, this splits voice mode
into three steps per turn:

  1. Speech-to-text: the recorded utterance is sent as a WAV file to Gemini's
     standard multimodal generateContent endpoint (Gemini natively
     understands audio input - no Live API needed for this direction).
  2. Reasoning + tools + RAG: the transcript is handed to the *same*
     Agent.respond() text-mode uses. One agent implementation, not two.
  3. Text-to-speech: the reply text is sent into a short-lived Gemini Live
     API session configured for audio-only output, and the streamed audio
     chunks are played back as they arrive.

This keeps the interesting agent logic (RAG injection, escalation policy,
MCP tool-calling) identical between text and voice, and avoids depending on
the harder-to-verify-without-a-live-key details of driving tool calls
through an open Live session. It's also cheaper: Gemini's standard
generateContent audio input is $1.00/1M tokens vs. $3.00/1M on the
native-audio Live model, so routing transcription through the standard
endpoint instead of Live saves money on top of being simpler to test. The
trade-off is an extra model round trip per turn instead of one continuous
session - acceptable for a demo, and a documented stretch goal (collapse to
one live session) if lower latency matters later.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from vocare.agent.core import Agent
from vocare.audio.io import (
    SAMPLE_RATE_OUT,
    pcm_to_wav_bytes,
    play_pcm,
    record_until_enter,
)
from vocare.config import Settings
from vocare.logging_config import get_logger

logger = get_logger(__name__)

TRANSCRIBE_PROMPT = (
    "Transcribe the following audio verbatim. Output only the transcript text, "
    "nothing else. If the audio is silent or unintelligible, output nothing."
)
SPEAK_SYSTEM_INSTRUCTION = (
    "Speak the exact text you are given naturally and clearly. Do not add any "
    "commentary, greetings, or extra words of your own."
)


async def transcribe_audio(client: genai.Client, settings: Settings, pcm_bytes: bytes) -> str:
    wav_bytes = pcm_to_wav_bytes(pcm_bytes)
    response = await client.aio.models.generate_content(
        model=settings.vocare_text_model,
        contents=[
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            TRANSCRIBE_PROMPT,
        ],
    )
    return (response.text or "").strip()


async def synthesize_speech(client: genai.Client, settings: Settings, text: str) -> bytes:
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=SPEAK_SYSTEM_INSTRUCTION)]),
    )
    audio_chunks: list[bytes] = []
    async with client.aio.live.connect(model=settings.vocare_live_model, config=config) as session:
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": text}]}, turn_complete=True
        )
        async for response in session.receive():
            if response.data is not None:
                audio_chunks.append(response.data)
            if response.server_content and response.server_content.turn_complete:
                break
    return b"".join(audio_chunks)


async def run_voice_turn(client: genai.Client, settings: Settings, agent: Agent) -> bool:
    """Runs one push-to-talk turn. Returns False when the user types a quit
    command instead of recording (see audio.io.record_until_enter)."""
    raw_audio = record_until_enter()
    if raw_audio is None:
        return False
    if not raw_audio:
        print("(nothing recorded - try again)\n")
        return True

    transcript = await transcribe_audio(client, settings, raw_audio)
    if not transcript:
        print("(heard nothing - try again)\n")
        return True

    print(f"you: {transcript}")
    reply_text = await agent.respond(transcript)
    print(f"assistant: {reply_text}\n")

    if settings.vocare_voice_reply:
        try:
            audio_reply = await synthesize_speech(client, settings, reply_text)
            play_pcm(audio_reply, sample_rate=SAMPLE_RATE_OUT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("speech_synthesis_failed", error=str(exc))
            print("(could not play spoken reply - see log; the text reply above is still valid)\n")

    return True
