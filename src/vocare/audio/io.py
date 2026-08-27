from __future__ import annotations

import io
import wave

import numpy as np
import sounddevice as sd

# Must match the Gemini Live API's native audio format so no resampling code
# is needed: input 16-bit PCM mono 16kHz, output 16-bit PCM mono 24kHz.
# https://ai.google.dev/gemini-api/docs/live-api/capabilities
SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHANNELS = 1
DTYPE = "int16"


QUIT_COMMANDS = {"q", "quit", "exit"}


def record_until_enter() -> bytes | None:
    """Push-to-talk capture: press Enter to start, press Enter again to stop.

    Chosen over voice-activity-detection for v1 because it's simple and 100%
    reliable to reason about turn-taking - see plan.md's open decisions.

    Quitting is an explicit typed command ('q'/'quit'/'exit'), not "press
    Enter without recording anything" - a mic reliably captures *something*
    even when you meant to record nothing (room noise, a breath), and
    speech-to-text will often turn that into a short bogus word rather than
    an empty transcript, so silence is not a reliable quit signal.

    Returns raw 16-bit PCM mono samples at SAMPLE_RATE_IN, or None if the
    user typed a quit command instead of starting to record.
    """
    command = input("Press Enter to start speaking (or type 'q' to quit): ").strip().lower()
    if command in QUIT_COMMANDS:
        return None
    print("Recording... press Enter to stop.")
    frames: list[np.ndarray] = []

    def callback(indata: np.ndarray, frame_count: int, time_info: object, status: object) -> None:
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE_IN, channels=CHANNELS, dtype=DTYPE, callback=callback
    ):
        input()

    if not frames:
        return b""
    audio = np.concatenate(frames, axis=0)
    return audio.tobytes()


def play_pcm(data: bytes, sample_rate: int = SAMPLE_RATE_OUT) -> None:
    """Blocking playback of raw 16-bit PCM mono audio."""
    if not data:
        return
    audio = np.frombuffer(data, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate, blocking=True)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE_IN) -> bytes:
    """Wrap raw PCM16 mono samples in a WAV container - needed for the
    transcription call, which goes through the standard multimodal
    generateContent endpoint (expects a real audio file format), not the
    Live API's raw-PCM streaming input."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
