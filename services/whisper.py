"""
services/whisper.py — Whisper STT helpers (Groq → OpenAI → local fallback).
"""

import subprocess
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from config import GROQ_API_KEY, OPENAI_API_KEY
from dependencies import get_http_client
from services.ffmpeg import FFMPEG_BIN


# ─── Provider selection ───────────────────────────────────────────────────────

def whisper_provider() -> tuple[str, str]:
    """Returns (provider_name, api_key). Priority: Groq → OpenAI → local."""
    if GROQ_API_KEY:
        return ("groq", GROQ_API_KEY)
    if OPENAI_API_KEY:
        return ("openai", OPENAI_API_KEY)
    return ("local", "")


# ─── Audio extraction ─────────────────────────────────────────────────────────

async def extract_audio_segment(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
) -> bool:
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab",     "128k",
        "-ar",     "16000",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 100


# ─── API callers ──────────────────────────────────────────────────────────────

async def call_whisper_groq(audio_path: Path, language: Optional[str] = None) -> dict:
    endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    data: dict = {
        "model":           "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    if language:
        data["language"] = language

    print(f"   🟣 Using Groq Whisper (free) → {len(audio_bytes):,} bytes")

    resp = await get_http_client().post(
        endpoint,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        data=data,
        files={"file": ("audio.mp3", audio_bytes, "audio/mpeg")},
    )

    if not resp.is_success:
        raise HTTPException(502, f"Groq Whisper error {resp.status_code}: {resp.text[:300]}")

    result = resp.json()
    words  = result.get("words", [])
    if not words:
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append(w)
        result["words"] = words

    return result


async def call_whisper_openai(audio_path: Path, language: Optional[str] = None) -> dict:
    endpoint = "https://api.openai.com/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    data: dict = {
        "model":           "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    if language:
        data["language"] = language

    print(f"   🔵 Using OpenAI Whisper → {len(audio_bytes):,} bytes")

    resp = await get_http_client().post(
        endpoint,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        data=data,
        files={"file": ("audio.mp3", audio_bytes, "audio/mpeg")},
    )

    if not resp.is_success:
        raise HTTPException(502, f"Whisper API error {resp.status_code}: {resp.text[:300]}")

    return resp.json()


def call_whisper_local(audio_path: Path, language: Optional[str] = None) -> dict:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

    print("   ⚙️  Using local faster-whisper (CPU mode)…")
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        word_timestamps=True,
    )

    words: list[dict] = []
    full_text_parts: list[str] = []
    for seg in segments:
        full_text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    return {"text": " ".join(full_text_parts), "language": info.language, "words": words}


async def call_whisper_api(audio_path: Path, language: Optional[str] = None) -> dict:
    import asyncio

    provider, _ = whisper_provider()
    print(f"🎙️  STT provider: {provider}")

    if provider == "groq":
        return await call_whisper_groq(audio_path, language)
    elif provider == "openai":
        return await call_whisper_openai(audio_path, language)
    else:
        return await asyncio.to_thread(call_whisper_local, audio_path, language)


# ─── Word → subtitle chunks ───────────────────────────────────────────────────

def group_words_to_subtitles(
    words: list[dict],
    words_per_chunk: int = 3,
) -> list[dict]:
    subtitles = []
    for i in range(0, len(words), words_per_chunk):
        chunk = words[i : i + words_per_chunk]
        if not chunk:
            break
        text = " ".join(w.get("word", "").strip() for w in chunk).strip()
        if not text:
            continue
        subtitles.append({
            "text":  text,
            "start": round(chunk[0].get("start", 0), 3),
            "end":   round(chunk[-1].get("end", 0), 3),
        })
    return subtitles
