"""
AI Viral Clipper — Python Backend
FastAPI + ffmpeg (stream-only, no permanent storage)
Deploy: Railway / Render (free tier)
"""

import os
import re
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

# Load .env file (untuk development lokal)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Viral Clipper Backend", version="3.0.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://viral-clipper-ai.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-File-Name"],
)

# ─── AI Config ────────────────────────────────────────────────────────────────
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")      # OpenAI Whisper (paid)
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")        # Groq Whisper FREE ✅

AI_MODELS = [
    "arcee-ai/trinity-large-preview:free",
]

# ─── Temp dir ─────────────────────────────────────────────────────────────────
TEMP_DIR = Path(tempfile.gettempdir()) / "clipper-ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FONT_CACHE_DIR = TEMP_DIR / "fonts"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FONT_PATH_CACHE: dict[str, str] = {}

# ══════════════════════════════════════════════════════════════════════════════
# FFMPEG DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_ffmpeg() -> str:
    candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    for candidate in candidates:
        path = candidate if Path(candidate).is_file() else shutil.which(candidate)
        if not path:
            continue
        try:
            r = subprocess.run([path, "-filters"], capture_output=True, text=True, timeout=10)
            if "drawtext" in r.stdout or "drawtext" in r.stderr:
                print(f"✅ ffmpeg (drawtext=YES): {path}")
                return path
        except Exception:
            continue
    fallback = shutil.which("ffmpeg") or "ffmpeg"
    print(f"⚠️  ffmpeg drawtext NOT available, using: {fallback}")
    return fallback

FFMPEG_BIN = find_ffmpeg()

try:
    _r = subprocess.run([FFMPEG_BIN, "-filters"], capture_output=True, text=True, timeout=10)
    DRAWTEXT_OK = "drawtext" in _r.stdout or "drawtext" in _r.stderr
except Exception:
    DRAWTEXT_OK = False
print(f"✅ drawtext support: {DRAWTEXT_OK}")


# ══════════════════════════════════════════════════════════════════════════════
# FONT RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def find_system_font() -> Optional[str]:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    try:
        result = subprocess.run(
            ["fc-list", "--format=%{file}\n"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            f = line.strip()
            if f.lower().endswith(".ttf") and Path(f).exists():
                return f
    except Exception:
        pass
    return None

SYSTEM_FONT = find_system_font()
print(f"✅ System font: {SYSTEM_FONT or 'none'}")

_TTF_USER_AGENTS = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.28) Gecko/20120306 Firefox/3.6.28",
    "Mozilla/5.0 (Linux; U; Android 2.2; en-us; ADR6300 Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
]


async def resolve_google_font(
    font_family: str,
    bold: bool = False,
    italic: bool = False,
) -> Optional[str]:
    variant_parts = []
    if bold:   variant_parts.append("bold")
    if italic: variant_parts.append("italic")
    variant   = "_".join(variant_parts) if variant_parts else "regular"
    safe_name = re.sub(r"[^\w]", "_", font_family)
    cache_key = f"{safe_name}_{variant}"

    if cache_key in _FONT_PATH_CACHE:
        return _FONT_PATH_CACHE[cache_key]

    local_path = FONT_CACHE_DIR / f"{cache_key}.ttf"
    if local_path.exists() and local_path.stat().st_size > 1000:
        _FONT_PATH_CACHE[cache_key] = str(local_path)
        return str(local_path)

    family_param  = font_family.replace(" ", "+")
    wght          = "700" if bold else "400"
    italic_prefix = "1," if italic else "0,"

    css_urls = [
        f"https://fonts.googleapis.com/css?family={family_param}:{wght}&subset=latin",
        f"https://fonts.googleapis.com/css?family={family_param}&subset=latin",
        f"https://fonts.googleapis.com/css2?family={family_param}:ital,wght@{italic_prefix}{wght}&display=swap",
    ]

    print(f"🔤 Downloading font: {font_family} ({variant})")

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            ttf_url: Optional[str] = None

            for ua in _TTF_USER_AGENTS:
                if ttf_url:
                    break
                headers = {"User-Agent": ua}

                for css_url in css_urls:
                    if ttf_url:
                        break
                    try:
                        css_resp = await client.get(css_url, headers=headers)
                        if not css_resp.is_success:
                            continue

                        all_urls = re.findall(
                            r"url\((https://fonts\.gstatic\.com/[^\)\"']+)\)",
                            css_resp.text,
                        )
                        ttf_candidates = [u for u in all_urls if u.lower().endswith(".ttf")]

                        if ttf_candidates:
                            ttf_url = ttf_candidates[0]
                            break

                    except Exception:
                        continue

            if not ttf_url:
                return SYSTEM_FONT

            font_resp = await client.get(ttf_url, timeout=30.0)
            if not font_resp.is_success or len(font_resp.content) < 1000:
                return SYSTEM_FONT

            local_path.write_bytes(font_resp.content)
            _FONT_PATH_CACHE[cache_key] = str(local_path)
            return str(local_path)

    except Exception as e:
        print(f"❌ Font download exception for '{font_family}': {e}")
        return SYSTEM_FONT


async def resolve_fonts_for_overlays(overlays: list[dict]) -> dict[str, Optional[str]]:
    if not overlays:
        return {}

    ids: list[str] = []
    coros = []

    for t in overlays:
        oid         = t.get("id", "")
        font_family = t.get("fontFamily") or "Roboto"
        bold        = bool(t.get("bold", True))
        italic      = bool(t.get("italic", False))
        ids.append(oid)
        coros.append(resolve_google_font(font_family, bold=bold, italic=italic))

    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    results: dict[str, Optional[str]] = {}
    for oid, result in zip(ids, raw_results):
        if isinstance(result, Exception):
            results[oid] = SYSTEM_FONT
        else:
            results[oid] = result

    return results


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER AUTO-SUBTITLE
# ══════════════════════════════════════════════════════════════════════════════

async def extract_audio_segment(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
) -> bool:
    """Extract audio from video segment as mp3 for Whisper."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-vn",                          # no video
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "16000",                 # Whisper prefers 16kHz
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 100


def _whisper_available_provider() -> tuple[str, str]:
    """
    Return (provider_name, api_key) for the first available STT provider.
    Priority: Groq (FREE) → OpenAI → local faster-whisper (no key needed)
    """
    if GROQ_API_KEY:
        return ("groq", GROQ_API_KEY)
    if OPENAI_API_KEY:
        return ("openai", OPENAI_API_KEY)
    return ("local", "")


async def call_whisper_groq(audio_path: Path, language: Optional[str] = None) -> dict:
    """
    Groq Whisper — FREE tier, very fast, word timestamps.
    Endpoint mirrors OpenAI's audio/transcriptions API.
    Model: whisper-large-v3-turbo (fastest) or whisper-large-v3
    """
    endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        data = {
            "model":             "whisper-large-v3-turbo",  # Free on Groq
            "response_format":   "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language:
            data["language"] = language

        print(f"   🟣 Using Groq Whisper (free) → {len(audio_bytes):,} bytes")

        resp = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            data=data,
            files={"file": ("audio.mp3", audio_bytes, "audio/mpeg")},
        )

        if not resp.is_success:
            raise HTTPException(502, f"Groq Whisper error {resp.status_code}: {resp.text[:300]}")

        result = resp.json()
        # Groq returns words inside segments; flatten if needed
        words = result.get("words", [])
        if not words:
            # Try extracting from segments
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    words.append(w)
            result["words"] = words

        return result


async def call_whisper_openai(audio_path: Path, language: Optional[str] = None) -> dict:
    """OpenAI Whisper API — requires paid credits."""
    endpoint = "https://api.openai.com/v1/audio/transcriptions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        data = {
            "model":           "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language:
            data["language"] = language

        print(f"   🔵 Using OpenAI Whisper → {len(audio_bytes):,} bytes")

        resp = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data=data,
            files={"file": ("audio.mp3", audio_bytes, "audio/mpeg")},
        )

        if not resp.is_success:
            raise HTTPException(502, f"Whisper API error {resp.status_code}: {resp.text[:300]}")

        return resp.json()


def call_whisper_local(audio_path: Path, language: Optional[str] = None) -> dict:
    """
    Local faster-whisper transcription — no API key needed, runs on CPU.
    Install: pip install faster-whisper
    Downloads model on first run (~150 MB for 'base').
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        raise RuntimeError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        )

    print(f"   ⚙️  Using local faster-whisper (CPU mode)…")
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        word_timestamps=True,
    )

    words = []
    full_text_parts = []
    for seg in segments:
        full_text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":  w.word.strip(),
                    "start": w.start,
                    "end":   w.end,
                })

    return {
        "text":     " ".join(full_text_parts),
        "language": info.language,
        "words":    words,
    }


async def call_whisper_api(audio_path: Path, language: Optional[str] = None) -> dict:
    """
    Auto-select STT provider:
      1. Groq  (GROQ_API_KEY)    — FREE, recommended
      2. OpenAI (OPENAI_API_KEY) — paid
      3. local faster-whisper    — no key needed, slower
    """
    provider, _ = _whisper_available_provider()
    print(f"🎙️  STT provider: {provider}")

    if provider == "groq":
        return await call_whisper_groq(audio_path, language)
    elif provider == "openai":
        return await call_whisper_openai(audio_path, language)
    else:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, call_whisper_local, audio_path, language
            )
        except RuntimeError as e:
            raise HTTPException(500, str(e))


def group_words_to_subtitles(
    words: list[dict],
    words_per_chunk: int = 3,
) -> list[dict]:
    """
    Group word-level timestamps into N-word subtitle chunks.
    Each word dict has keys: word, start, end.
    Returns list of { text, start, end }.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# AI HELPER
# ══════════════════════════════════════════════════════════════════════════════

async def call_openrouter(
    messages: list,
    max_tokens: int = 3000,
    temperature: float = 0.3,
) -> str:
    key = OPENROUTER_API_KEY
    if not key:
        raise HTTPException(400, "OPENROUTER_API_KEY not configured on server")

    last_error: Exception = RuntimeError("No models tried")

    for model in AI_MODELS:
        for attempt in range(2):
            try:
                print(f"🤖 model={model} attempt={attempt + 1}")
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(90.0, connect=15.0)
                ) as client:
                    resp = await client.post(
                        f"{OPENROUTER_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type":  "application/json",
                            "HTTP-Referer":  "https://viral-clipper-ai.vercel.app",
                            "X-Title":       "AI Viral Clipper",
                        },
                        json={
                            "model":       model,
                            "messages":    messages,
                            "max_tokens":  max_tokens,
                            "temperature": temperature,
                        },
                    )

                if resp.status_code == 429:
                    await asyncio.sleep(1)
                    break

                if resp.status_code in (502, 503, 504):
                    await asyncio.sleep(2)
                    continue

                if not resp.is_success:
                    raise HTTPException(502, f"OpenRouter {resp.status_code}: {resp.text[:200]}")

                content = resp.json()["choices"][0]["message"]["content"]
                return content

            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                wait = 2 if attempt == 0 else 4
                await asyncio.sleep(wait)
                continue

            except httpx.TimeoutException as e:
                last_error = e
                break

            except HTTPException:
                raise

            except Exception as e:
                last_error = e
                break

    raise HTTPException(
        502,
        f"Semua model AI gagal. Error: {type(last_error).__name__}: {str(last_error)[:200]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def seconds_to_ffmpeg(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def safe_delete(path: Optional[Path]):
    try:
        if path and path.exists():
            path.unlink()
    except Exception as e:
        print(f"⚠️  Could not delete: {path} — {e}")


def hex_to_ffmpeg_color(hex_color: str, opacity: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "000000"
    aa = format(int(opacity * 255), "02X")
    return f"0x{h.upper()}{aa}"


def escape_drawtext(text: str) -> str:
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace("'",  "\\'")
    text = text.replace(":",  "\\:")
    text = text.replace("%",  "\\%")
    text = text.replace("\n", " ")
    return text


# ── Font reference width: font size disimpan sbg "px di 1080px lebar" ─────────
FONT_REFERENCE_WIDTH = 1080.0


def build_ffmpeg_filters(
    edits: dict,
    resolved_fonts: Optional[dict[str, Optional[str]]] = None,
) -> list[str]:
    filters: list[str] = []

    # ── Aspect ratio crop ────────────────────────────────────────────────────
    aspect_ratio = edits.get("aspectRatio", "original")
    if aspect_ratio and aspect_ratio != "original":
        rw, rh = [int(x) for x in aspect_ratio.split(":")]
        crop_w = f"if(gt(iw/ih\\,{rw}/{rh})\\,trunc(ih*{rw}/{rh}/2)*2\\,iw)"
        crop_h = f"if(gt(iw/ih\\,{rw}/{rh})\\,ih\\,trunc(iw*{rh}/{rw}/2)*2)"
        filters.append(f"crop={crop_w}:{crop_h}:(iw-out_w)/2:(ih-out_h)/2")

    # ── Color eq ─────────────────────────────────────────────────────────────
    eq_parts   = []
    brightness = edits.get("brightness", 0)
    contrast   = edits.get("contrast", 0)
    saturation = edits.get("saturation", 0)
    if brightness != 0:
        eq_parts.append(f"brightness={brightness}")
    if contrast != 0:
        eq_parts.append(f"contrast={1 + contrast:.4f}")
    if saturation != 0:
        eq_parts.append(f"saturation={1 + saturation:.4f}")
    if eq_parts:
        filters.append(f"eq={':'.join(eq_parts)}")

    # ── Speed ─────────────────────────────────────────────────────────────────
    speed = edits.get("speed", 1)
    if speed and speed != 1:
        filters.append(f"setpts={1 / speed:.6f}*PTS")

    # ── Text overlays ─────────────────────────────────────────────────────────
    text_overlays = edits.get("textOverlays", [])
    if not DRAWTEXT_OK and text_overlays:
        print("⚠️  Subtitles skipped — ffmpeg missing drawtext filter")
        text_overlays = []

    for t in text_overlays:
        overlay_id = t.get("id", "")

        raw_text = t.get("text", "")
        uppercase = t.get("uppercase", False)
        if uppercase:
            raw_text = raw_text.upper()
        safe_text = escape_drawtext(raw_text)

        font_path: Optional[str] = None
        if resolved_fonts:
            font_path = resolved_fonts.get(overlay_id)
        if not font_path:
            font_path = SYSTEM_FONT
        if font_path:
            escaped_font_path = font_path.replace("\\", "\\\\").replace(":", "\\:")
            font_part = f"fontfile='{escaped_font_path}':"
        else:
            font_part = ""

        # ── Font size: disimpan sebagai px di referensi 1080px lebar.
        # Pakai ekspresi ffmpeg `w * ratio` supaya otomatis scale ke
        # resolusi output setelah crop (9:16, 1:1, dll).
        font_size_stored = float(t.get("fontSize", 36))
        font_size_ratio  = font_size_stored / FONT_REFERENCE_WIDTH
        font_size_expr   = f"w*{font_size_ratio:.6f}"

        opacity       = float(t.get("opacity", 1.0))
        font_color_hex = t.get("color", "#FFFFFF")
        font_color    = hex_to_ffmpeg_color(font_color_hex, opacity)

        raw_x = t.get("x")
        raw_y = t.get("y")
        text_align = t.get("textAlign", "center")

        if raw_x is None or abs(raw_x - 0.5) < 0.02:
            if text_align == "left":
                x = "w*0.02"
            elif text_align == "right":
                x = "w*0.98-text_w"
            else:
                x = "(w-text_w)/2"
        else:
            if text_align == "left":
                x = f"w*{raw_x:.4f}"
            elif text_align == "right":
                x = f"w*{raw_x:.4f}-text_w"
            else:
                x = f"w*{raw_x:.4f}-text_w/2"

        if raw_y is None or abs(raw_y - 0.5) < 0.02:
            y = "(h-text_h)/2"
        else:
            y = f"h*{raw_y:.4f}-text_h/2"

        enable = ""
        start_sec = t.get("startSec")
        end_sec   = t.get("endSec")
        if start_sec is not None and end_sec is not None:
            enable = f":enable='gte(t\\,{start_sec:.3f})*lte(t\\,{end_sec:.3f})'"

        outline_width = float(t.get("outlineWidth", 0))
        outline_color_hex = t.get("outlineColor", "#000000")
        border_part = ""
        if outline_width > 0:
            # Scale outline width proporsional terhadap lebar video
            outline_ratio = outline_width / FONT_REFERENCE_WIDTH
            border_color = hex_to_ffmpeg_color(outline_color_hex, 1.0)
            border_part = f"borderw=w*{outline_ratio:.6f}:bordercolor={border_color}:"

        shadow_enabled = t.get("shadowEnabled", True)
        shadow_part = ""
        if shadow_enabled:
            shadow_color_hex = t.get("shadowColor", "#000000")
            shadow_color     = hex_to_ffmpeg_color(shadow_color_hex, 0.85)
            # Scale shadow offset proporsional
            sx = float(t.get("shadowX", 2)) / FONT_REFERENCE_WIDTH * 1080
            sy = float(t.get("shadowY", 2)) / FONT_REFERENCE_WIDTH * 1080
            shadow_part = f"shadowcolor={shadow_color}:shadowx={sx:.1f}:shadowy={sy:.1f}:"
        else:
            shadow_part = "shadowcolor=0x00000000:shadowx=0:shadowy=0:"

        bg_enabled = t.get("backgroundEnabled", False)
        box_part = ""
        if bg_enabled:
            bg_color_hex  = t.get("backgroundColor", "#000000")
            bg_opacity    = float(t.get("backgroundOpacity", 0.6))
            bg_padding    = int(t.get("backgroundPadding", 10))
            box_color     = hex_to_ffmpeg_color(bg_color_hex, bg_opacity)
            box_part = f"box=1:boxcolor={box_color}:boxborderw={bg_padding}:"

        drawtext = (
            f"drawtext="
            f"{font_part}"
            f"text='{safe_text}':"
            f"fontsize={font_size_expr}:"
            f"fontcolor={font_color}:"
            f"{shadow_part}"
            f"{border_part}"
            f"{box_part}"
            f"x={x}:y={y}"
            f"{enable}"
        )

        filters.append(drawtext)

    return filters


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    provider, _ = _whisper_available_provider()
    stt_detail = {
        "groq":   "Groq Whisper (FREE) ✅",
        "openai": "OpenAI Whisper (paid)",
        "local":  "local faster-whisper (CPU)",
    }.get(provider, provider)

    return {
        "ok":          True,
        "ffmpeg":      FFMPEG_BIN,
        "drawtext":    DRAWTEXT_OK,
        "font":        SYSTEM_FONT,
        "stt_provider": provider,
        "stt_detail":  stt_detail,
        "mode":        "stream-only",
        "tmpDir":      str(TEMP_DIR),
        "ai_models":   AI_MODELS,
    }


@app.get("/api/test-font")
async def test_font(
    family: str = "Roboto",
    bold: bool = False,
    italic: bool = False,
):
    path = await resolve_google_font(family, bold=bold, italic=italic)
    if path and Path(path).exists():
        size = Path(path).stat().st_size
        return {"ok": True, "family": family, "path": path, "size_bytes": size}
    return {"ok": False, "family": family, "path": path}


# ─── POST /api/get-video-duration ─────────────────────────────────────────────
@app.post("/api/get-video-duration")
async def get_video_duration(video: UploadFile = File(...)):
    tmp_path = TEMP_DIR / f"dur_{os.urandom(8).hex()}{Path(video.filename or 'x.mp4').suffix}"
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        ffprobe_bin = FFMPEG_BIN.replace("ffmpeg", "ffprobe")
        if not Path(ffprobe_bin).is_file():
            ffprobe_bin = shutil.which("ffprobe") or "ffprobe"

        result = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_format", str(tmp_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise HTTPException(500, "ffprobe failed")
        info = json.loads(result.stdout)
        return {"duration": float(info.get("format", {}).get("duration", 0))}
    finally:
        safe_delete(tmp_path)


# ─── POST /api/analyze-video ──────────────────────────────────────────────────
@app.post("/api/analyze-video")
async def analyze_video(
    file_name: str   = Form(...),
    file_size: int   = Form(...),
    duration:  float = Form(...),
    mime_type: str   = Form(default="video/mp4"),
):
    def fmt_time(s: float) -> str:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    file_size_mb = file_size / 1_048_576

    system_prompt = "Kamu adalah analis konten viral profesional. SELALU respons Bahasa Indonesia. Hanya JSON valid."

    user_prompt = f"""Analisis file video dan identifikasi 5-8 momen viral terbaik.

INFO: nama={file_name}, ukuran={file_size_mb:.1f}MB, durasi={duration}s ({fmt_time(duration)}), format={mime_type}

Distribusi merata:
- 0-{duration*0.2:.0f}s (intro)
- {duration*0.2:.0f}-{duration*0.4:.0f}s
- {duration*0.4:.0f}-{duration*0.6:.0f}s
- {duration*0.6:.0f}-{duration*0.8:.0f}s
- {duration*0.8:.0f}-{duration}s (outro)

Aturan: integer detik, durasi 15-90s, tidak overlap, tidak melebihi {duration}s.
Kategori: funny/emotional/educational/shocking/satisfying/drama/highlight

JSON:
{{"summary":"...","totalViralPotential":7,"moments":[{{"id":"moment_1","label":"...","startTime":10,"endTime":55,"reason":"...","viralScore":8,"category":"highlight"}}]}}"""

    content = await call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=3000,
        temperature=0.3,
    )

    cleaned = re.sub(r"```json\s*|```\s*", "", content).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\[{][\s\S]*[\]}]", cleaned)
        if not match:
            raise HTTPException(502, "AI returned invalid JSON")
        parsed = json.loads(match.group(0))

    moments = []
    for i, m in enumerate(parsed.get("moments", [])):
        start = round(m.get("startTime", 0))
        end   = round(m.get("endTime", 0))
        if start < 0 or end > duration or start >= end or (end - start) < 10:
            continue
        moments.append({
            **m,
            "id":         m.get("id") or f"moment_{i+1}",
            "startTime":  start,
            "endTime":    min(end, int(duration)),
            "viralScore": max(1, min(10, m.get("viralScore", 5))),
        })

    moments.sort(key=lambda x: x["viralScore"], reverse=True)
    return {
        "moments":             moments,
        "summary":             parsed.get("summary", "Analisis selesai."),
        "totalViralPotential": parsed.get("totalViralPotential", 5),
    }


# ─── POST /api/generate-clip-content ─────────────────────────────────────────
@app.post("/api/generate-clip-content")
async def generate_clip_content(
    moment_label:    str   = Form(...),
    moment_category: str   = Form(...),
    moment_reason:   str   = Form(...),
    start_time:      float = Form(...),
    end_time:        float = Form(...),
    video_file_name: str   = Form(...),
):
    def fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"

    prompt = f"""Buat konten media sosial Bahasa Indonesia untuk:
Video: "{video_file_name}" | Klip: "{moment_label}" ({fmt(start_time)}-{fmt(end_time)}) | {moment_category}: {moment_reason}

JSON: {{"titles":["...","...","..."],"captions":["...dengan emoji","...singkat"],"hashtags":["#tag1","#tag2","#tag3","#tag4","#tag5"]}}"""

    content = await call_openrouter(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )

    cleaned = re.sub(r"```json\s*|```\s*", "", content).strip()
    return json.loads(cleaned)


# ─── POST /api/auto-subtitle ──────────────────────────────────────────────────
@app.post("/api/auto-subtitle")
async def auto_subtitle(
    video:      UploadFile = File(...),
    start_time: float      = Form(...),
    end_time:   float      = Form(...),
    words_per_chunk: int   = Form(default=3),
    language:   str        = Form(default=""),
):
    provider, _ = _whisper_available_provider()

    duration = end_time - start_time
    if duration <= 0 or duration > 600:
        raise HTTPException(400, "Invalid clip duration (must be 0–600s)")

    suffix      = Path(video.filename or "source.mp4").suffix or ".mp4"
    upload_path = TEMP_DIR / f"asr_{os.urandom(8).hex()}{suffix}"
    audio_path  = TEMP_DIR / f"asr_{os.urandom(8).hex()}.mp3"

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        print(f"🎙️  Auto-subtitle: provider={provider} start={start_time:.1f}s dur={duration:.1f}s words_per_chunk={words_per_chunk}")

        ok = await extract_audio_segment(upload_path, start_time, duration, audio_path)
        if not ok:
            raise HTTPException(500, "Failed to extract audio from video")

        result = await call_whisper_api(audio_path, language=language or None)
        words  = result.get("words", [])

        print(f"   ✅ Transcribed {len(words)} words")

        chunks = group_words_to_subtitles(words, words_per_chunk=words_per_chunk)

        return {
            "ok":       True,
            "chunks":   chunks,
            "language": result.get("language", ""),
            "full_text": result.get("text", ""),
            "word_count": len(words),
        }

    finally:
        safe_delete(upload_path)
        safe_delete(audio_path)


# ─── POST /api/export-clip ────────────────────────────────────────────────────
@app.post("/api/export-clip")
async def export_clip(
    video:     UploadFile = File(...),
    clipJson:  str        = Form(...),
    editsJson: str        = Form(...),
):
    suffix      = Path(video.filename or "source.mp4").suffix or ".mp4"
    upload_path = TEMP_DIR / f"upload_{os.urandom(8).hex()}{suffix}"

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        clip  = json.loads(clipJson)
        edits = json.loads(editsJson)

        start_sec    = float(clip["startTime"])
        duration_sec = float(clip["endTime"]) - start_sec
        speed        = float(edits.get("speed", 1))

        text_overlays = edits.get("textOverlays", [])
        resolved_fonts: dict[str, Optional[str]] = {}

        if DRAWTEXT_OK and text_overlays:
            print(f"🔤 Resolving fonts for {len(text_overlays)} overlays…")
            resolved_fonts = await resolve_fonts_for_overlays(text_overlays)

        filters      = build_ffmpeg_filters(edits, resolved_fonts=resolved_fonts)
        has_overlays = DRAWTEXT_OK and len(text_overlays) > 0

        args = [
            FFMPEG_BIN, "-y",
            "-ss", seconds_to_ffmpeg(start_sec),
            "-i",  str(upload_path),
            "-t",  seconds_to_ffmpeg(duration_sec),
        ]
        if filters:
            args += ["-vf", ",".join(filters)]
        if speed != 1:
            args += ["-af", f"atempo={max(0.5, min(2.0, speed))}"]
        args += [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]

        print(f"🎬 Export start={start_sec:.1f}s dur={duration_sec:.1f}s overlays={len(text_overlays)}")

        if has_overlays:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_data, stderr_data = await process.communicate()

            if process.returncode != 0 or len(stdout_data) == 0:
                err = stderr_data.decode("utf-8", errors="replace")
                print(f"[ffmpeg error]\n{err[-800:]}")
                safe_delete(upload_path)
                hint = ""
                if "No such file or directory" in err and "font" in err.lower():
                    hint = " (font file not found)"
                elif "drawtext" in err.lower():
                    hint = " (drawtext filter error)"
                raise HTTPException(500, f"ffmpeg failed{hint}: {err[-300:]}")

            safe_delete(upload_path)
            print(f"✅ Export OK — {len(stdout_data):,} bytes (buffered)")
            file_name = f"clip_{int(start_sec)}_{int(start_sec + duration_sec)}.mp4"

            async def yield_buffer():
                yield stdout_data

            return StreamingResponse(
                yield_buffer(),
                media_type="video/mp4",
                headers={
                    "X-File-Name":                  file_name,
                    "Access-Control-Expose-Headers": "X-File-Name",
                    "Content-Length":               str(len(stdout_data)),
                },
            )

        else:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stderr_buf: list[bytes] = []

            async def drain_stderr():
                assert process.stderr
                while chunk := await process.stderr.read(4096):
                    stderr_buf.append(chunk)

            stderr_task = asyncio.ensure_future(drain_stderr())
            file_name   = f"clip_{int(start_sec)}_{int(start_sec + duration_sec)}.mp4"

            async def stream_stdout():
                total = 0
                try:
                    while chunk := await process.stdout.read(65536):
                        total += len(chunk)
                        yield chunk
                    await process.wait()
                    await stderr_task
                    if process.returncode != 0:
                        print(f"[ffmpeg error]\n{b''.join(stderr_buf).decode(errors='replace')[-500:]}")
                    else:
                        print(f"✅ Export OK — {total:,} bytes (streaming)")
                finally:
                    safe_delete(upload_path)
                    if process.returncode is None:
                        process.kill()
                    stderr_task.cancel()

            return StreamingResponse(
                stream_stdout(),
                media_type="video/mp4",
                headers={
                    "X-File-Name":                  file_name,
                    "Access-Control-Expose-Headers": "X-File-Name",
                },
            )

    except HTTPException:
        safe_delete(upload_path)
        raise
    except Exception as e:
        safe_delete(upload_path)
        print(f"[export exception] {e}")
        raise HTTPException(500, f"Export failed: {str(e)}")


# ─── Run (dev) ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3001)),
        reload=True,
    )