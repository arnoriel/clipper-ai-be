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
app = FastAPI(title="AI Viral Clipper Backend", version="2.2.0")

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

# Model priority — dicoba berurutan kalau yang sebelumnya gagal/timeout
AI_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",  # paling stabil & cepat
    "mistralai/mistral-7b-instruct:free",        # fallback cepat
    "arcee-ai/trinity-large-preview:free",       # fallback terakhir
]

# ─── Temp dir ─────────────────────────────────────────────────────────────────
TEMP_DIR = Path(tempfile.gettempdir()) / "clipper-ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FFMPEG DETECTION — cari binary yang punya drawtext support
# ══════════════════════════════════════════════════════════════════════════════

def find_ffmpeg() -> str:
    """
    Cari ffmpeg binary yang punya drawtext (libfreetype) support.
    macOS bawaan /usr/bin/ffmpeg biasanya TIDAK punya drawtext.
    Homebrew ffmpeg (/opt/homebrew/bin/ffmpeg) biasanya PUNYA.
    """
    candidates = [
        "/opt/homebrew/bin/ffmpeg",  # macOS Apple Silicon Homebrew
        "/usr/local/bin/ffmpeg",      # macOS Intel Homebrew
        "ffmpeg",                      # PATH default / Linux
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
    print(f"   Fix: brew install homebrew-ffmpeg/ffmpeg/ffmpeg")
    return fallback

FFMPEG_BIN = find_ffmpeg()

# Verifikasi drawtext tersedia
try:
    _r = subprocess.run([FFMPEG_BIN, "-filters"], capture_output=True, text=True, timeout=10)
    DRAWTEXT_OK = "drawtext" in _r.stdout or "drawtext" in _r.stderr
except Exception:
    DRAWTEXT_OK = False
print(f"✅ drawtext support: {DRAWTEXT_OK}")


# ══════════════════════════════════════════════════════════════════════════════
# FONT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_font() -> Optional[str]:
    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux Ubuntu/Debian
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

SYSTEM_FONT = find_font()
print(f"✅ Font: {SYSTEM_FONT or 'none (ffmpeg default)'}")


# ══════════════════════════════════════════════════════════════════════════════
# AI HELPER — retry otomatis + fallback model
# ══════════════════════════════════════════════════════════════════════════════

async def call_openrouter(
    messages: list,
    max_tokens: int = 3000,
    temperature: float = 0.3,
) -> str:
    """
    Panggil OpenRouter dengan retry otomatis.
    - Coba setiap model di AI_MODELS secara berurutan
    - Retry 2x per model untuk network error (ReadError, ConnectError, timeout)
    """
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
                    print(f"   ⚠️  Rate limit {model}, next model…")
                    await asyncio.sleep(1)
                    break  # coba model berikutnya

                if resp.status_code in (502, 503, 504):
                    print(f"   ⚠️  Server error {resp.status_code}, retry…")
                    await asyncio.sleep(2)
                    continue

                if not resp.is_success:
                    raise HTTPException(502, f"OpenRouter {resp.status_code}: {resp.text[:200]}")

                content = resp.json()["choices"][0]["message"]["content"]
                print(f"   ✅ OK model={model}")
                return content

            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                wait = 2 if attempt == 0 else 4
                print(f"   ⚠️  Network error ({type(e).__name__}), retry in {wait}s…")
                await asyncio.sleep(wait)
                continue

            except httpx.TimeoutException as e:
                last_error = e
                print(f"   ⚠️  Timeout {model}, next model…")
                break

            except HTTPException:
                raise

            except Exception as e:
                last_error = e
                print(f"   ❌ Error {model}: {e}")
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


def build_ffmpeg_filters(edits: dict) -> list[str]:
    filters = []

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

    # ── Text overlays — hanya kalau drawtext tersedia ─────────────────────────
    text_overlays = edits.get("textOverlays", [])
    if not DRAWTEXT_OK and text_overlays:
        print("⚠️  Subtitle dilewati — ffmpeg tidak punya drawtext.")
        text_overlays = []

    for t in text_overlays:
        color = t.get("color", "#FFFFFF").replace("#", "0x")
        size  = t.get("fontSize", 36)

        raw_x = t.get("x")
        raw_y = t.get("y")

        if raw_x is None or abs(raw_x - 0.5) < 0.02:
            x = "(w-text_w)/2"
        else:
            x = f"w*{raw_x:.4f}-text_w/2"

        if raw_y is None or abs(raw_y - 0.5) < 0.02:
            y = "(h-text_h)/2"
        else:
            y = f"h*{raw_y:.4f}-text_h/2"

        # FIX: gunakan gte/lte bukan between() — hindari konflik koma di filter chain
        enable = ""
        start_sec = t.get("startSec")
        end_sec   = t.get("endSec")
        if start_sec is not None and end_sec is not None:
            enable = f":enable='gte(t\\,{start_sec:.3f})*lte(t\\,{end_sec:.3f})'"

        # Escape text
        safe_text = (
            t.get("text", "")
            .replace("\\", "\\\\\\\\")
            .replace("'",  "\\'")
            .replace(":",  "\\:")
            .replace("%",  "\\%")
        )

        # Font file
        font_part = ""
        if SYSTEM_FONT:
            is_bold   = t.get("bold", True)
            bold_font = SYSTEM_FONT.replace(".ttf", "-Bold.ttf")
            if is_bold and Path(bold_font).exists():
                font_part = f"fontfile='{bold_font}':"
            else:
                font_part = f"fontfile='{SYSTEM_FONT}':"

        shadow = "shadowcolor=black@0.8:shadowx=2:shadowy=2"
        filters.append(
            f"drawtext={font_part}text='{safe_text}':fontsize={size}:fontcolor={color}:{shadow}:x={x}:y={y}{enable}"
        )

    return filters


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "ok":        True,
        "ffmpeg":    FFMPEG_BIN,
        "drawtext":  DRAWTEXT_OK,
        "font":      SYSTEM_FONT,
        "mode":      "stream-only",
        "tmpDir":    str(TEMP_DIR),
        "ai_models": AI_MODELS,
    }


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
        filters      = build_ffmpeg_filters(edits)
        has_overlays = DRAWTEXT_OK and len(edits.get("textOverlays", [])) > 0

        # Gunakan FFMPEG_BIN yang sudah terdeteksi punya drawtext
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

        print(f"🎬 Export start={start_sec:.1f}s dur={duration_sec:.1f}s overlays={len(edits.get('textOverlays', []))}")
        if filters:
            print(f"   -vf {','.join(filters)[:300]}")

        if has_overlays:
            # ── MODE BUFFER: stdout+stderr concurrent via communicate() ──────
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
                    "X-File-Name":                   file_name,
                    "Access-Control-Expose-Headers":  "X-File-Name",
                    "Content-Length":                str(len(stdout_data)),
                },
            )

        else:
            # ── MODE STREAMING: drain stderr di background task ──────────────
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
                    "X-File-Name":                   file_name,
                    "Access-Control-Expose-Headers":  "X-File-Name",
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