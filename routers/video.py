"""
routers/video.py — Semua route yang berhubungan dengan video & AI.

Endpoints:
  GET  /api/ping
  GET  /api/health
  GET  /api/test-font
  POST /api/get-video-duration
  POST /api/analyze-motion
  POST /api/analyze-video
  POST /api/generate-clip-content
  POST /api/auto-subtitle
  POST /api/export-clip
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import File as FAFile
from fastapi import UploadFile as FAUploadFile
from fastapi.responses import StreamingResponse

from config import TEMP_DIR
from dependencies import OPENCV_AVAILABLE
from services.ai import call_openrouter
from services.auth_utils import get_current_user
from services.ffmpeg import (
    DRAWTEXT_OK,
    FFMPEG_BIN,
    build_ffmpeg_filters,
    build_filter_complex_with_images,
    get_vignette_png_path,
    build_intro_text_filter,
    safe_delete,
    save_image_overlay_to_temp,
    seconds_to_ffmpeg,
)
from services.fonts import SYSTEM_FONT, resolve_fonts_for_overlays, resolve_google_font
from services.motion import analyze_video_motion_sync
from services.supabase import supa_deduct_credit, supa_get_user_credits
from services.whisper import (
    call_whisper_api,
    extract_audio_segment,
    group_words_to_subtitles,
    whisper_provider,
)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Utility routes
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/api/ping")
async def ping():
    """Keep-alive — daftarkan ke UptimeRobot setiap 14 menit."""
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/api/health")
async def health():
    provider, _ = whisper_provider()
    stt_detail = {
        "groq":   "Groq Whisper (FREE) ✅",
        "openai": "OpenAI Whisper (paid)",
        "local":  "local faster-whisper (CPU)",
    }.get(provider, provider)

    return {
        "ok":           True,
        "ffmpeg":       FFMPEG_BIN,
        "drawtext":     DRAWTEXT_OK,
        "font":         SYSTEM_FONT,
        "stt_provider": provider,
        "stt_detail":   stt_detail,
        "opencv":       OPENCV_AVAILABLE,
        "mode":         "stream-only",
        "tmpDir":       str(TEMP_DIR),
    }


@router.get("/api/test-font")
async def test_font(
    family: str  = "Roboto",
    bold:   bool = False,
    italic: bool = False,
):
    path = await resolve_google_font(family, bold=bold, italic=italic)
    if path and Path(path).exists():
        size = Path(path).stat().st_size
        return {"ok": True, "family": family, "path": path, "size_bytes": size}
    return {"ok": False, "family": family, "path": path}


# ──────────────────────────────────────────────────────────────────────────────
# Video info
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/get-video-duration")
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
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(500, "ffprobe failed")

        info = json.loads(result.stdout)
        return {"duration": float(info.get("format", {}).get("duration", 0))}
    finally:
        safe_delete(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Motion analysis
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/analyze-motion")
async def analyze_motion_endpoint(
    video:        UploadFile = File(...),
    start_time:   float      = Form(...),
    end_time:     float      = Form(...),
    aspect_ratio: str        = Form(...),
):
    if not OPENCV_AVAILABLE:
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   None,
            "message":     "opencv-python-headless not installed — install it for motion tracking",
            "available":   False,
        }

    suffix   = Path(video.filename or "source.mp4").suffix or ".mp4"
    tmp_path = TEMP_DIR / f"motion_{os.urandom(8).hex()}{suffix}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        duration = end_time - start_time
        if duration <= 0 or duration > 3600:
            raise HTTPException(400, "Invalid clip duration (must be 0–3600s)")

        print(f"🎯 Analyze motion: {aspect_ratio}, {start_time:.1f}s–{end_time:.1f}s")

        result = await asyncio.to_thread(
            analyze_video_motion_sync, tmp_path, start_time, end_time, aspect_ratio
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[analyze-motion error] {e}")
        raise HTTPException(500, f"Motion analysis failed: {str(e)[:200]}")
    finally:
        safe_delete(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# AI video analysis
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/analyze-video")
async def analyze_video(
    file_name:    str   = Form(...),
    file_size:    int   = Form(...),
    duration:     float = Form(...),
    mime_type:    str   = Form(default="video/mp4"),
    num_clips:    int   = Form(default=5),
    current_user: dict  = Depends(get_current_user),
):
    num_clips = max(1, min(7, num_clips))

    credits = await supa_get_user_credits(current_user["sub"])
    if credits <= 0:
        raise HTTPException(402, "Kredit tidak cukup. Silakan top up kredit kamu.")

    def fmt_time(s: float) -> str:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    file_size_mb  = file_size / 1_048_576
    system_prompt = (
        "Kamu adalah analis konten viral profesional. "
        "SELALU respons Bahasa Indonesia. Hanya JSON valid."
    )
    user_prompt = f"""Analisis file video dan identifikasi TEPAT {num_clips} momen viral terbaik yang paling berpotensi viral di media sosial.

INFO: nama={file_name}, ukuran={file_size_mb:.1f}MB, durasi={duration}s ({fmt_time(duration)}), format={mime_type}

Distribusi merata di seluruh video:
- 0-{duration*0.25:.0f}s (bagian awal)
- {duration*0.25:.0f}-{duration*0.5:.0f}s (bagian tengah awal)
- {duration*0.5:.0f}-{duration*0.75:.0f}s (bagian tengah akhir)
- {duration*0.75:.0f}-{duration}s (bagian akhir)

Aturan WAJIB:
- Kembalikan TEPAT {num_clips} momen, tidak lebih tidak kurang
- Gunakan integer detik, durasi 15-90s per clip
- Tidak boleh overlap antar momen
- Tidak melebihi {duration}s
- Pilih momen dengan viralScore tertinggi
- Kategori: funny/emotional/educational/shocking/satisfying/drama/highlight

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
            "id":         m.get("id") or f"moment_{i + 1}",
            "startTime":  start,
            "endTime":    min(end, int(duration)),
            "viralScore": max(1, min(10, m.get("viralScore", 5))),
        })

    moments.sort(key=lambda x: x["viralScore"], reverse=True)
    moments = moments[:num_clips]

    deducted  = await supa_deduct_credit(current_user["sub"])
    remaining = await supa_get_user_credits(current_user["sub"])
    if deducted:
        print(
            f"💳 Credit deducted: user={current_user['email']} "
            f"| analyze-video | remaining={remaining}"
        )

    return {
        "moments":             moments,
        "summary":             parsed.get("summary", "Analisis selesai."),
        "totalViralPotential": parsed.get("totalViralPotential", 5),
        "credits_remaining":   remaining,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generate clip content (captions / hooks)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/generate-clip-content")
async def generate_clip_content(
    moment_label:    str   = Form(...),
    moment_category: str   = Form(...),
    moment_reason:   str   = Form(...),
    start_time:      float = Form(...),
    end_time:        float = Form(...),
    video_file_name: str   = Form(...),
    current_user:    dict  = Depends(get_current_user),
):
    credits = await supa_get_user_credits(current_user["sub"])
    if credits <= 0:
        raise HTTPException(402, "Kredit tidak cukup. Silakan top up kredit kamu.")

    def fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"

    prompt = f"""..."""  # tidak berubah

    content = await call_openrouter(...)  # tidak berubah

    cleaned = re.sub(r"```json\s*|```\s*", "", content).strip()
    result  = json.loads(cleaned)

    deducted  = await supa_deduct_credit(current_user["sub"])
    remaining = await supa_get_user_credits(current_user["sub"])
    if deducted:
        print(
            f"💳 Credit deducted: user={current_user['email']} "
            f"| generate-clip-content | remaining={remaining}"
        )

    result["credits_remaining"] = remaining
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Auto subtitle (Whisper)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/auto-subtitle")
async def auto_subtitle(
    video:           FAUploadFile = FAFile(...),
    start_time:      float        = Form(...),
    end_time:        float        = Form(...),
    words_per_chunk: int          = Form(default=3),
    language:        str          = Form(default=""),
    current_user:    dict         = Depends(get_current_user),
):
    credits = await supa_get_user_credits(current_user["sub"])
    if credits <= 0:
        raise HTTPException(402, "Kredit tidak cukup. Silakan top up kredit kamu.")

    provider, _ = whisper_provider()
    duration    = end_time - start_time
    if duration <= 0 or duration > 600:
        raise HTTPException(400, "Invalid clip duration (must be 0–600s)")

    suffix      = Path(video.filename or "source.mp4").suffix or ".mp4"
    upload_path = TEMP_DIR / f"asr_{os.urandom(8).hex()}{suffix}"
    audio_path  = TEMP_DIR / f"asr_{os.urandom(8).hex()}.mp3"

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        print(
            f"🎙️  Auto-subtitle: provider={provider} user={current_user['email']} "
            f"start={start_time:.1f}s dur={duration:.1f}s words_per_chunk={words_per_chunk}"
        )

        ok = await extract_audio_segment(upload_path, start_time, duration, audio_path)
        if not ok:
            raise HTTPException(500, "Failed to extract audio from video")

        result = await call_whisper_api(audio_path, language=language or None)
        words  = result.get("words", [])
        print(f"   ✅ Transcribed {len(words)} words")
        chunks = group_words_to_subtitles(words, words_per_chunk=words_per_chunk)

        deducted  = await supa_deduct_credit(current_user["sub"])
        remaining = await supa_get_user_credits(current_user["sub"])
        if deducted:
            print(
                f"💳 Credit deducted: user={current_user['email']} "
                f"| auto-subtitle | remaining={remaining}"
            )

        return {
            "ok":                True,
            "chunks":            chunks,
            "language":          result.get("language", ""),
            "full_text":         result.get("text", ""),
            "word_count":        len(words),
            "credits_remaining": remaining,
        }
    finally:
        safe_delete(upload_path)
        safe_delete(audio_path)


# ──────────────────────────────────────────────────────────────────────────────
# Export clip
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/export-clip")
async def export_clip(
    video:        UploadFile = File(...),
    clipJson:     str        = Form(...),
    editsJson:    str        = Form(...),
    current_user: dict       = Depends(get_current_user),
):
    suffix      = Path(video.filename or "source.mp4").suffix or ".mp4"
    upload_path = TEMP_DIR / f"upload_{os.urandom(8).hex()}{suffix}"
    img_temp_files: list[Path] = []

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        clip  = json.loads(clipJson)
        edits = json.loads(editsJson)

        start_sec    = float(clip["startTime"])
        duration_sec = float(clip["endTime"]) - start_sec
        speed        = float(edits.get("speed", 1))

        text_overlays  = edits.get("textOverlays", [])
        image_overlays = edits.get("imageOverlays", [])
        intro_text: Optional[str] = edits.get("introText") or None
        aspect_ratio_val: str     = edits.get("aspectRatio", "original")

        motion_keyframes: Optional[list[dict]] = edits.get("motionKeyframes") or None
        motion_vid_w: Optional[int]            = edits.get("motionVidW")
        motion_vid_h: Optional[int]            = edits.get("motionVidH")

        vid_info: Optional[dict] = None
        if motion_keyframes and motion_vid_w and motion_vid_h:
            vid_info = {"w": int(motion_vid_w), "h": int(motion_vid_h)}
            print(
                f"🎯 Export with motion tracking: {len(motion_keyframes)} kf, "
                f"vid={motion_vid_w}x{motion_vid_h}"
            )

        # Resolve Google Fonts secara concurrent
        resolved_fonts: dict[str, Optional[str]] = {}
        if DRAWTEXT_OK and text_overlays:
            print(f"🔤 Resolving fonts for {len(text_overlays)} text overlays…")
            resolved_fonts = await resolve_fonts_for_overlays(text_overlays)

        # Resolve font for intro copywriting text
        # Use the same font as the first subtitle overlay for visual consistency
        intro_font_path: Optional[str] = None
        if DRAWTEXT_OK and intro_text:
            from services.fonts import resolve_google_font, SYSTEM_FONT as _SYS_FONT
            _preset_font = None
            if text_overlays:
                _first = text_overlays[0]
                _preset_font = _first.get("fontFamily") or None
            if _preset_font:
                try:
                    intro_font_path = await resolve_google_font(
                        _preset_font, bold=True, italic=False
                    )
                except Exception:
                    intro_font_path = _SYS_FONT
            else:
                intro_font_path = _SYS_FONT
            print(f"📝 Intro font resolved: {intro_font_path}")

        # Decode & save image overlays ke temp files
        valid_image_overlays: list[dict] = []
        for i, img in enumerate(image_overlays):
            src = img.get("src", "")
            if not src:
                continue
            path = save_image_overlay_to_temp(src, i)
            if path:
                img_temp_files.append(path)
                valid_image_overlays.append(img)

        has_text_overlays  = DRAWTEXT_OK and bool(text_overlays)
        has_image_overlays = bool(valid_image_overlays)

        base_filters = build_ffmpeg_filters(
            edits,
            resolved_fonts,
            motion_keyframes=motion_keyframes,
            vid_info=vid_info,
            intro_text=intro_text,
            intro_font_path=intro_font_path,
        )

        # ── Vignette PNG — generate once, overlay via filter_complex ─────────
        # Tentukan dimensi frame setelah crop/resize
        if vid_info:
            _vid_w, _vid_h = int(vid_info["w"]), int(vid_info["h"])
        else:
            _vid_w, _vid_h = 1920, 1080  # sensible default; scale2ref handles mismatch
        vignette_path: Optional[Path] = get_vignette_png_path(_vid_w, _vid_h)
        has_vignette = vignette_path is not None

        print(
            f"🎬 Export start={start_sec:.1f}s dur={duration_sec:.1f}s "
            f"text={len(text_overlays)} images={len(valid_image_overlays)} "
            f"motion={'yes' if motion_keyframes else 'no'} "
            f"vignette={'yes' if has_vignette else 'no'}"
        )

        # ── Build ffmpeg args ────────────────────────────────────────────────
        # Vignette selalu pakai filter_complex (butuh 2 input: video + PNG)
        use_filter_complex = has_image_overlays or has_vignette

        args = [
            FFMPEG_BIN, "-y",
            "-ss", seconds_to_ffmpeg(start_sec),
            "-t",  seconds_to_ffmpeg(duration_sec),
            "-i",  str(upload_path),
        ]

        if use_filter_complex:
            # Input order: [0]=video, [1]=vignette PNG (jika ada), [2+]=image overlays
            if has_vignette:
                args += ["-loop", "1", "-t", seconds_to_ffmpeg(duration_sec), "-i", str(vignette_path)]
            for img_path in img_temp_files:
                args += ["-loop", "1", "-t", seconds_to_ffmpeg(duration_sec), "-i", str(img_path)]

            filter_complex, final_label = build_filter_complex_with_images(
                base_filters,
                valid_image_overlays,
                img_temp_files,
                vignette_png_path=vignette_path,
            )
            args += [
                "-filter_complex", filter_complex,
                "-map", f"[{final_label}]",
                "-map", "0:a?",
            ]
            if speed != 1:
                args += ["-af", f"atempo={max(0.5, min(2.0, speed))}"]
            args += ["-t", seconds_to_ffmpeg(duration_sec)]

        else:
            if base_filters:
                args += ["-vf", ",".join(base_filters)]
            if speed != 1:
                args += ["-af", f"atempo={max(0.5, min(2.0, speed))}"]

        args += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]

        use_buffered = use_filter_complex or has_text_overlays or bool(motion_keyframes)
        file_name    = f"clip_{int(start_sec)}_{int(start_sec + duration_sec)}.mp4"

        # ── Buffered mode (overlays / motion) ───────────────────────────────
        if use_buffered:
            print(f"[ffmpeg cmd] {' '.join(str(a) for a in args)}")
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_data, stderr_data = await process.communicate()

            for p in img_temp_files:
                safe_delete(p)
            img_temp_files.clear()

            if process.returncode != 0 or len(stdout_data) == 0:
                err  = stderr_data.decode("utf-8", errors="replace")
                hint = ""
                if "No such file or directory" in err and "font" in err.lower():
                    hint = " (font file not found)"
                elif "drawtext" in err.lower():
                    hint = " (drawtext filter error)"
                elif "scale2ref" in err.lower() or "overlay" in err.lower():
                    hint = " (image overlay filter error)"
                elif "crop" in err.lower():
                    hint = " (motion tracking crop error)"
                print(f"[ffmpeg error]\n{err[-800:]}")
                safe_delete(upload_path)
                raise HTTPException(500, f"ffmpeg failed{hint}: {err[-300:]}")

            safe_delete(upload_path)
            print(
                f"✅ Export OK — {len(stdout_data):,} bytes "
                f"(buffered, images={len(valid_image_overlays)}, "
                f"motion={'yes' if motion_keyframes else 'no'}, "
                f"dur={duration_sec:.1f}s)"
            )

            async def yield_buffer():
                yield stdout_data

            return StreamingResponse(
                yield_buffer(),
                media_type="video/mp4",
                headers={
                    "X-File-Name":                   file_name,
                    "Access-Control-Expose-Headers":  "X-File-Name",
                    "Content-Length":                 str(len(stdout_data)),
                },
            )

        # ── Streaming mode (no overlays) ─────────────────────────────────────
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

            async def stream_stdout():
                total = 0
                try:
                    while chunk := await process.stdout.read(65536):
                        total += len(chunk)
                        yield chunk
                    await process.wait()
                    await stderr_task
                    if process.returncode != 0:
                        print(
                            f"[ffmpeg error]\n"
                            f"{b''.join(stderr_buf).decode(errors='replace')[-500:]}"
                        )
                    else:
                        print(f"✅ Export OK — {total:,} bytes (streaming, dur={duration_sec:.1f}s)")
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
        for p in img_temp_files:
            safe_delete(p)
        raise
    except Exception as e:
        safe_delete(upload_path)
        for p in img_temp_files:
            safe_delete(p)
        print(f"[export exception] {e}")
        raise HTTPException(500, f"Export failed: {str(e)}")
