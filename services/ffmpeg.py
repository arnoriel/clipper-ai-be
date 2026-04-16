"""
services/ffmpeg.py — FFmpeg detection, helper functions, filter builders,
dan image overlay manager.
"""

import base64 as b64
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from config import TEMP_DIR
from services.fonts import SYSTEM_FONT, resolve_fonts_for_overlays  # noqa: F401 (re-exported)

# ─── FFmpeg detection ─────────────────────────────────────────────────────────

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


FFMPEG_BIN: str = find_ffmpeg()

try:
    _r = subprocess.run([FFMPEG_BIN, "-filters"], capture_output=True, text=True, timeout=10)
    DRAWTEXT_OK: bool = "drawtext" in _r.stdout or "drawtext" in _r.stderr
except Exception:
    DRAWTEXT_OK = False

print(f"✅ drawtext support: {DRAWTEXT_OK}")


# ─── Generic helpers ─────────────────────────────────────────────────────────

def seconds_to_ffmpeg(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def safe_delete(path: Optional[Path]) -> None:
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


# ─── Image overlay — base64 → temp file ──────────────────────────────────────

def save_image_overlay_to_temp(img_data_url: str, idx: int) -> Optional[Path]:
    try:
        if not img_data_url or "," not in img_data_url:
            return None

        header, data = img_data_url.split(",", 1)
        header_lower = header.lower()

        ext = ".png"
        if "jpeg" in header_lower or "jpg" in header_lower:
            ext = ".jpg"
        elif "webp" in header_lower:
            ext = ".webp"
        elif "gif" in header_lower:
            ext = ".gif"
        elif "svg" in header_lower:
            ext = ".svg"

        img_bytes = b64.b64decode(data)
        if len(img_bytes) < 16:
            print(f"⚠️  Image overlay {idx}: decoded bytes too small, skipping")
            return None

        path = TEMP_DIR / f"imgoverlay_{idx}_{os.urandom(4).hex()}{ext}"
        path.write_bytes(img_bytes)
        print(f"🖼️  Image overlay {idx} saved: {path.name} ({len(img_bytes):,} bytes)")
        return path

    except Exception as e:
        print(f"❌ Image overlay {idx} decode error: {e}")
        return None


# ─── FFmpeg filter builders ───────────────────────────────────────────────────

FONT_REFERENCE_WIDTH = 1080.0

# ─── Vignette — PNG overlay approach (fast, no per-pixel math) ───────────────
#
# Strategi: generate PNG RGBA sekali (black layer dengan alpha gradient radial),
# lalu overlay ke video via filter_complex. Jauh lebih cepat dari geq karena:
#   • PNG di-decode sekali, bukan per-frame
#   • FFmpeg optimized overlay path (SIMD/GPU friendly)
#   • Ukuran PNG kecil (~200KB), negligible I/O
#
# Alpha channel PNG:
#   0   = transparan (tengah frame, tidak menggelapkan)
#   255 = hitam solid (sudut frame, paling gelap)
#
# Parameter:
#   POWER    = 1.6  → vignette menyebar merata ke seluruh area
#   STRENGTH = 0.88 → gelap kuat di tepi, terasa cinematic

import io as _io
import numpy as _np
try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("⚠️  Pillow not installed — vignette PNG disabled")

# Cache PNG per resolusi agar tidak di-regenerate setiap export
_VIGNETTE_PNG_CACHE: dict[tuple[int, int], Path] = {}


def _make_vignette_array(width: int, height: int) -> "_np.ndarray":
    """
    Generate RGBA numpy array untuk vintage cinematic FRAME overlay.

    INI BUKAN VIGNETTE — ini adalah FRAME (bingkai film), perbedaannya:
      • Vignette : gradient gelap dari tepi ke tengah, tengah sedikit gelap
      • Film Frame: border gelap di 4 sisi, TENGAH 100% TRANSPARAN (alpha=0)

    Karakteristik frame ini:
      - Center frame: alpha = 0 (video tampil penuh, tidak ada penggelapan)
      - Border frame: alpha = 0.91 (hitam solid di tepi, seperti bingkai kamera)
      - Sudut: rounded rectangle via SDF → sudut membulat seperti frame proyektor
      - Inner edge: highlight putih tipis (efek sinar proyektor vintage)
      - Transisi inner edge: softness 2.5% → garis batas yang jelas, bukan blur lebar

    Parameter:
      frame_w  = 0.06  → frame 6% dari dimensi di tiap sisi (12% total per axis)
      corner_r = 0.05  → radius sudut membulat (proporsional terhadap frame)
      softness = 0.025 → transisi tepi dalam yang cukup halus tapi jelas
    """
    xs = _np.linspace(0.0, 1.0, width,  dtype=_np.float32)
    ys = _np.linspace(0.0, 1.0, height, dtype=_np.float32)
    xg, yg = _np.meshgrid(xs, ys)

    # Koordinat relatif dari tengah (0 = tengah, 0.5 = tepi)
    dx = _np.abs(xg - 0.5)
    dy = _np.abs(yg - 0.5)

    # Parameter frame
    frame_w  = 0.06    # lebar frame di tiap sisi (6% dari dimensi)
    corner_r = 0.05    # radius sudut rounded rectangle (normalized)
    softness = 0.025   # softness transisi inner edge — jelas tapi tidak keras

    # Inner clear area: setengah lebar/tinggi dikurangi frame_w
    inner_hw = 0.5 - frame_w   # = 0.44
    inner_hh = 0.5 - frame_w   # = 0.44

    # Rounded Rectangle SDF (Signed Distance Function)
    # SDF < 0 → di dalam (transparan, tengah frame)
    # SDF > 0 → di frame border (gelap)
    # SDF = 0 → tepat di inner edge frame
    qx  = _np.maximum(dx - (inner_hw - corner_r), 0.0)
    qy  = _np.maximum(dy - (inner_hh - corner_r), 0.0)
    sdf = _np.sqrt(qx ** 2 + qy ** 2) - corner_r

    # Alpha: 0 = transparan (tengah), 1 = frame border gelap
    raw_alpha = _np.clip(sdf / softness, 0.0, 1.0)

    # Frame darkness: hampir penuh opaque di tepi
    frame_strength = 0.91
    final_alpha    = raw_alpha * frame_strength

    # Inner highlight: kilat putih tipis di inner edge frame (vintage projector feel)
    # Peak di SDF ≈ 0 (batas dalam frame), melemah ke dalam dan keluar
    norm_dist = sdf / softness
    glow_band = _np.clip(1.0 - _np.abs(norm_dist - 0.15) * 4.5, 0.0, 1.0)
    glow_rgb  = (_np.clip(glow_band * 0.18 * (1.0 - raw_alpha * 0.7), 0.0, 1.0) * 255).astype(_np.uint8)

    arr = _np.zeros((height, width, 4), dtype=_np.uint8)
    arr[:, :, 0] = glow_rgb   # R — hitam + sedikit putih di inner edge
    arr[:, :, 1] = glow_rgb   # G
    arr[:, :, 2] = glow_rgb   # B
    arr[:, :, 3] = (final_alpha * 255).astype(_np.uint8)
    return arr


def get_vignette_png_path(width: int = 1920, height: int = 1080) -> Optional[Path]:
    """
    Kembalikan path ke vignette PNG untuk resolusi (width x height).
    Di-cache di disk (TEMP_DIR) dan memori — hanya dibuat sekali per resolusi.
    Return None jika Pillow tidak tersedia.
    """
    if not _PIL_OK:
        return None

    key = (width, height)
    if key in _VIGNETTE_PNG_CACHE:
        cached = _VIGNETTE_PNG_CACHE[key]
        if cached.exists():
            return cached

    path = TEMP_DIR / f"cinematic_frame_v3_{width}x{height}.png"

    if not (path.exists() and path.stat().st_size > 1000):
        try:
            arr = _make_vignette_array(width, height)
            img = _PILImage.fromarray(arr, mode="RGBA")
            img.save(str(path), format="PNG", compress_level=1)  # level 1 = fastest write
            print(f"🎞️  Cinematic frame PNG generated: {path.name} ({path.stat().st_size:,} bytes)")
        except Exception as ex:
            print(f"⚠️  Vignette PNG generation failed: {ex}")
            return None

    _VIGNETTE_PNG_CACHE[key] = path
    return path


# ─── Intro copywriting text (0 → 4 s, fade-out from 3 → 4 s) ────────────────

def build_intro_text_filter(
    intro_text: str,
    font_path: Optional[str],
    aspect_ratio: str = "original",
) -> list[str]:
    """
    Renders a bold copywriting headline over the first 4 seconds of the clip.
    Appears immediately at t=0, fades out smoothly from t=3 to t=4.
    Font style adapts to the active subtitle preset font for visual consistency.
    """
    if not intro_text or not intro_text.strip():
        return []

    safe = escape_drawtext(intro_text.strip().upper())

    if font_path:
        escaped_fp = font_path.replace("\\", "\\\\").replace(":", "\\:")
        font_part = f"fontfile=\'{escaped_fp}\':"
    else:
        font_part = ""

    # ~6% of frame width — large, punchy, dominant
    font_size_expr = "w*0.062"

    # Upper area: 14% from top on portrait, 16% on other ratios
    y_ratio = 0.14 if aspect_ratio == "9:16" else 0.16
    y_expr  = f"h*{y_ratio:.4f}"

    # Alpha: hold 1.0 for first 3s, then linear fade to 0 by t=4
    alpha_expr = "if(lte(t\\,3.0)\\,1.0\\,if(lte(t\\,4.0)\\,(4.0-t)\\,0.0))"

    enable_expr = "between(t\\,0\\,4)"

    filters = []

    # Subtle shadow pass — offset by 3px for depth
    shadow_f = (
        f"drawtext="
        f"{font_part}"
        f"text=\'{safe}\':"
        f"fontsize={font_size_expr}:"
        f"fontcolor=0x00000099:"
        f"x=(w-text_w)/2+3:y={y_expr}+3:"
        f"alpha=\'{alpha_expr}\':"
        f"enable=\'{enable_expr}\'"
    )
    filters.append(shadow_f)

    # Main white text with black outline
    main_f = (
        f"drawtext="
        f"{font_part}"
        f"text=\'{safe}\':"
        f"fontsize={font_size_expr}:"
        f"fontcolor=0xFFFFFFFF:"
        f"borderw=w*0.004:"
        f"bordercolor=0x000000FF:"
        f"x=(w-text_w)/2:y={y_expr}:"
        f"alpha=\'{alpha_expr}\':"
        f"enable=\'{enable_expr}\'"
    )
    filters.append(main_f)

    return filters


def build_ffmpeg_filters(
    edits: dict,
    resolved_fonts: Optional[dict[str, Optional[str]]] = None,
    motion_keyframes: Optional[list[dict]] = None,
    vid_info: Optional[dict] = None,
    intro_text: Optional[str] = None,
    intro_font_path: Optional[str] = None,
) -> list[str]:
    from services.motion import build_motion_tracking_crop, _compute_crop_dimensions  # local import avoids circular

    filters: list[str] = []

    aspect_ratio = edits.get("aspectRatio", "original")

    if motion_keyframes and aspect_ratio and aspect_ratio != "original" and vid_info:
        vid_w = int(vid_info.get("w", 1920))
        vid_h = int(vid_info.get("h", 1080))
        crop_w, crop_h = _compute_crop_dimensions(vid_w, vid_h, aspect_ratio)
        tracking_crop = build_motion_tracking_crop(
            motion_keyframes, crop_w, crop_h, vid_w, vid_h
        )
        filters.append(tracking_crop)
        filters.append("setsar=1:1")
        print(f"🎯 Motion crop: {crop_w}x{crop_h}, {len(motion_keyframes)} keyframes")

    elif aspect_ratio and aspect_ratio != "original":
        rw, rh = [int(x) for x in aspect_ratio.split(":")]
        crop_w_expr = f"if(gt(iw/ih\\,{rw}/{rh})\\,trunc(ih*{rw}/{rh}/2)*2\\,iw)"
        crop_h_expr = f"if(gt(iw/ih\\,{rw}/{rh})\\,ih\\,trunc(iw*{rh}/{rw}/2)*2)"
        filters.append(f"crop={crop_w_expr}:{crop_h_expr}:(iw-out_w)/2:(ih-out_h)/2")
        filters.append("setsar=1:1")

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

    speed = edits.get("speed", 1)
    if speed and speed != 1:
        filters.append(f"setpts={1 / speed:.6f}*PTS")

    text_overlays = edits.get("textOverlays", [])
    if not DRAWTEXT_OK and text_overlays:
        print("⚠️  Subtitles skipped — ffmpeg missing drawtext filter")
        text_overlays = []

    for t in text_overlays:
        overlay_id = t.get("id", "")

        raw_text  = t.get("text", "")
        if t.get("uppercase", False):
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

        font_size_stored = float(t.get("fontSize", 36))
        font_size_ratio  = font_size_stored / FONT_REFERENCE_WIDTH
        font_size_expr   = f"w*{font_size_ratio:.6f}"

        opacity        = float(t.get("opacity", 1.0))
        font_color_hex = t.get("color", "#FFFFFF")
        font_color     = hex_to_ffmpeg_color(font_color_hex, opacity)

        raw_x      = t.get("x")
        raw_y      = t.get("y")
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
            enable = f":enable=between(t\\,{start_sec:.3f}\\,{end_sec:.3f})"

        outline_width     = float(t.get("outlineWidth", 0))
        outline_color_hex = t.get("outlineColor", "#000000")
        border_part = ""
        if outline_width > 0:
            outline_ratio = outline_width / FONT_REFERENCE_WIDTH
            border_color  = hex_to_ffmpeg_color(outline_color_hex, 1.0)
            border_part   = f"borderw=w*{outline_ratio:.6f}:bordercolor={border_color}:"

        shadow_enabled = t.get("shadowEnabled", True)
        if shadow_enabled:
            shadow_color_hex = t.get("shadowColor", "#000000")
            shadow_color     = hex_to_ffmpeg_color(shadow_color_hex, 0.85)
            sx = float(t.get("shadowX", 2)) / FONT_REFERENCE_WIDTH * 1080
            sy = float(t.get("shadowY", 2)) / FONT_REFERENCE_WIDTH * 1080
            shadow_part = f"shadowcolor={shadow_color}:shadowx={sx:.1f}:shadowy={sy:.1f}:"
        else:
            shadow_part = "shadowcolor=0x00000000:shadowx=0:shadowy=0:"

        bg_enabled = t.get("backgroundEnabled", False)
        box_part   = ""
        if bg_enabled:
            bg_color_hex = t.get("backgroundColor", "#000000")
            bg_opacity   = float(t.get("backgroundOpacity", 0.6))
            bg_padding   = int(t.get("backgroundPadding", 10))
            box_color    = hex_to_ffmpeg_color(bg_color_hex, bg_opacity)
            box_part     = f"box=1:boxcolor={box_color}:boxborderw={bg_padding}:"

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

    # ── Intro copywriting text (0–4 s, fade out 3→4 s) ──────────────────────
    # NOTE: Vignette is now applied as PNG overlay in build_filter_complex_with_vignette()
    if DRAWTEXT_OK and intro_text and intro_text.strip():
        intro_filters = build_intro_text_filter(intro_text, intro_font_path, aspect_ratio)
        filters.extend(intro_filters)
        print(f"📝 Intro text: '{intro_text[:40]}' ({len(intro_filters)} layers)")

    return filters


def build_filter_complex_with_images(
    base_filters: list[str],
    image_overlays: list[dict],
    img_paths: list[Path],
    vignette_png_path: Optional[Path] = None,
) -> tuple[str, str]:
    """
    Build filter_complex string untuk video + image overlays + vignette PNG.

    Input index assignment:
      [0:v]  = video utama
      [1:v]  = vignette PNG (jika ada)
      [2:v]+ = image overlays user

    Vignette diinjeksi SETELAH semua filter video (crop, eq, drawtext),
    namun SEBELUM image overlays user — hasilnya vignette berada di bawah
    watermark/logo tapi di atas video asli.
    """
    parts: list[str] = []

    # ── Step 1: Apply base filters ke video ─────────────────────────────────
    if base_filters:
        parts.append(f"[0:v]{','.join(base_filters)}[vafter_filters]")
    else:
        parts.append("[0:v]null[vafter_filters]")

    current = "vafter_filters"

    # ── Step 2: Overlay vignette PNG (input index 1) ─────────────────────────
    if vignette_png_path is not None:
        vig_input_idx = 1
        # Scale vignette PNG to match video dimensions using scale2ref,
        # then overlay at (0,0). format=auto handles YUV↔RGBA conversion safely.
        # scale2ref: scale [1:v] (PNG) to same W×H as [current] (video).
        parts.append(
            f"[{vig_input_idx}:v][{current}]scale2ref[vig_scaled][vbase_vig]"
        )
        parts.append(
            f"[vbase_vig][vig_scaled]overlay=x=0:y=0:format=auto[vafter_vig]"
        )
        current = "vafter_vig"
        img_offset = 2  # image overlays start at input index 2
        print(f"🎞️  Vignette PNG injected into filter_complex")
    else:
        img_offset = 1  # no vignette, image overlays start at index 1

    # ── Step 3: Image overlays user ──────────────────────────────────────────
    for i, (img, img_path) in enumerate(zip(image_overlays, img_paths)):
        input_idx   = i + img_offset
        width_ratio = max(0.01, min(1.0, float(img.get("width", 0.25))))
        opacity     = max(0.0,  min(1.0, float(img.get("opacity", 1.0))))
        x_pct       = float(img.get("x", 0.5))
        y_pct       = float(img.get("y", 0.1))
        start_sec   = img.get("startSec")
        end_sec     = img.get("endSec")

        raw_label   = f"imgraw{i}"
        alpha_label = f"imgalpha{i}"
        vref_label  = f"vref{i}"
        vbase_label = f"vbase_ov{i}"
        next_base   = f"vbase{i + 1}"

        parts.append(f"[{current}]split[{vref_label}][{vbase_label}]")
        parts.append(
            f"[{input_idx}:v][{vref_label}]"
            f"scale=w=rw*{width_ratio:.4f}:h=-1"
            f"[{raw_label}]"
        )
        parts.append(
            f"[{raw_label}]format=rgba,"
            f"colorchannelmixer=aa={opacity:.3f}"
            f"[{alpha_label}]"
        )

        x_expr = f"W*{x_pct:.4f}-w/2"
        y_expr = f"H*{y_pct:.4f}-h/2"

        enable_part = ""
        if start_sec is not None and end_sec is not None:
            enable_part = (
                f":enable='between(t,"
                f"{float(start_sec):.3f},"
                f"{float(end_sec):.3f})'"
            )

        parts.append(
            f"[{vbase_label}][{alpha_label}]"
            f"overlay=x={x_expr}:y={y_expr}{enable_part}"
            f"[{next_base}]"
        )

        current = next_base

    return ";".join(parts), current