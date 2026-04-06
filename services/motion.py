"""
services/motion.py — OpenCV motion tracking analysis.
Semua operasi OpenCV blocking → jalankan via asyncio.to_thread().
"""

from pathlib import Path
from typing import Optional

import dependencies as _deps  # pakai namespace agar tidak ambiguous


# ─── Math / interpolation helpers ────────────────────────────────────────────

def _fill_kf_gaps(raw: list[dict], detected: list[dict]) -> list[dict]:
    result = []
    for kf in raw:
        if kf["detected"]:
            result.append({"t": kf["t"], "cx": kf["cx"], "cy": kf["cy"]})
        else:
            prev_det = next((d for d in reversed(detected) if d["t"] <= kf["t"]), None)
            next_det = next((d for d in detected if d["t"] > kf["t"]), None)

            if prev_det and next_det:
                dt    = next_det["t"] - prev_det["t"]
                ratio = (kf["t"] - prev_det["t"]) / dt if dt > 0 else 0.0
                cx    = prev_det["cx"] + (next_det["cx"] - prev_det["cx"]) * ratio
                cy    = prev_det["cy"] + (next_det["cy"] - prev_det["cy"]) * ratio
            elif prev_det:
                cx, cy = prev_det["cx"], prev_det["cy"]
            elif next_det:
                cx, cy = next_det["cx"], next_det["cy"]
            else:
                cx, cy = 0.5, 0.4

            result.append({"t": kf["t"], "cx": cx, "cy": cy})
    return result


def _smooth_kf(keyframes: list[dict], alpha: float = 0.2) -> list[dict]:
    if not keyframes:
        return keyframes

    fwd = [{"t": keyframes[0]["t"], "cx": keyframes[0]["cx"], "cy": keyframes[0]["cy"]}]
    for kf in keyframes[1:]:
        fwd.append({
            "t":  kf["t"],
            "cx": alpha * kf["cx"] + (1.0 - alpha) * fwd[-1]["cx"],
            "cy": alpha * kf["cy"] + (1.0 - alpha) * fwd[-1]["cy"],
        })

    bwd = list(reversed(fwd))
    for i in range(1, len(bwd)):
        bwd[i]["cx"] = alpha * bwd[i]["cx"] + (1.0 - alpha) * bwd[i - 1]["cx"]
        bwd[i]["cy"] = alpha * bwd[i]["cy"] + (1.0 - alpha) * bwd[i - 1]["cy"]

    return list(reversed(bwd))


def _decimate_kf(keyframes: list[dict], max_count: int = 80) -> list[dict]:
    if len(keyframes) <= max_count:
        return keyframes
    step = len(keyframes) / max_count
    return [keyframes[int(i * step)] for i in range(max_count)]


# ─── Crop dimension calculator ────────────────────────────────────────────────

def _compute_crop_dimensions(vid_w: int, vid_h: int, aspect_ratio: str) -> tuple[int, int]:
    ar_w, ar_h = [int(x) for x in aspect_ratio.split(":")]
    if vid_w / vid_h > ar_w / ar_h:
        crop_h = vid_h
        crop_w = (int(vid_h * ar_w / ar_h) // 2) * 2
    else:
        crop_w = vid_w
        crop_h = (int(vid_w * ar_h / ar_w) // 2) * 2
    crop_w = min(crop_w, vid_w)
    crop_h = min(crop_h, vid_h)
    return crop_w, crop_h


# ─── FFmpeg crop expression builder ──────────────────────────────────────────

def build_motion_tracking_crop(
    keyframes: list[dict],
    crop_w: int,
    crop_h: int,
    vid_w: int,
    vid_h: int,
) -> str:
    max_x = max(0, vid_w - crop_w)
    max_y = max(0, vid_h - crop_h)

    def to_px_x(kf: dict) -> int:
        return int(max(0, min(max_x, kf["cx"] * vid_w - crop_w / 2)))

    def to_px_y(kf: dict) -> int:
        return int(max(0, min(max_y, kf["cy"] * vid_h - crop_h / 2)))

    if not keyframes:
        return f"crop={crop_w}:{crop_h}:{max_x // 2}:{max_y // 2}"

    if len(keyframes) == 1:
        return f"crop={crop_w}:{crop_h}:{to_px_x(keyframes[0])}:{to_px_y(keyframes[0])}"

    def build_axis_expr(get_px) -> str:
        expr = str(get_px(keyframes[-1]))

        for i in range(len(keyframes) - 2, -1, -1):
            kf0 = keyframes[i]
            kf1 = keyframes[i + 1]
            t0  = round(kf0["t"], 3)
            t1  = round(kf1["t"], 3)
            p0  = get_px(kf0)
            p1  = get_px(kf1)
            dt  = round(t1 - t0, 3)

            if dt < 0.001 or p0 == p1:
                seg = str(p0)
            else:
                dp  = p1 - p0
                seg = f"({p0}+{dp}*(t-{t0})/{dt})"

            expr = f"if(lt(t\\,{t1})\\,{seg}\\,{expr})"

        t_first = round(keyframes[0]["t"], 3)
        p_first = get_px(keyframes[0])
        expr = f"if(lt(t\\,{t_first})\\,{p_first}\\,{expr})"

        return expr

    x_expr = build_axis_expr(to_px_x)
    y_expr = build_axis_expr(to_px_y)

    return f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr}"


# ─── Main sync analysis function ─────────────────────────────────────────────

def analyze_video_motion_sync(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    aspect_ratio: str,
) -> dict:
    """
    Sinkron — dijalankan via asyncio.to_thread() dari route handler.
    Semua operasi OpenCV blocking aman di thread pool.
    """
    if not _deps.OPENCV_AVAILABLE:
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   None,
            "message":     "opencv-python-headless not installed on server",
            "available":   False,
        }

    import cv2

    # Gunakan cascade yang sudah di-preload (startup)
    face_cascade = _deps.FACE_CASCADE
    body_cascade = _deps.BODY_CASCADE
    if face_cascade is None or body_cascade is None:
        _deps.init_cascades()
        face_cascade = _deps.FACE_CASCADE
        body_cascade = _deps.BODY_CASCADE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   None,
            "message":     "Cannot open video file",
            "available":   True,
        }

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w, crop_h = _compute_crop_dimensions(vid_w, vid_h, aspect_ratio)

    duration     = max(end_sec - start_sec, 1.0)
    target_fps   = min(2.0, 120.0 / duration)
    frame_step   = max(1, int(fps / target_fps))
    start_frame  = int(start_sec * fps)
    end_frame    = int(end_sec   * fps)

    raw: list[dict] = []
    frame_idx = start_frame

    print(f"🎯 Motion analysis: {vid_w}x{vid_h}, crop={crop_w}x{crop_h}, "
          f"duration={duration:.1f}s, step={frame_step}")

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = round((frame_idx - start_frame) / fps, 3)
        scale = min(1.0, 480.0 / max(vid_w, vid_h, 1))
        small = cv2.resize(frame, (int(vid_w * scale), int(vid_h * scale))) if scale < 0.99 else frame
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        cx: Optional[float] = None
        cy: Optional[float] = None

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(int(20 * scale), int(20 * scale)),
        )

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx = (fx / scale + fw / scale / 2.0) / vid_w
            cy = (fy / scale + fh / scale / 2.0) / vid_h
        else:
            bodies = body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(30 * scale), int(30 * scale)),
            )
            if len(bodies) > 0:
                bx, by, bw, bh = max(bodies, key=lambda b: b[2] * b[3])
                cx = (bx / scale + bw / scale / 2.0) / vid_w
                cy = (by / scale + bh / scale / 2.0) / vid_h

        raw.append({
            "t":        t_sec,
            "cx":       float(cx) if cx is not None else None,
            "cy":       float(cy) if cy is not None else None,
            "detected": cx is not None,
        })

        frame_idx += frame_step

    cap.release()

    detected    = [kf for kf in raw if kf["detected"]]
    detect_rate = len(detected) / max(len(raw), 1)
    print(f"   Detected {len(detected)}/{len(raw)} frames ({detect_rate:.0%})")

    if len(detected) < 2 or detect_rate < 0.15:
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   [{"t": 0.0, "cx": 0.5, "cy": 0.4}],
            "cropW": crop_w, "cropH": crop_h,
            "vidW":  vid_w,  "vidH":  vid_h,
            "message":   "No person detected — center crop applied",
            "available": True,
        }

    filled   = _fill_kf_gaps(raw, detected)
    smoothed = _smooth_kf(filled, alpha=0.25)

    cx_vals  = [kf["cx"] for kf in smoothed]
    cy_vals  = [kf["cy"] for kf in smoothed]
    cx_range = max(cx_vals) - min(cx_vals)
    cy_range = max(cy_vals) - min(cy_vals)

    MOVEMENT_THRESHOLD = 0.06

    if cx_range < MOVEMENT_THRESHOLD and cy_range < MOVEMENT_THRESHOLD:
        avg_cx = sum(cx_vals) / len(cx_vals)
        avg_cy = sum(cy_vals) / len(cy_vals)
        print(f"   → Static: cx_range={cx_range:.3f}, cy_range={cy_range:.3f}")
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   [{"t": 0.0, "cx": round(avg_cx, 4), "cy": round(avg_cy, 4)}],
            "cropW": crop_w, "cropH": crop_h,
            "vidW":  vid_w,  "vidH":  vid_h,
            "message":   "Person is stationary — static crop applied to face",
            "available": True,
        }

    final = _decimate_kf(smoothed, max_count=80)
    print(f"   → Motion tracking: {len(final)} keyframes, "
          f"cx_range={cx_range:.3f}, cy_range={cy_range:.3f}")

    return {
        "hasTracking": True,
        "isStatic":    False,
        "keyframes": [
            {"t": round(kf["t"], 3), "cx": round(kf["cx"], 4), "cy": round(kf["cy"], 4)}
            for kf in final
        ],
        "cropW": crop_w, "cropH": crop_h,
        "vidW":  vid_w,  "vidH":  vid_h,
        "message":   f"Motion tracking enabled — {len(final)} keyframes detected",
        "available": True,
    }
