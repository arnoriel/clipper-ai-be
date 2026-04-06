"""
services/fonts.py — Google Font resolver + system font fallback.
Font di-download sekali lalu di-cache ke disk & memori.
"""

import asyncio
import re
import subprocess
from pathlib import Path
from typing import Optional

from config import FONT_CACHE_DIR
from dependencies import get_http_client

# ─── System font fallback ─────────────────────────────────────────────────────

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
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            f = line.strip()
            if f.lower().endswith(".ttf") and Path(f).exists():
                return f
    except Exception:
        pass
    return None


SYSTEM_FONT: Optional[str] = find_system_font()
print(f"✅ System font: {SYSTEM_FONT or 'none'}")

# ─── In-memory font path cache ────────────────────────────────────────────────
_FONT_PATH_CACHE: dict[str, str] = {}

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
    if bold:
        variant_parts.append("bold")
    if italic:
        variant_parts.append("italic")
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
        client = get_http_client()
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

        font_resp = await client.get(ttf_url)
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
        results[oid] = SYSTEM_FONT if isinstance(result, Exception) else result  # type: ignore[assignment]

    return results
