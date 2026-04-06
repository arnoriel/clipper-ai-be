"""
config.py — Semua environment variables & konstanta global.
Import modul ini dari mana saja tanpa risiko circular import.
"""

import os
import tempfile
from pathlib import Path

# Load .env (dev lokal)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── AI ───────────────────────────────────────────────────────────────────────
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")

AI_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
]

# ─── Supabase / Auth ──────────────────────────────────────────────────────────
SUPABASE_URL              = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
JWT_SECRET                = os.getenv("JWT_SECRET", "CHANGE_ME_USE_RANDOM_SECRET_STRING")
JWT_ALGORITHM             = "HS256"
JWT_EXPIRE_HOURS          = int(os.getenv("JWT_EXPIRE_HOURS", "168"))  # 7 hari

# ─── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://viral-clipper-ai.vercel.app,https://ai.cuanclip.com",
).split(",")

# ─── Temp & Font dirs ─────────────────────────────────────────────────────────
TEMP_DIR = Path(tempfile.gettempdir()) / "clipper-ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FONT_CACHE_DIR = TEMP_DIR / "fonts"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
