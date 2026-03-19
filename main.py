"""
AI Viral Clipper — Python Backend
FastAPI + ffmpeg (stream-only, no permanent storage)
Deploy: Railway / Render (free tier)
"""

import os
import re
import json
import base64 as b64
import asyncio
import tempfile
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote
import hashlib

# Load .env file (untuk development lokal)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi import File as FAFile, UploadFile as FAUploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator
import shutil

# Auth deps — pip install passlib[bcrypt] python-jose[cryptography]
from passlib.context import CryptContext
from jose import JWTError, jwt

# ─── OpenCV availability check ────────────────────────────────────────────────
try:
    import cv2 as _cv2
    _OPENCV_AVAILABLE = True
    print(f"✅ OpenCV available: {_cv2.__version__}")
except ImportError:
    _OPENCV_AVAILABLE = False
    print("⚠️  OpenCV not available. Install: pip install opencv-python-headless")

# ─── Pre-load OpenCV cascades at module level (OPTIMIZED) ─────────────────────
# Menghindari load ulang CascadeClassifier di setiap request analyze-motion
_FACE_CASCADE: Optional[object] = None
_BODY_CASCADE: Optional[object] = None

def _init_cascades():
    global _FACE_CASCADE, _BODY_CASCADE
    if _OPENCV_AVAILABLE and _FACE_CASCADE is None:
        import cv2
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _BODY_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_upperbody.xml"
        )
        print("✅ OpenCV cascades pre-loaded")

# ─── Global HTTP Client (OPTIMIZED) ──────────────────────────────────────────
# Satu client dengan connection pooling untuk semua request ke Supabase,
# OpenRouter, Google Fonts, Groq, OpenAI — menghindari TCP handshake ulang.
_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client belum diinisialisasi")
    return _http_client

# ─── Lifespan: startup & shutdown (OPTIMIZED) ────────────────────────────────
@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global _http_client
    # Startup
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30,
        ),
        http2=False,  # HTTP/1.1 lebih reliable di free-tier proxy
    )
    _init_cascades()
    print("✅ Global HTTP client started")
    yield
    # Shutdown
    await _http_client.aclose()
    print("✅ Global HTTP client closed")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Viral Clipper Backend",
    version="3.4.0",
    lifespan=lifespan,
)

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
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")

AI_MODELS = [
    "arcee-ai/trinity-large-preview:free",
]

# ─── Supabase / Auth Config ───────────────────────────────────────────────────
SUPABASE_URL              = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
JWT_SECRET                = os.getenv("JWT_SECRET", "CHANGE_ME_USE_RANDOM_SECRET_STRING")
JWT_ALGORITHM             = "HS256"
JWT_EXPIRE_HOURS          = int(os.getenv("JWT_EXPIRE_HOURS", "168"))  # 7 hari

# ─── Temp dir ─────────────────────────────────────────────────────────────────
TEMP_DIR = Path(tempfile.gettempdir()) / "clipper-ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FONT_CACHE_DIR = TEMP_DIR / "fonts"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FONT_PATH_CACHE: dict[str, str] = {}

# ══════════════════════════════════════════════════════════════════════════════
# AUTH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _safe_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()

# OPTIMIZED: bcrypt dijalankan di thread pool agar tidak memblok event loop
# bcrypt.hash/verify memakan ~200–400ms sinkron — sangat merusak throughput FastAPI
async def hash_password(plain: str) -> str:
    safe = _safe_password(plain)
    return await asyncio.to_thread(pwd_ctx.hash, safe)

async def verify_password(plain: str, hashed: str) -> bool:
    safe = _safe_password(plain)
    return await asyncio.to_thread(pwd_ctx.verify, safe, hashed)


def create_access_token(user_id: str, email: str, name: str, role: str = "user") -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub":   user_id,
        "email": email,
        "name":  name,
        "role":  role,
        "exp":   expire,
        "iat":   datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token tidak valid atau kedaluwarsa")


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Autentikasi diperlukan")
    return decode_token(credentials.credentials)

def require_superadmin(
    credentials: "Optional[HTTPAuthorizationCredentials]" = Depends(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Autentikasi diperlukan")
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "superadmin":
        raise HTTPException(status_code=403, detail="Akses superadmin diperlukan")
    return payload

# ─── Supabase REST helpers ────────────────────────────────────────────────────

def _supa_headers() -> dict:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(500, "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY belum dikonfigurasi")
    return {
        "apikey":        SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

# OPTIMIZED: semua Supabase call sekarang pakai global client (tidak buat client baru)
async def supa_find_user_by_email(email: str) -> Optional[dict]:
    url = f"{SUPABASE_URL}/rest/v1/users?email=eq.{quote(email)}&limit=1"
    r = await get_http_client().get(url, headers=_supa_headers())
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
    rows = r.json()
    return rows[0] if rows else None

async def supa_find_users_by_email(email: str) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/users?email=eq.{quote(email)}&limit=10"
    r = await get_http_client().get(url, headers=_supa_headers())
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
    return r.json() or []

async def supa_create_user(name: str, email: str, password_hash: str, role: str = "user") -> dict:
    url = f"{SUPABASE_URL}/rest/v1/users"
    r = await get_http_client().post(
        url,
        headers=_supa_headers(),
        json={"name": name, "email": email, "password_hash": password_hash, "role": role},
    )
    if not r.is_success:
        detail = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        if r.status_code in (409, 422) or "unique" in str(detail).lower():
            raise HTTPException(409, "Email sudah terdaftar")
        raise HTTPException(502, f"Supabase error: {str(detail)[:200]}")
    rows = r.json()
    return rows[0] if isinstance(rows, list) else rows

async def supa_get_user_credits(user_id: str) -> int:
    url = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=credits&limit=1"
    r = await get_http_client().get(url, headers=_supa_headers())
    if not r.is_success:
        raise HTTPException(502, "Gagal mengambil data kredit")
    rows = r.json()
    return rows[0]["credits"] if rows else 0


async def supa_deduct_credit(user_id: str) -> bool:
    """
    Atomically deduct 1 credit. Returns True if successful, False if
    insufficient credits. Uses optimistic concurrency via credits=gt.0 filter.
    """
    # 1. Read current credits
    url_read = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=credits&limit=1"
    r = await get_http_client().get(url_read, headers=_supa_headers())
    if not r.is_success:
        return False
    rows = r.json()
    if not rows or rows[0]["credits"] <= 0:
        return False

    current = rows[0]["credits"]
    # 2. Conditional update: only deduct if credits still == current (prevents race)
    url_update = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&credits=eq.{current}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r = await get_http_client().patch(url_update, headers=headers, json={"credits": current - 1})
    if not r.is_success:
        return False
    updated = r.json() if r.headers.get("content-type", "").startswith("application/json") else []
    return len(updated) > 0


async def supa_add_credits(user_id: str, amount: int) -> int:
    url_read = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=credits&limit=1"
    r = await get_http_client().get(url_read, headers=_supa_headers())
    if not r.is_success or not r.json():
        raise HTTPException(404, "User tidak ditemukan")
    new_credits = r.json()[0]["credits"] + amount
    url_update = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r = await get_http_client().patch(url_update, headers=headers, json={"credits": new_credits})
    if not r.is_success:
        raise HTTPException(502, "Gagal update kredit")
    return new_credits

async def supa_get_all_users_admin() -> list[dict]:
    url = (
        f"{SUPABASE_URL}/rest/v1/users"
        f"?select=id,name,email,credits,role,created_at,updated_at"
        f"&order=created_at.desc"
    )
    r = await get_http_client().get(url, headers=_supa_headers())
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
    return r.json() or []

async def supa_get_templates(user_id: str) -> list[dict]:
    url = (
        f"{SUPABASE_URL}/rest/v1/clip_templates"
        f"?user_id=eq.{user_id}"
        f"&order=updated_at.desc"
    )
    r = await get_http_client().get(url, headers=_supa_headers())
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
    return r.json() or []
 
 
async def supa_create_template(user_id: str, data: dict) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/clip_templates"
    payload = {"user_id": user_id, **data}
    r = await get_http_client().post(url, headers=_supa_headers(), json=payload)
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
    rows = r.json()
    return rows[0] if isinstance(rows, list) and rows else rows
 
 
async def supa_update_template(template_id: str, user_id: str, data: dict) -> dict:
    # Require user ownership via both id AND user_id in the filter
    url = (
        f"{SUPABASE_URL}/rest/v1/clip_templates"
        f"?id=eq.{template_id}&user_id=eq.{user_id}"
    )
    payload = {**data, "updated_at": "now()"}
    r = await get_http_client().patch(url, headers=_supa_headers(), json=payload)
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
    rows = r.json()
    if not rows:
        raise HTTPException(404, "Template tidak ditemukan atau bukan milik kamu")
    return rows[0] if isinstance(rows, list) else rows
 
 
async def supa_delete_template(template_id: str, user_id: str) -> None:
    url = (
        f"{SUPABASE_URL}/rest/v1/clip_templates"
        f"?id=eq.{template_id}&user_id=eq.{user_id}"
    )
    r = await get_http_client().delete(url, headers=_supa_headers())
    if not r.is_success:
        raise HTTPException(502, f"Supabase error: {r.text[:200]}")
 

# ─── Auth Pydantic schemas ────────────────────────────────────────────────────

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str
    confirm_password: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Nama minimal 2 karakter")
        return v

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password minimal 8 karakter")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Password tidak cocok")
        return v


class SignInRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.strip().lower()

class TemplateCreate(BaseModel):
    name: str
    aspect_ratio: str = "original"
    subtitle_preset_id: str = "bold-impact"
    subtitle_enabled: bool = True
    watermark_name: Optional[str] = None
    watermark_x: float = 0.88
    watermark_y: float = 0.06
    watermark_width: float = 0.18
    watermark_opacity: float = 0.85
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    speed: float = 1.0

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Nama template tidak boleh kosong")
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def valid_aspect_ratio(cls, v: str) -> str:
        valid = {"original", "9:16", "16:9", "1:1", "4:3"}
        if v not in valid:
            raise ValueError(f"Aspect ratio tidak valid: {v}")
        return v

    @field_validator("speed")
    @classmethod
    def valid_speed(cls, v: float) -> float:
        if v <= 0 or v > 4:
            raise ValueError("Speed harus antara 0 dan 4")
        return v


@app.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(require_superadmin)):
    users = await supa_get_all_users_admin()
    total_credits = sum(u.get("credits", 0) for u in users if u.get("role") != "superadmin")
    regular_users = [u for u in users if u.get("role") != "superadmin"]
    return {
        "users": users,
        "stats": {
            "total_users":   len(regular_users),
            "total_credits": total_credits,
            "total_accounts": len(users),
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/signup")
async def signup(body: "SignUpRequest"):
    existing = await supa_find_user_by_email(body.email)
    if existing:
        raise HTTPException(409, "Email sudah terdaftar")
    pw_hash = await hash_password(body.password)   # OPTIMIZED: async bcrypt
    user    = await supa_create_user(body.name, body.email, pw_hash, role="user")
    token = create_access_token(
        user_id=str(user["id"]),
        email=user["email"],
        name=user["name"],
        role=user.get("role", "user"),
    )
    return {
        "token": token,
        "user":  {
            "id":      user["id"],
            "name":    user["name"],
            "email":   user["email"],
            "role":    user.get("role", "user"),
            "credits": user.get("credits", 10),
        },
    }


@app.post("/api/auth/signin")
async def signin(body: "SignInRequest"):
    users = await supa_find_users_by_email(body.email)
    matched_user = None
    for user in users:
        # OPTIMIZED: async bcrypt verify
        if await verify_password(body.password, user.get("password_hash", "")):
            matched_user = user
            break
    if not matched_user:
        raise HTTPException(401, "Email atau password salah")
    token = create_access_token(
        user_id=str(matched_user["id"]),
        email=matched_user["email"],
        name=matched_user["name"],
        role=matched_user.get("role", "user"),
    )
    return {
        "token": token,
        "user":  {
            "id":      matched_user["id"],
            "name":    matched_user["name"],
            "email":   matched_user["email"],
            "role":    matched_user.get("role", "user"),
            "credits": matched_user.get("credits", 0),
        },
    }


@app.get("/api/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    credits = await supa_get_user_credits(current_user["sub"])
    return {
        "id":      current_user["sub"],
        "name":    current_user.get("name"),
        "email":   current_user.get("email"),
        "role":    current_user.get("role", "user"),
        "credits": credits,
    }

@app.get("/api/user/credits")
async def get_credits(current_user: dict = Depends(get_current_user)):
    credits = await supa_get_user_credits(current_user["sub"])
    return {"credits": credits, "user_id": current_user["sub"]}

@app.get("/api/templates")
async def list_templates(current_user: dict = Depends(get_current_user)):
    """Return all templates belonging to the authenticated user."""
    templates = await supa_get_templates(current_user["sub"])
    return {"templates": templates}
 
 
@app.post("/api/templates")
async def create_template(
    body: TemplateCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create a new template for the authenticated user."""
    data = body.model_dump()
    template = await supa_create_template(current_user["sub"], data)
    return template
 
 
@app.put("/api/templates/{template_id}")
async def update_template(
    template_id: str,
    body: TemplateCreate,
    current_user: dict = Depends(get_current_user),
):
    """Update an existing template (must belong to the current user)."""
    data = body.model_dump()
    template = await supa_update_template(template_id, current_user["sub"], data)
    return template
 
@app.delete("/api/templates/{template_id}")
async def delete_template(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a template (must belong to the current user)."""
    await supa_delete_template(template_id, current_user["sub"])
    return {"success": True, "id": template_id}

class AddCreditsRequest(BaseModel):
    amount: int
    note: str = ""

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Jumlah kredit harus > 0")
        if v > 100_000:
            raise ValueError("Jumlah kredit terlalu besar (maks. 100.000)")
        return v
    


@app.post("/api/admin/users/{user_id}/add-credits")
async def admin_add_user_credits(
    user_id: str,
    body: AddCreditsRequest,
    admin: dict = Depends(require_superadmin),
):
    new_balance = await supa_add_credits(user_id, body.amount)
    print(f"💰 Admin {admin['email']} added {body.amount} credits to {user_id}. "
          f"New balance: {new_balance}. Note: {body.note or '-'}")
    return {
        "success":     True,
        "user_id":     user_id,
        "added":       body.amount,
        "new_balance": new_balance,
        "note":        body.note,
    }


@app.delete("/api/admin/users/{user_id}/credits")
async def admin_set_user_credits(
    user_id: str,
    amount: int,
    admin: dict = Depends(require_superadmin),
):
    url = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r = await get_http_client().patch(url, headers=headers, json={"credits": max(0, amount)})
    if not r.is_success:
        raise HTTPException(502, "Gagal mengatur kredit")
    return {"success": True, "user_id": user_id, "credits": max(0, amount)}


# ══════════════════════════════════════════════════════════════════════════════
# KEEP-ALIVE ENDPOINT (OPTIMIZED untuk Render free plan)
# ══════════════════════════════════════════════════════════════════════════════
# Daftarkan URL ini di UptimeRobot / cron-job.org setiap 14 menit agar
# service tidak tertidur. Endpoint ini super ringan — hanya return JSON.

@app.get("/api/ping")
async def ping():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


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
        if isinstance(result, Exception):
            results[oid] = SYSTEM_FONT
        else:
            results[oid] = result

    return results


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE OVERLAY — base64 decode + save to temp
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# AI MOTION FACIAL TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def _fill_kf_gaps(raw: list[dict], detected: list[dict]) -> list[dict]:
    result = []
    for kf in raw:
        if kf["detected"]:
            result.append({"t": kf["t"], "cx": kf["cx"], "cy": kf["cy"]})
        else:
            prev_det = next((d for d in reversed(detected) if d["t"] <= kf["t"]), None)
            next_det = next((d for d in detected if d["t"] > kf["t"]), None)

            if prev_det and next_det:
                dt = next_det["t"] - prev_det["t"]
                ratio = (kf["t"] - prev_det["t"]) / dt if dt > 0 else 0.0
                cx = prev_det["cx"] + (next_det["cx"] - prev_det["cx"]) * ratio
                cy = prev_det["cy"] + (next_det["cy"] - prev_det["cy"]) * ratio
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
                dp = p1 - p0
                seg = f"({p0}+{dp}*(t-{t0})/{dt})"

            expr = f"if(lt(t\\,{t1})\\,{seg}\\,{expr})"

        t_first = round(keyframes[0]["t"], 3)
        p_first = get_px(keyframes[0])
        expr = f"if(lt(t\\,{t_first})\\,{p_first}\\,{expr})"

        return expr

    x_expr = build_axis_expr(to_px_x)
    y_expr = build_axis_expr(to_px_y)

    return f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr}"


# OPTIMIZED: fungsi ini sekarang murni sinkron agar aman dijalankan di thread pool
# (tidak ada event loop baru di dalamnya — bug asyncio.run() sudah dihapus)
def _analyze_video_motion_sync(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    aspect_ratio: str,
) -> dict:
    """
    Versi sinkron dari analyze_video_motion — dijalankan via asyncio.to_thread().
    Semua operasi OpenCV di sini adalah blocking I/O, sehingga harus di thread pool.
    """
    if not _OPENCV_AVAILABLE:
        return {
            "hasTracking": False,
            "isStatic": True,
            "keyframes": None,
            "message": "opencv-python-headless not installed on server",
            "available": False,
        }

    import cv2

    # Gunakan cascade yang sudah di-preload (OPTIMIZED)
    face_cascade = _FACE_CASCADE
    body_cascade = _BODY_CASCADE
    if face_cascade is None or body_cascade is None:
        _init_cascades()
        face_cascade = _FACE_CASCADE
        body_cascade = _BODY_CASCADE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "hasTracking": False,
            "isStatic": True,
            "keyframes": None,
            "message": "Cannot open video file",
            "available": True,
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
        if scale < 0.99:
            small = cv2.resize(frame, (int(vid_w * scale), int(vid_h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cx, cy = None, None

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

    detected     = [kf for kf in raw if kf["detected"]]
    detect_rate  = len(detected) / max(len(raw), 1)
    print(f"   Detected {len(detected)}/{len(raw)} frames ({detect_rate:.0%})")

    if len(detected) < 2 or detect_rate < 0.15:
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   [{"t": 0.0, "cx": 0.5, "cy": 0.4}],
            "cropW": crop_w, "cropH": crop_h,
            "vidW":  vid_w,  "vidH":  vid_h,
            "message": "No person detected — center crop applied",
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
            "message": "Person is stationary — static crop applied to face",
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
        "message": f"Motion tracking enabled — {len(final)} keyframes detected",
        "available": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER AUTO-SUBTITLE
# ══════════════════════════════════════════════════════════════════════════════

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
        "-ab", "128k",
        "-ar", "16000",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 100


def _whisper_available_provider() -> tuple[str, str]:
    if GROQ_API_KEY:
        return ("groq", GROQ_API_KEY)
    if OPENAI_API_KEY:
        return ("openai", OPENAI_API_KEY)
    return ("local", "")


async def call_whisper_groq(audio_path: Path, language: Optional[str] = None) -> dict:
    endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    data = {
        "model":             "whisper-large-v3-turbo",
        "response_format":   "verbose_json",
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
    words = result.get("words", [])
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

    data = {
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
    provider, _ = _whisper_available_provider()
    print(f"🎙️  STT provider: {provider}")

    if provider == "groq":
        return await call_whisper_groq(audio_path, language)
    elif provider == "openai":
        return await call_whisper_openai(audio_path, language)
    else:
        return await asyncio.to_thread(call_whisper_local, audio_path, language)


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
                resp = await get_http_client().post(
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


FONT_REFERENCE_WIDTH = 1080.0


def build_ffmpeg_filters(
    edits: dict,
    resolved_fonts: Optional[dict[str, Optional[str]]] = None,
    motion_keyframes: Optional[list[dict]] = None,
    vid_info: Optional[dict] = None,
) -> list[str]:
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
            outline_ratio = outline_width / FONT_REFERENCE_WIDTH
            border_color = hex_to_ffmpeg_color(outline_color_hex, 1.0)
            border_part = f"borderw=w*{outline_ratio:.6f}:bordercolor={border_color}:"

        shadow_enabled = t.get("shadowEnabled", True)
        shadow_part = ""
        if shadow_enabled:
            shadow_color_hex = t.get("shadowColor", "#000000")
            shadow_color     = hex_to_ffmpeg_color(shadow_color_hex, 0.85)
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
# IMAGE OVERLAY — filter_complex builder
# ══════════════════════════════════════════════════════════════════════════════

def build_filter_complex_with_images(
    base_filters: list[str],
    image_overlays: list[dict],
    img_paths: list[Path],
) -> tuple[str, str]:
    parts: list[str] = []

    if base_filters:
        parts.append(f"[0:v]{','.join(base_filters)}[vbase0]")
    else:
        parts.append("[0:v]null[vbase0]")

    current = "vbase0"

    for i, (img, img_path) in enumerate(zip(image_overlays, img_paths)):
        input_idx   = i + 1
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
        "ok":           True,
        "ffmpeg":       FFMPEG_BIN,
        "drawtext":     DRAWTEXT_OK,
        "font":         SYSTEM_FONT,
        "stt_provider": provider,
        "stt_detail":   stt_detail,
        "opencv":       _OPENCV_AVAILABLE,
        "mode":         "stream-only",
        "tmpDir":       str(TEMP_DIR),
        "ai_models":    AI_MODELS,
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


# ── Motion Analysis Endpoint ─────────────────────────────────────────────────

@app.post("/api/analyze-motion")
async def analyze_motion_endpoint(
    video:        UploadFile = File(...),
    start_time:   float      = Form(...),
    end_time:     float      = Form(...),
    aspect_ratio: str        = Form(...),
):
    if not _OPENCV_AVAILABLE:
        return {
            "hasTracking": False,
            "isStatic":    True,
            "keyframes":   None,
            "message":     "opencv-python-headless not installed — install it for motion tracking",
            "available":   False,
        }

    suffix = Path(video.filename or "source.mp4").suffix or ".mp4"
    tmp_path = TEMP_DIR / f"motion_{os.urandom(8).hex()}{suffix}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        duration = end_time - start_time
        if duration <= 0 or duration > 3600:
            raise HTTPException(400, "Invalid clip duration (must be 0–3600s)")

        print(f"🎯 Analyze motion: {aspect_ratio}, {start_time:.1f}s–{end_time:.1f}s")

        # OPTIMIZED: panggil fungsi sinkron via to_thread — tidak ada nested asyncio.run()
        result = await asyncio.to_thread(
            _analyze_video_motion_sync, tmp_path, start_time, end_time, aspect_ratio
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[analyze-motion error] {e}")
        raise HTTPException(500, f"Motion analysis failed: {str(e)[:200]}")
    finally:
        safe_delete(tmp_path)


@app.post("/api/analyze-video")
async def analyze_video(
    file_name: str   = Form(...),
    file_size: int   = Form(...),
    duration:  float = Form(...),
    mime_type: str   = Form(default="video/mp4"),
    current_user: dict = Depends(get_current_user),
):
    credits = await supa_get_user_credits(current_user["sub"])
    if credits <= 0:
        raise HTTPException(402, "Kredit tidak cukup. Silakan top up kredit kamu.")

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

    result = {
        "moments":             moments,
        "summary":             parsed.get("summary", "Analisis selesai."),
        "totalViralPotential": parsed.get("totalViralPotential", 5),
    }

    deducted = await supa_deduct_credit(current_user["sub"])
    remaining = await supa_get_user_credits(current_user["sub"])
    result["credits_remaining"] = remaining
    if deducted:
        print(f"💳 Credit deducted: user={current_user['email']} | analyze-video | remaining={remaining}")

    return result


@app.post("/api/generate-clip-content")
async def generate_clip_content(
    moment_label:    str   = Form(...),
    moment_category: str   = Form(...),
    moment_reason:   str   = Form(...),
    start_time:      float = Form(...),
    end_time:        float = Form(...),
    video_file_name: str   = Form(...),
      current_user: dict     = Depends(get_current_user),   # ← tambah ini
):
    # ── Credit check ──────────────────────────────────────────────────────
    credits = await supa_get_user_credits(current_user["sub"])
    if credits <= 0:
        raise HTTPException(402, "Kredit tidak cukup. Silakan top up kredit kamu.")

    def fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"

    prompt = f"""..."""  # tidak berubah

    content = await call_openrouter(...)  # tidak berubah

    cleaned = re.sub(r"```json\s*|```\s*", "", content).strip()
    result = json.loads(cleaned)

    # ── Deduct kredit SETELAH AI berhasil ─────────────────────────────────
    deducted = await supa_deduct_credit(current_user["sub"])
    remaining = await supa_get_user_credits(current_user["sub"])
    if deducted:
        print(f"💳 Credit deducted: user={current_user['email']} | generate-clip-content | remaining={remaining}")

    result["credits_remaining"] = remaining
    return result


@app.post("/api/auto-subtitle")
async def auto_subtitle(
    video:      FAUploadFile = FAFile(...),
    start_time: float        = Form(...),
    end_time:   float        = Form(...),
    words_per_chunk: int     = Form(default=3),
    language:   str          = Form(default=""),
    current_user: dict       = Depends(get_current_user),
):
    credits = await supa_get_user_credits(current_user["sub"])
    if credits <= 0:
        raise HTTPException(402, "Kredit tidak cukup. Silakan top up kredit kamu.")

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

        print(f"🎙️  Auto-subtitle: provider={provider} user={current_user['email']} "
              f"start={start_time:.1f}s dur={duration:.1f}s words_per_chunk={words_per_chunk}")

        ok = await extract_audio_segment(upload_path, start_time, duration, audio_path)
        if not ok:
            raise HTTPException(500, "Failed to extract audio from video")

        result = await call_whisper_api(audio_path, language=language or None)
        words  = result.get("words", [])
        print(f"   ✅ Transcribed {len(words)} words")
        chunks = group_words_to_subtitles(words, words_per_chunk=words_per_chunk)

        deducted = await supa_deduct_credit(current_user["sub"])
        remaining = await supa_get_user_credits(current_user["sub"])
        if deducted:
            print(f"💳 Credit deducted: user={current_user['email']} | auto-subtitle | remaining={remaining}")

        return {
            "ok":               True,
            "chunks":           chunks,
            "language":         result.get("language", ""),
            "full_text":        result.get("text", ""),
            "word_count":       len(words),
            "credits_remaining": remaining,
        }
    finally:
        safe_delete(upload_path)
        safe_delete(audio_path)


@app.post("/api/export-clip")
async def export_clip(
    video:     UploadFile = File(...),
    clipJson:  str        = Form(...),
    editsJson: str        = Form(...),
    current_user: dict    = Depends(get_current_user), 
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

        motion_keyframes: Optional[list[dict]] = edits.get("motionKeyframes") or None
        motion_vid_w: Optional[int] = edits.get("motionVidW")
        motion_vid_h: Optional[int] = edits.get("motionVidH")

        vid_info: Optional[dict] = None
        if motion_keyframes and motion_vid_w and motion_vid_h:
            vid_info = {"w": int(motion_vid_w), "h": int(motion_vid_h)}
            print(f"🎯 Export with motion tracking: {len(motion_keyframes)} kf, "
                  f"vid={motion_vid_w}x{motion_vid_h}")

        resolved_fonts: dict[str, Optional[str]] = {}
        if DRAWTEXT_OK and text_overlays:
            print(f"🔤 Resolving fonts for {len(text_overlays)} text overlays…")
            resolved_fonts = await resolve_fonts_for_overlays(text_overlays)

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
        )

        print(
            f"🎬 Export start={start_sec:.1f}s dur={duration_sec:.1f}s "
            f"text={len(text_overlays)} images={len(valid_image_overlays)} "
            f"motion={'yes' if motion_keyframes else 'no'}"
        )

        args = [
            FFMPEG_BIN, "-y",
            "-ss", seconds_to_ffmpeg(start_sec),
            "-t",  seconds_to_ffmpeg(duration_sec),
            "-i",  str(upload_path),
        ]

        if has_image_overlays:
            for img_path in img_temp_files:
                args += [
                    "-loop", "1",
                    "-t",   seconds_to_ffmpeg(duration_sec),
                    "-i",   str(img_path),
                ]

            filter_complex, final_label = build_filter_complex_with_images(
                base_filters,
                valid_image_overlays,
                img_temp_files,
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

        use_buffered = has_image_overlays or has_text_overlays or bool(motion_keyframes)
        file_name    = f"clip_{int(start_sec)}_{int(start_sec + duration_sec)}.mp4"

        if use_buffered:
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
                err = stderr_data.decode("utf-8", errors="replace")
                print(f"[ffmpeg error]\n{err[-800:]}")
                safe_delete(upload_path)

                hint = ""
                if "No such file or directory" in err and "font" in err.lower():
                    hint = " (font file not found)"
                elif "drawtext" in err.lower():
                    hint = " (drawtext filter error)"
                elif "scale2ref" in err.lower() or "overlay" in err.lower():
                    hint = " (image overlay filter error)"
                elif "crop" in err.lower():
                    hint = " (motion tracking crop error)"
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


# ─── Run (dev) ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3001)),
        reload=True,
    )