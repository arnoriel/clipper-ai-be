"""
services/auth_utils.py — JWT helpers, bcrypt, dan FastAPI dependency injections.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, JWT_SECRET

# ─── Password utils ───────────────────────────────────────────────────────────

def _safe_password(plain: str) -> bytes:
    """SHA-256 → 32 bytes binary, jauh di bawah limit 72 bytes bcrypt."""
    return hashlib.sha256(plain.encode("utf-8")).digest()


async def hash_password(plain: str) -> str:
    safe = _safe_password(plain)
    hashed = await asyncio.to_thread(_bcrypt.hashpw, safe, _bcrypt.gensalt(12))
    return hashed.decode("utf-8")


async def verify_password(plain: str, hashed: str) -> bool:
    safe = _safe_password(plain)
    try:
        return await asyncio.to_thread(_bcrypt.checkpw, safe, hashed.encode("utf-8"))
    except Exception:
        return False


# ─── JWT utils ────────────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    email: str,
    name: str,
    role: str = "user",
) -> str:
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


# ─── FastAPI dependency injections ───────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Autentikasi diperlukan")
    return decode_token(credentials.credentials)


def require_superadmin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Autentikasi diperlukan")
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "superadmin":
        raise HTTPException(status_code=403, detail="Akses superadmin diperlukan")
    return payload
