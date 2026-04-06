"""
routers/auth.py — Auth routes, user profile, dan template CRUD.

Endpoints:
  POST /api/auth/signup
  POST /api/auth/signin
  GET  /api/auth/me
  GET  /api/user/credits
  GET  /api/templates
  POST /api/templates
  PUT  /api/templates/{template_id}
  DEL  /api/templates/{template_id}
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from services.auth_utils import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from services.supabase import (
    supa_create_template,
    supa_create_user,
    supa_delete_template,
    supa_find_user_by_email,
    supa_find_users_by_email,
    supa_get_templates,
    supa_get_user_credits,
    supa_update_template,
)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Auth routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/api/auth/signup")
async def signup(body: SignUpRequest):
    existing = await supa_find_user_by_email(body.email)
    if existing:
        raise HTTPException(409, "Email sudah terdaftar")

    pw_hash = await hash_password(body.password)
    user    = await supa_create_user(body.name, body.email, pw_hash, role="user")

    token = create_access_token(
        user_id=str(user["id"]),
        email=user["email"],
        name=user["name"],
        role=user.get("role", "user"),
    )
    return {
        "token": token,
        "user": {
            "id":      user["id"],
            "name":    user["name"],
            "email":   user["email"],
            "role":    user.get("role", "user"),
            "credits": user.get("credits", 10),
        },
    }


@router.post("/api/auth/signin")
async def signin(body: SignInRequest):
    users = await supa_find_users_by_email(body.email)
    matched_user = None
    for user in users:
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
        "user": {
            "id":      matched_user["id"],
            "name":    matched_user["name"],
            "email":   matched_user["email"],
            "role":    matched_user.get("role", "user"),
            "credits": matched_user.get("credits", 0),
        },
    }


@router.get("/api/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    credits = await supa_get_user_credits(current_user["sub"])
    return {
        "id":      current_user["sub"],
        "name":    current_user.get("name"),
        "email":   current_user.get("email"),
        "role":    current_user.get("role", "user"),
        "credits": credits,
    }


@router.get("/api/user/credits")
async def get_credits(current_user: dict = Depends(get_current_user)):
    credits = await supa_get_user_credits(current_user["sub"])
    return {"credits": credits, "user_id": current_user["sub"]}


# ──────────────────────────────────────────────────────────────────────────────
# Template CRUD routes
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/api/templates")
async def list_templates(current_user: dict = Depends(get_current_user)):
    """Return semua template milik user yang sedang login."""
    templates = await supa_get_templates(current_user["sub"])
    return {"templates": templates}


@router.post("/api/templates")
async def create_template(
    body: TemplateCreate,
    current_user: dict = Depends(get_current_user),
):
    """Buat template baru untuk user yang sedang login."""
    data     = body.model_dump()
    template = await supa_create_template(current_user["sub"], data)
    return template


@router.put("/api/templates/{template_id}")
async def update_template(
    template_id: str,
    body: TemplateCreate,
    current_user: dict = Depends(get_current_user),
):
    """Update template (harus milik user yang sedang login)."""
    data     = body.model_dump()
    template = await supa_update_template(template_id, current_user["sub"], data)
    return template


@router.delete("/api/templates/{template_id}")
async def delete_template(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Hapus template (harus milik user yang sedang login)."""
    await supa_delete_template(template_id, current_user["sub"])
    return {"success": True, "id": template_id}
