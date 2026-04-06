"""
routers/admin.py — Superadmin-only routes.

Endpoints:
  GET    /api/admin/users
  POST   /api/admin/users/{user_id}/add-credits
  PATCH  /api/admin/users/{user_id}/set-credits
  DELETE /api/admin/users/{user_id}/credits   (legacy — set credits via query param)
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from config import SUPABASE_URL
from dependencies import get_http_client
from services.auth_utils import require_superadmin
from services.supabase import _supa_headers, supa_add_credits, supa_get_all_users_admin

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

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


class SetCreditsRequest(BaseModel):
    amount: int

    @field_validator("amount")
    @classmethod
    def amount_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Jumlah kredit tidak boleh negatif")
        if v > 1_000_000:
            raise ValueError("Jumlah kredit terlalu besar (maks. 1.000.000)")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(require_superadmin)):
    users         = await supa_get_all_users_admin()
    regular_users = [u for u in users if u.get("role") != "superadmin"]
    total_credits = sum(u.get("credits", 0) for u in regular_users)
    return {
        "users": users,
        "stats": {
            "total_users":    len(regular_users),
            "total_credits":  total_credits,
            "total_accounts": len(users),
        },
    }


@router.post("/api/admin/users/{user_id}/add-credits")
async def admin_add_user_credits(
    user_id: str,
    body: AddCreditsRequest,
    admin: dict = Depends(require_superadmin),
):
    new_balance = await supa_add_credits(user_id, body.amount)
    print(
        f"💰 Admin {admin['email']} added {body.amount} credits to {user_id}. "
        f"New balance: {new_balance}. Note: {body.note or '-'}"
    )
    return {
        "success":     True,
        "user_id":     user_id,
        "added":       body.amount,
        "new_balance": new_balance,
        "note":        body.note,
    }


@router.patch("/api/admin/users/{user_id}/set-credits")
async def admin_set_user_credits_patch(
    user_id: str,
    body: SetCreditsRequest,
    admin: dict = Depends(require_superadmin),
):
    url     = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r       = await get_http_client().patch(url, headers=headers, json={"credits": body.amount})

    if not r.is_success:
        raise HTTPException(502, "Gagal mengatur kredit")
    rows = r.json()
    if not rows:
        raise HTTPException(404, "User tidak ditemukan")

    print(f"✏️  Admin {admin['email']} SET credits of {user_id} → {body.amount}")
    return {"success": True, "user_id": user_id, "new_balance": body.amount}


@router.delete("/api/admin/users/{user_id}/credits")
async def admin_set_user_credits_legacy(
    user_id: str,
    amount: int,
    admin: dict = Depends(require_superadmin),
):
    """Legacy endpoint — set credits via query param (kept for backward compatibility)."""
    url     = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r       = await get_http_client().patch(
        url, headers=headers, json={"credits": max(0, amount)}
    )

    if not r.is_success:
        raise HTTPException(502, "Gagal mengatur kredit")
    return {"success": True, "user_id": user_id, "credits": max(0, amount)}
