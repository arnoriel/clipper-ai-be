"""
services/supabase.py — Semua Supabase REST helpers.
Semua request pakai global HTTP client (connection pooling).
"""

from typing import Optional
from urllib.parse import quote

from fastapi import HTTPException

from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL
from dependencies import get_http_client


# ─── Header helper ────────────────────────────────────────────────────────────

def _supa_headers() -> dict:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            500, "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY belum dikonfigurasi"
        )
    return {
        "apikey":        SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


# ─── User queries ─────────────────────────────────────────────────────────────

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


async def supa_create_user(
    name: str, email: str, password_hash: str, role: str = "user"
) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/users"
    r = await get_http_client().post(
        url,
        headers=_supa_headers(),
        json={"name": name, "email": email, "password_hash": password_hash, "role": role},
    )
    if not r.is_success:
        detail = (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text
        )
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
    Atomically deduct 1 credit. Returns True jika berhasil,
    False jika kredit tidak cukup. Pakai optimistic concurrency.
    """
    # 1. Baca kredit saat ini
    url_read = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=credits&limit=1"
    r = await get_http_client().get(url_read, headers=_supa_headers())
    if not r.is_success:
        return False
    rows = r.json()
    if not rows or rows[0]["credits"] <= 0:
        return False

    current = rows[0]["credits"]

    # 2. Conditional update: hanya kurangi kalau credits masih == current
    url_update = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&credits=eq.{current}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r = await get_http_client().patch(
        url_update, headers=headers, json={"credits": current - 1}
    )
    if not r.is_success:
        return False
    updated = (
        r.json()
        if r.headers.get("content-type", "").startswith("application/json")
        else []
    )
    return len(updated) > 0


async def supa_add_credits(user_id: str, amount: int) -> int:
    url_read = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=credits&limit=1"
    r = await get_http_client().get(url_read, headers=_supa_headers())
    if not r.is_success or not r.json():
        raise HTTPException(404, "User tidak ditemukan")
    new_credits = r.json()[0]["credits"] + amount
    url_update = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r = await get_http_client().patch(
        url_update, headers=headers, json={"credits": new_credits}
    )
    if not r.is_success:
        raise HTTPException(502, "Gagal update kredit")
    return new_credits


async def supa_set_credits(user_id: str, amount: int) -> bool:
    """Set kredit user ke nilai tertentu (dipakai oleh superadmin)."""
    url = f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
    headers = {**_supa_headers(), "Prefer": "return=representation"}
    r = await get_http_client().patch(url, headers=headers, json={"credits": amount})
    if not r.is_success:
        raise HTTPException(502, "Gagal mengatur kredit")
    rows = r.json()
    if not rows:
        raise HTTPException(404, "User tidak ditemukan")
    return True


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


# ─── Template queries ─────────────────────────────────────────────────────────

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
    # Ownership check via filter ganda: id + user_id
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
