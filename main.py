"""
main.py — Entry point aplikasi AI Viral Clipper.

Tanggung jawab file ini dibatasi pada:
  1. Inisialisasi FastAPI app
  2. Lifespan (startup / shutdown global resources)
  3. CORS middleware
  4. Rate limiting middleware (slowapi)
  5. Include routers
  6. uvicorn runner (dev)
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import ALLOWED_ORIGINS, TEMP_DIR
from dependencies import init_cascades, shutdown_http_client, startup_http_client

# ─── Routers ──────────────────────────────────────────────────────────────────
from routers.admin import router as admin_router
from routers.auth import router as auth_router
from routers.video import router as video_router


# ─── Rate limiter ─────────────────────────────────────────────────────────────
# Key: IP address — mencegah 1 user flood semua worker slot
limiter = Limiter(key_func=get_remote_address)


# ─── Background temp cleanup ──────────────────────────────────────────────────

async def _cleanup_old_temps() -> None:
    """
    Hapus file temp yang lebih dari 10 menit.
    Berjalan setiap 5 menit di background.

    Tanpa ini, worker crash di tengah FFmpeg meninggalkan file upload
    (bisa 100 MB+) yang akumulasi sampai disk Render penuh.
    """
    while True:
        await asyncio.sleep(300)  # setiap 5 menit
        try:
            now = time.time()
            deleted = 0
            for f in TEMP_DIR.glob("*"):
                try:
                    if f.is_file() and now - f.stat().st_mtime > 600:  # > 10 menit
                        f.unlink(missing_ok=True)
                        deleted += 1
                except OSError:
                    pass  # file sudah dihapus / race condition, skip
            if deleted:
                print(f"🧹 Temp cleanup: {deleted} file(s) removed from {TEMP_DIR}")
        except Exception as e:
            print(f"[cleanup error] {e}")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_http_client()
    init_cascades()
    asyncio.create_task(_cleanup_old_temps())
    print("🧹 Temp cleanup task started (interval: 5 min, ttl: 10 min)")
    yield
    # Shutdown
    await shutdown_http_client()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Viral Clipper Backend",
    version="4.1.0",
    lifespan=lifespan,
)

# Pasang rate limiter state & error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-File-Name"],
)

# ─── Register routers ─────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(video_router)


# ─── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3001)),
        reload=True,
    )