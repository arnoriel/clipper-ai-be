"""
main.py — Entry point aplikasi AI Viral Clipper.

Tanggung jawab file ini dibatasi pada:
  1. Inisialisasi FastAPI app
  2. Lifespan (startup / shutdown global resources)
  3. CORS middleware
  4. Include routers
  5. uvicorn runner (dev)

Semua business logic ada di services/ dan routers/.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from dependencies import init_cascades, shutdown_http_client, startup_http_client

# ─── Routers ──────────────────────────────────────────────────────────────────
from routers.admin import router as admin_router
from routers.auth import router as auth_router
from routers.video import router as video_router


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_http_client()
    init_cascades()
    yield
    # Shutdown
    await shutdown_http_client()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Viral Clipper Backend",
    version="4.0.0",
    lifespan=lifespan,
)

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
