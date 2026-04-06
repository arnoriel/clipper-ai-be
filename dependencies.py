"""
dependencies.py — Global singletons yang di-share ke seluruh app:
  - httpx.AsyncClient (connection pooling)
  - OpenCV CascadeClassifier (preloaded, bukan tiap request)

Di-init saat lifespan startup di main.py.
"""

from typing import Optional
import httpx

# ─── OpenCV availability check ────────────────────────────────────────────────
try:
    import cv2 as _cv2
    OPENCV_AVAILABLE = True
    print(f"✅ OpenCV available: {_cv2.__version__}")
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  OpenCV not available. Install: pip install opencv-python-headless")

# ─── OpenCV cascades (preloaded saat startup) ─────────────────────────────────
FACE_CASCADE: Optional[object] = None
BODY_CASCADE: Optional[object] = None


def init_cascades() -> None:
    """Panggil sekali saat startup — lazy-init agar tidak crash di cold start."""
    global FACE_CASCADE, BODY_CASCADE
    if not OPENCV_AVAILABLE or FACE_CASCADE is not None:
        return
    import cv2
    FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    BODY_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_upperbody.xml"
    )
    print("✅ OpenCV cascades pre-loaded")


# ─── Global HTTP client ───────────────────────────────────────────────────────
_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client belum diinisialisasi — pastikan lifespan berjalan")
    return _http_client


async def startup_http_client() -> None:
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30,
        ),
        http2=False,  # HTTP/1.1 lebih reliable di free-tier
    )
    print("✅ Global HTTP client started")


async def shutdown_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        print("✅ Global HTTP client closed")
