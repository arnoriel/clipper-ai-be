FROM python:3.11-slim

# Install system dependencies dalam satu layer untuk minimize image size
# fonts-dejavu-core & fonts-liberation sudah termasuk TTF yang langsung bisa dipakai ffmpeg drawtext
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libfreetype6 \
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    # curl untuk health check internal (optional)
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Verifikasi drawtext tersedia (build gagal kalau tidak ada)
RUN ffmpeg -filters 2>/dev/null | grep -q drawtext \
    && echo "✅ drawtext OK" \
    || (echo "❌ drawtext NOT FOUND — build failed" && exit 1)

WORKDIR /app

# Copy requirements dulu (layer ini di-cache selama requirements.txt tidak berubah)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: hemat disk
# opencv-python-headless di-install terakhir karena paling besar & jarang berubah
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (layer ini sering berubah, taruh paling akhir)
COPY . .

# Pre-create direktori temp & font cache agar tidak perlu dibuat saat startup
RUN mkdir -p /tmp/clipper-ai/fonts

# Expose port (Render inject $PORT otomatis)
EXPOSE 3001

# Gunicorn + UvicornWorker — optimal untuk Render 8 GB / 6 core
# --workers 4        : 4 process paralel, masing-masing handle request sendiri
#                      → FFmpeg encode 1 user tidak block user lain
# --worker-connections 50 : max concurrent connections per worker
# --timeout 120      : FFmpeg long encode bisa 60-90s, beri slack cukup
# --keep-alive 65    : sedikit lebih lama dari default agar koneksi Supabase tidak drop
# --graceful-timeout 30 : kasih waktu worker selesaikan request sebelum di-restart
CMD ["sh", "-c", "gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --worker-connections 50 \
  --bind 0.0.0.0:${PORT:-3001} \
  --timeout 120 \
  --keep-alive 65 \
  --graceful-timeout 30 \
  --log-level info"]