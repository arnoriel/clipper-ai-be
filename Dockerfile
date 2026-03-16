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
COPY main.py .

# Pre-create direktori temp & font cache agar tidak perlu dibuat saat startup
RUN mkdir -p /tmp/clipper-ai/fonts

# Expose port (Render inject $PORT otomatis)
EXPOSE 3001

# Gunakan sh -c agar $PORT bisa di-expand dari environment
# --workers 1: Render free plan punya RAM terbatas (~512MB), 1 worker sudah cukup
# --timeout-keep-alive 65: sedikit lebih lama dari default 5s agar koneksi ke Supabase tidak drop
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-3001} --workers 1 --timeout-keep-alive 65"]