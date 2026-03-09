FROM python:3.11-slim

# Install ffmpeg + freetype (wajib untuk drawtext) + fontconfig (fc-list fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libfreetype6 \
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Verifikasi drawtext tersedia (build akan gagal kalau tidak ada)
RUN ffmpeg -filters 2>/dev/null | grep -q drawtext \
    && echo "✅ drawtext OK" \
    || (echo "❌ drawtext NOT FOUND — build failed" && exit 1)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render inject $PORT otomatis, fallback 3001 untuk lokal
EXPOSE 3001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-3001} --workers 1"]