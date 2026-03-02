FROM python:3.11-slim

# Install ffmpeg dengan libfreetype (drawtext support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Verifikasi drawtext tersedia
RUN ffmpeg -filters 2>/dev/null | grep drawtext || echo "⚠️ drawtext tidak ditemukan!"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 3001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]