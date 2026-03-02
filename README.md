# AI Viral Clipper — Python Backend

Backend baru berbasis **FastAPI + Python** menggantikan Express.js.  
AI (OpenRouter) sekarang dipanggil dari **server**, bukan dari browser — API key aman.

## Stack
- **FastAPI** — web framework async
- **uvicorn** — ASGI server
- **httpx** — async HTTP client (untuk OpenRouter)
- **ffmpeg** — video processing (wajib tersedia di sistem)
- **python-multipart** — handle file upload

## Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/api/health` | Status server |
| POST | `/api/get-video-duration` | ffprobe durasi video |
| POST | `/api/analyze-video` | **AI analisis momen viral** (server-side) |
| POST | `/api/generate-clip-content` | AI generate judul/caption/hashtag |
| POST | `/api/export-clip` | ffmpeg clip + stream ke browser |

## Setup Lokal

### 1. Install dependencies

```bash
cd clipper-ai-be
pip install -r requirements.txt
```

### 2. Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**  
Download dari https://ffmpeg.org/download.html dan tambahkan ke PATH.

### 3. Buat file .env

```bash
cp .env.example .env
```

Isi `OPENROUTER_API_KEY` dengan API key dari https://openrouter.ai/

### 4. Jalankan server

```bash
python main.py
# atau
uvicorn main:app --reload --port 3001
```

Server berjalan di http://localhost:3001

---

## Deploy ke Railway (Gratis)

Railway otomatis install ffmpeg via Nixpacks.

1. Push repo ke GitHub
2. Buka https://railway.app → New Project → Deploy from GitHub
3. Tambahkan environment variable:
   - `OPENROUTER_API_KEY` = api key kamu
   - `ALLOWED_ORIGINS` = URL frontend kamu (misal `https://app-kamu.vercel.app`)
4. Railway akan otomatis detect `railway.toml` dan install ffmpeg

---

## Deploy ke Render (Gratis)

Render sudah punya ffmpeg tersedia secara default.

1. Push repo ke GitHub  
2. Buka https://render.com → New Web Service → Connect repo
3. Render otomatis detect `render.yaml`
4. Tambahkan environment variable `OPENROUTER_API_KEY`
5. Deploy!

---

## Update Frontend

Setelah deploy backend, update `.env.local` frontend:

```env
# Hapus ini (tidak dipakai lagi):
# VITE_OPENROUTER_API_KEY=sk-or-xxxxx

# Update ini dengan URL backend baru:
VITE_API_BASE=https://your-backend.railway.app
```

### File frontend yang perlu diganti:

1. **`src/lib/AI.ts`** → Ganti dengan `AI.ts.updated` dari folder ini
2. **`src/lib/storage.ts`** → Ganti dengan `storage.ts.updated` dari folder ini

Perubahan utama:
- `AI.ts`: Semua panggilan AI sekarang ke `/api/analyze-video` di backend (bukan langsung ke OpenRouter)
- `storage.ts`: `getApiKey()` return `"server-side"`, `isApiKeyConfigured()` selalu `true`
- Frontend tidak perlu `VITE_OPENROUTER_API_KEY` lagi

---

## Perbedaan dari Express.js

| | Express.js | FastAPI Python |
|---|---|---|
| Install ffmpeg | Manual, repot | Railway/Render auto install |
| AI API key | Di browser (tidak aman) | Di server (aman) |
| Async | Callback hell | `async/await` native |
| Deploy gratis | Butuh workaround | Railway/Render langsung support |
| ffmpeg streaming | `ffmpeg.stdout.pipe(res)` | `asyncio.create_subprocess_exec` + `StreamingResponse` |
| File upload | multer | `python-multipart` bawaan FastAPI |
