"""
services/ai.py — OpenRouter AI call helper dengan retry logic.
"""

import asyncio

import httpx
from fastapi import HTTPException

from config import AI_MODELS, OPENROUTER_API_KEY, OPENROUTER_BASE
from dependencies import get_http_client


async def call_openrouter(
    messages: list,
    max_tokens: int = 3000,
    temperature: float = 0.3,
) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(400, "OPENROUTER_API_KEY not configured on server")

    last_error: Exception = RuntimeError("No models tried")

    for model in AI_MODELS:
        for attempt in range(2):
            try:
                print(f"🤖 model={model} attempt={attempt + 1}")
                resp = await get_http_client().post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://viral-clipper-ai.vercel.app",
                        "X-Title":       "AI Viral Clipper",
                    },
                    json={
                        "model":       model,
                        "messages":    messages,
                        "max_tokens":  max_tokens,
                        "temperature": temperature,
                    },
                )

                if resp.status_code == 429:
                    await asyncio.sleep(1)
                    break

                if resp.status_code in (502, 503, 504):
                    await asyncio.sleep(2)
                    continue

                if not resp.is_success:
                    raise HTTPException(502, f"OpenRouter {resp.status_code}: {resp.text[:200]}")

                content = resp.json()["choices"][0]["message"]["content"]
                return content

            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                wait = 2 if attempt == 0 else 4
                await asyncio.sleep(wait)
                continue

            except httpx.TimeoutException as e:
                last_error = e
                break

            except HTTPException:
                raise

            except Exception as e:
                last_error = e
                break

    raise HTTPException(
        502,
        f"Semua model AI gagal. Error: {type(last_error).__name__}: {str(last_error)[:200]}",
    )
