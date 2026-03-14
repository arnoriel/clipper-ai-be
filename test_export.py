import httpx  # type: ignore
import asyncio

async def test_export():
    async with httpx.AsyncClient() as client:
        with open("test.mp4", "rb") as f:
            resp = await client.post(
                "http://127.0.0.1:3001/api/export-clip",
                data={
                    "clipJson": '{"startTime": 0, "endTime": 5}',
                    "editsJson": '{"trimStart":0, "trimEnd":0, "textOverlays": [{"id":"1","text":"test","startSec":0,"endSec":5}]}'
                },
                files={"video": ("test.mp4", f, "video/mp4")},
                timeout=120
            )
        print("Status:", resp.status_code)
        if resp.status_code != 200:
            print("Text:", resp.text)
        else:
            print(f"Success! Received {len(resp.content)} bytes.")

asyncio.run(test_export())
