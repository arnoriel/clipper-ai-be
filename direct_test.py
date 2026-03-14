import asyncio
import json
from fastapi import UploadFile  # type: ignore
from io import BytesIO

# Dummy file to mimic FastAPI UploadFile
class DummyFile:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")

async def run():
    from main import export_clip  # type: ignore
    try:
        vid = DummyFile("test.mp4")
        clip = '{"startTime": 0, "endTime": 5}'
        edits = '{"trimStart":0, "trimEnd":0, "textOverlays": [{"id":"1","text":"test","startSec":0,"endSec":5}]}'
        await export_clip(video=vid, clipJson=clip, editsJson=edits)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(run())
