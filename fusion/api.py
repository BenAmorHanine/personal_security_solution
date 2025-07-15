from fastapi import FastAPI
from pydantic import BaseModel
from fusion.process_capture import process_capture
from unittest.mock import MagicMock
import fusion.process_capture
app = FastAPI()

class CaptureInput(BaseModel):
    user_id: str
    lat: float
    lon: float

@app.post("/process_capture")
async def process_capture_endpoint(input: CaptureInput):
    ts_collection = MagicMock()
    geo_collection = MagicMock()
    users_collection = MagicMock()
    ts_collection.find.return_value = [{"hour": 12}]
    result = process_capture(input.user_id, input.lat, input.lon, ts_collection, geo_collection, users_collection)
    return result