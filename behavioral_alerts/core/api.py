from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from pymongo import MongoClient
from .capture import process_capture
from .config import MONGO_URI

app = FastAPI()
client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
geo_collection = db["geo_data"]
users_collection = db["users"]

class SOSRequest(BaseModel):
    user_id: str
    device_id: str
    latitude: float
    longitude: float


"""
still lacks the login, signup, and user management functionalities
I only implemented the sos, periodic check functionalities, and register user
"""

@app.post("/sos")
async def trigger_sos(request: SOSRequest):
    try:
        result = process_capture(
            request.user_id, request.latitude, request.longitude, sos_pressed=True,
            ts_collection=locations_collection,
            geo_collection=geo_collection,
            users_collection=users_collection
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process SOS")
        return {
            "status": "Alert processed",
            "alert_id": result["alert_id"],
            "incident_probability": result["incident_probability"],
            "anomaly_flag": result["anomaly_flag"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/periodic-check/{user_id}/{device_id}")
async def periodic_check(user_id: str, device_id: str):
    try:
        recent_location = locations_collection.find_one(
            {"user_id": user_id}, sort=[("timestamp", -1)]
        )
        if not recent_location:
            raise HTTPException(status_code=404, detail="No recent location")
        result = process_capture(
            user_id, recent_location["latitude"], recent_location["longitude"], sos_pressed=False,
            ts_collection=locations_collection,
            geo_collection=geo_collection,
            users_collection=users_collection
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process periodic check")
        return {
            "status": "Periodic check completed",
            "incident_probability": result["incident_probability"],
            "anomaly_flag": result["anomaly_flag"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class RegisterRequest(BaseModel):
    display_id: str
    device_name: str
@app.post("/register")
async def register_user(request: RegisterRequest):
    try:
        user_id = str(request.uuid())
        device_id = str(request.uuid())
        users_collection.insert_one({
            "user_id": user_id,
            "display_id": request.display_id,
            "device_id": device_id,
            "device_name": request.device_name,
            "profile": {},
            "created_at": datetime.now(timezone.utc)
        })
        return {"user_id": user_id, "device_id": device_id, "display_id": request.display_id, "device_name": request.device_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")