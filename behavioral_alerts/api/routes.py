from fastapi import APIRouter
from db.mongo import get_collection
from core.anomalies import *
from core.profiling import build_user_profile
from utils.store import store_capture
from utils.alert import trigger_alert
from models.schemas import CaptureRequest
from datetime import datetime
import pandas as pd

router = APIRouter()

@router.post("/capture")
def capture(data: CaptureRequest):
    now = datetime.now()
    data_dict = data.dict()
    data_dict.update({
        "timestamp": now,
        "hour": now.hour,
        "weekday": now.weekday(),
        "month": now.month
    })
    store_capture(data.user_id, data_dict)
    loc_anomaly, time_anomaly = detect_anomalies(data.user_id, data.latitude, data.longitude, now)
    if loc_anomaly > 0.5 or time_anomaly > 0.5 or data.emergency:
        trigger_alert(data.user_id, data.latitude, data.longitude, loc_anomaly, time_anomaly)
    return {
        "status": "success",
        "loc_anomaly": loc_anomaly,
        "time_anomaly": time_anomaly
    }
