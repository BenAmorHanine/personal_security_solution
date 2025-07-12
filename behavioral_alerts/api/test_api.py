from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient
from behavioral_alerts.core.profiling import should_retrain, build_user_profile, detect_user_anomalies
from behavioral_alerts.core.threshold_adjustment import prepare_threshold_data
from behavioral_alerts.core.incident_prediction import prepare_incident_data
from behavioral_alerts.core.train_models import retrain_all_models_for_user
from behavioral_alerts.core.scoring import evaluate_user_behavior
from behavioral_alerts.core.utils import setup_timeseries_collection, setup_users_collection

app = FastAPI()
#all of this will be updated bcz we will merge the other models trained earlier.
# ---------- 1) Startup: connect to MongoDB ----------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "hydatis"

client            = MongoClient(MONGO_URI)
db                = client[DB_NAME]
ts_collection     = setup_timeseries_collection()   # time-series GPS points
users_collection  = setup_users_collection()        # one doc per user

# ---------- 2) Pydantic model for incoming request ----------
class CapturePayload(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    emergency: bool = False      # user pressed SOS?
    timestamp: datetime = None   # default: now()

    def dict_with_time(self):
        d = self.dict()
        d["timestamp"] = d["timestamp"] or datetime.utcnow()
        return d

# ---------- 3) Placeholder trigger_alert function ----------
def trigger_alert(user_id: str, lat: float, lon: float, loc_score: float, time_score: float):
    # In production, integrate here with SMS provider, email, push notification,
    # or call upstream emergency‑services API.
    print(f"🚨 ALERT for {user_id} @ ({lat:.5f},{lon:.5f}): "
          f"loc={loc_score:.2f}, time={time_score:.2f}")

# ---------- 4) Capture endpoint ----------
@app.post("/capture")
def capture(payload: CapturePayload):
    data = payload.dict_with_time()

    # 1) Store raw point in time-series collection
    ts_collection.insert_one({
        "user_id":   data["user_id"],
        "latitude":  data["latitude"],
        "longitude": data["longitude"],
        "timestamp": data["timestamp"]
    })

    # 2) Optionally retrain behavioral profile if enough new data
    #    (could be made asynchronous / background)
    try:
        if should_retrain(ts_collection, data["user_id"], None):
            build_user_profile(data["user_id"], ts_collection, save_to_mongo=True)
    except Exception:
        # log & continue
        pass

    # 3) Compute location & time anomaly
    loc_score, time_score = detect_user_anomalies(
        data["latitude"], data["longitude"],
        data["timestamp"].hour,
        data["timestamp"].weekday(),
        data["timestamp"].month,
        data["user_id"],
        ts_collection
    )

    # 4) Append to alert_history in users_collection
    users_collection.update_one(
        {"user_id": data["user_id"]},
        {
            "$push": {
                "alert_history": {
                    "timestamp":              data["timestamp"],
                    "location_anomaly_score": loc_score,
                    "time_anomaly_score":     time_score,
                    "is_incident":            data["emergency"]
                }
            },
            "$set": {
                "model_metadata.last_trained": datetime.utcnow()
            }
        },
        upsert=True
    )

    # 5) Fuse via evaluate_user_behavior (loads threshold+incident models internally)
    result = evaluate_user_behavior(
        user_id=data["user_id"],
        location_score=loc_score,
        time_score=time_score,
        ts_collection=ts_collection
    )

    # 6) If flagged, trigger downstream alert
    if result.get("anomaly") or data["emergency"]:
        trigger_alert(data["user_id"], data["latitude"], data["longitude"], loc_score, time_score)

    # 7) Return full scoring result
    return {
        "location_anomaly_score": loc_score,
        "time_anomaly_score":     time_score,
        **result
    }
