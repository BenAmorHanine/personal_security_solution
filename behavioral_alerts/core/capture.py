from datetime import datetime
import pandas as pd
from pymongo.collection import Collection
from .profiling import detect_user_anomalies
from .incident_prediction import predict_incident, load_incident_model
from .threshold_adjustment import predict_threshold, load_threshold_model
from .utils import insert_location, insert_geo_data, insert_user_alert

def capture_and_store(user_id: str, lat: float, lon: float, ts_collection: Collection, geo_collection: Collection, users_collection: Collection):
    now = datetime.now()
    data = {
        "user_id": user_id,
        "latitude": lat,
        "longitude": lon,
        "timestamp": now
    }
    insert_location(ts_collection, user_id, lat, lon, now)
    insert_geo_data(geo_collection, user_id, lat, lon)
    return now

def process_capture(user_id: str, lat: float, lon: float, ts_collection: Collection, geo_collection: Collection, users_collection: Collection):
    now = capture_and_store(user_id, lat, lon, ts_collection, geo_collection, users_collection)
    threshold_model = load_threshold_model(user_id)
    df = pd.DataFrame(list(ts_collection.find({"user_id": user_id})))
    features = [df["hour"].std() if not df.empty else 0, 0, len(df)]  # Placeholder transition_freq
    prob_threshold = predict_threshold(threshold_model, features) if threshold_model else 0.05
    loc_anomaly, time_anomaly = detect_user_anomalies(lat, lon, now.hour, now.weekday(), now.month, user_id, ts_collection, prob_threshold)
    incident_model, scaler = load_incident_model(user_id)
    incident_prob = predict_incident(incident_model, scaler, loc_anomaly, time_anomaly) if incident_model else None
    insert_user_alert(users_collection, user_id, loc_anomaly, time_anomaly, incident_prob > 0.7 if incident_prob else None)
    return loc_anomaly, time_anomaly, incident_prob

#or in preprocessing.py file:

from datetime import datetime
import pandas as pd
from pymongo.collection import Collection

from .profiling           import should_retrain, build_user_profile, detect_user_anomalies
from .threshold_adjustment import load_threshold_model, predict_threshold
from .incident_prediction  import load_incident_model, predict_incident
from .data_utils           import insert_location, insert_geo_data, insert_user_alert


def process_capture(
    user_id: str,
    lat: float,
    lon: float,
    emergency: bool,
    ts_collection: Collection,
    geo_collection: Collection,
    users_collection: Collection
):
    """Ingest one new GPS ping + optional SOS, update profiles & models, return scores."""
    now = datetime.utcnow()

    # 1) Store raw GPS + Geo
    insert_location(ts_collection, user_id, lat, lon, now)
    insert_geo_data   (geo_collection, user_id, lat, lon)

    # 2) Rebuild behavior profile if needed
    if should_retrain(ts_collection, user_id, None):
        build_user_profile(user_id, ts_collection, save_to_mongo=True)

    # 3) Compute anomaly scores
    loc_score, time_score = detect_user_anomalies(
        lat, lon, now.hour, now.weekday(), now.month,
        user_id, ts_collection
    )

    # 4) Determine dynamic threshold
    thresh_model = load_threshold_model(user_id)
    # fallback to default if missing
    if thresh_model:
        # reassemble exactly the same features used during training:
        df = pd.DataFrame(list(ts_collection.find({"user_id": user_id})))
        hour_std = df["timestamp"].dt.hour.std() if not df.empty else 0
        transition = df["timestamp"].diff().dt.total_seconds().ne(0).mean() if "timestamp" in df else 0
        volume     = len(df)
        threshold  = predict_threshold(thresh_model, [hour_std, transition, volume])
    else:
        threshold = 0.05

    # 5) Load incident model & predict
    incident_model, scaler = load_incident_model(user_id)
    incident_prob = None
    if incident_model and scaler:
        incident_prob = predict_incident(incident_model, scaler, loc_score, time_score)

    # 6) Append to user’s alert_history
    insert_user_alert(
        users_collection,
        user_id,
        loc_score,
        time_score,
        is_incident = emergency or (incident_prob is not None and incident_prob >= threshold)
    )

    # 7) Return everything
    return {
        "timestamp":            now,
        "location_anomaly":     loc_score,
        "time_anomaly":         time_score,
        "dynamic_threshold":    threshold,
        "incident_probability": incident_prob,
        "anomaly_flag":         emergency or (incident_prob is not None and incident_prob >= threshold)
    }
