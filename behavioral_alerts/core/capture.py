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