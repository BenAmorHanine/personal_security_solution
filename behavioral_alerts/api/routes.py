from fastapi import FastAPI
from core.profiling import detect_user_anomalies, load_profile
from core.incident_prediction import predict_incident, load_incident_model
from core.threshold_adjustment import predict_threshold, load_threshold_model
from core.data_utils import setup_timeseries_collection, setup_geospatial_collection, setup_users_collection
import pandas as pd

app = FastAPI()
ts_collection = setup_timeseries_collection()
geo_collection = setup_geospatial_collection()
users_collection = setup_users_collection()

@app.post("/anomaly_score")
async def get_anomaly_score(user_id: str, lat: float, lon: float, hour: int, weekday: int, month: int):
    threshold_model = load_threshold_model(user_id)
    df = pd.DataFrame(list(ts_collection.find({"user_id": user_id})))
    features = [df["hour"].std() if not df.empty else 0, 0, len(df)]
    prob_threshold = predict_threshold(threshold_model, features) if threshold_model else 0.05
    loc_anomaly, time_anomaly = detect_user_anomalies(lat, lon, hour, weekday, month, user_id, ts_collection, prob_threshold)
    incident_model, scaler = load_incident_model(user_id)
    incident_prob = predict_incident(incident_model, scaler, loc_anomaly, time_anomaly) if incident_model else None
    return {
        "user_id": user_id,
        "location_anomaly": loc_anomaly,
        "time_anomaly": time_anomaly,
        "incident_probability": incident_prob,
        "prob_threshold": prob_threshold
    }