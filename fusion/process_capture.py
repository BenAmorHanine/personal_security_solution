import pandas as pd
from fusion_core import *


import sys, os
from pathlib import Path

proj_root = Path(os.getcwd()).parent
sys.path.insert(0, str(proj_root))

from behavioral_alerts.core.incident_prediction import *
from behavioral_alerts.core.threshold_adjustment import *
from behavioral_alerts.core.profiling import detect_user_anomalies
from behavioral_alerts.core.capture import capture_and_store

def process_capture(user_id: str, lat: float, lon: float,
                    ts_collection, geo_collection, users_collection):
    """
    Process user data, predict risks, and fuse alerts.

    Args:
        user_id: User identifier.
        lat, lon: Geographic coordinates.
        ts_collection, geo_collection, users_collection: Database collections.

    Returns:
        Dictionary with signals and fusion result.
    """
    try:
        now = capture_and_store(user_id, lat, lon, ts_collection, geo_collection, users_collection)

        # Predict Unusual Time
        threshold_model = load_threshold_model(user_id)
        df = pd.DataFrame(list(ts_collection.find({"user_id": user_id})))
        features = [df["hour"].std() if not df.empty else 0, 0, len(df)]
        prob_threshold = predict_threshold(threshold_model, features) if threshold_model else 0.05

        loc_anomaly, time_anomaly = detect_user_anomalies(
            lat, lon, now.hour, now.weekday(), now.month, user_id, ts_collection, prob_threshold
        )

        # Predict Incident / Behavior Pattern
        incident_model, scaler = load_incident_model(user_id)
        incident_prob = predict_incident(incident_model, scaler, loc_anomaly, time_anomaly) if incident_model else 0.0

        # Other Model Predictions
        risk_location_score = predict_crime_risk(lat, lon)
        movement_score = detect_movement_anomaly(user_id, ts_collection, lat, lon, now)
        audio_stress_score = predict_audio_stress(user_id)
        keyword_score = detect_emergency_keyword(user_id)

        # Fuse Signals
        signals = {
            "risk_location": risk_location_score,
            "unusual_time": time_anomaly,
            "abnormal_movement": movement_score,
            "audio_stress": audio_stress_score,
            "keyword_alert": keyword_score,
            "behavior_pattern": incident_prob
        }
        fusion_result = fuse_alerts(signals)

        # Store and Trigger Alert
        insert_user_alert(users_collection, user_id, loc_anomaly, time_anomaly, incident_prob, fusion_result["trigger_alert"])
        if fusion_result["trigger_alert"]:
            trigger_alert(user_id)

        # Display Results
        display_outputs(signals, fuse_alerts.DEFAULT_WEIGHTS, fusion_result)

        return {"signals": signals, "fusion": fusion_result}

    except Exception as e:
        logging.error(f"Error in process_capture for user {user_id}: {str(e)}")
        return {"signals": {}, "fusion": {"trigger_alert": False, "error": str(e)}}