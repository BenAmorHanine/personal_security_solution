from datetime import datetime
from pymongo import MongoClient
from .config import MONGO_URI, DEFAULT_PROB_THRESHOLD
from .profiling import detect_user_anomalies
from .incident_prediction import predict_incident, load_incident_model
from .db_functions import log_alert
from ..fusion.risky_location_inference import predict_risk

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]

def get_risky_location_score(lat, lon):
    """Query the risky_location_model to get a risk score for the given location."""
    try:
        score, _, _ = predict_risk(lat, lon, method='dbscan')
        print(f"[DEBUG] Risky location score for ({lat}, {lon}): {score} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return float(score)
    except Exception as e:
        print(f"[✗] Error loading risky_location_model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return 0.5  # Neutral score if model fails

def trigger_alert(user_id, device_id, latitude, longitude, sos_pressed=False, audio_file=None, location_anomaly=None, time_anomaly=None, ai_score=None, risky_location_score=None):
    """
    Decides whether to trigger an alert based on SOS button or risk scores.
    Returns (alert_id, loc_anomaly, time_anomaly, risky_location_score, ai_score, threshold) if triggered, else None.
    """
    try:
        # Automatic trigger on SOS button press
        if sos_pressed:
            alert_id = log_alert(user_id, device_id, latitude, longitude, auto_triggered=True, audio_file=audio_file)
            if alert_id:
                return alert_id, 1.0, 1.0, 1.0, 1.0, DEFAULT_PROB_THRESHOLD
            return None

        # Periodic check: trigger if risk scores exceed thresholds
        if location_anomaly is None or time_anomaly is None or ai_score is None or risky_location_score is None:
            loc_anomaly, time_anomaly = detect_user_anomalies(
                latitude, longitude, datetime.utcnow().hour,
                datetime.utcnow().weekday(), datetime.utcnow().month,
                user_id, locations_collection
            )
            model, scaler = load_incident_model(user_id)
            ai_score = predict_incident(model, scaler, loc_anomaly, time_anomaly) if model else max(loc_anomaly, time_anomaly, 0.5)
            risky_location_score = get_risky_location_score(latitude, longitude)
        else:
            loc_anomaly, time_anomaly = location_anomaly, time_anomaly

        # Threshold-based decision
        threshold = DEFAULT_PROB_THRESHOLD
        if loc_anomaly > 0.8 or time_anomaly > 0.8 or ai_score > 0.7 or risky_location_score > 0.7:
            alert_id = log_alert(user_id, device_id, latitude, longitude, auto_triggered=(risky_location_score > 0.7), audio_file=audio_file)
            if alert_id:
                return alert_id, loc_anomaly, time_anomaly, risky_location_score, ai_score, threshold
        
        return None
    except Exception as e:
        print(f"[✗] Error triggering alert for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def periodic_risk_check(user_id, device_id, latitude, longitude, audio_file=None):
    """
    Periodically checks risk based on location and (future) audio analysis.
    Returns (alert_id, loc_anomaly, time_anomaly, risky_location_score, ai_score, threshold) if triggered, else None.
    """
    try:
        # Get anomaly scores
        loc_anomaly, time_anomaly = detect_user_anomalies(
            latitude, longitude, datetime.utcnow().hour,
            datetime.utcnow().weekday(), datetime.utcnow().month,
            user_id, locations_collection
        )
        
        # Get incident probability
        model, scaler = load_incident_model(user_id)
        ai_score = predict_incident(model, scaler, loc_anomaly, time_anomaly) if model else max(loc_anomaly, time_anomaly, 0.5)
        
        # Get risky location score
        risky_location_score = get_risky_location_score(latitude, longitude)

        # Future: Add audio analysis when implemented
        # audio_stress = process_audio(audio_file) if audio_file else 0.0

        # Trigger alert if risk is high
        return trigger_alert(
            user_id, device_id, latitude, longitude,
            sos_pressed=False, audio_file=audio_file,
            location_anomaly=loc_anomaly, time_anomaly=time_anomaly,
            ai_score=ai_score, risky_location_score=risky_location_score
        )
    except Exception as e:
        print(f"[✗] Error in periodic risk check for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None