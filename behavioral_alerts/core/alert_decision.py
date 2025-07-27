from datetime import datetime
from .profiling import detect_user_anomalies
from .db_functions import log_alert

def decide_alert(user_id, device_id, lat, lon, sos_pressed=False):
    """
    Decide whether to trigger an alert based on SOS or anomaly scores.
    Returns dict with alert_id, is_incident, and ai_score if triggered, else None.
    """
    try:
        hour = datetime.utcnow().hour
        weekday = datetime.utcnow().weekday()
        month = datetime.utcnow().month

        loc_anomaly, time_anomaly = detect_user_anomalies(lat, lon, hour, weekday, month, user_id, None)
        ai_score = loc_anomaly * 0.5 + time_anomaly * 0.5  # Placeholder scoring
        threshold = 0.5  # Placeholder threshold

        is_incident = sos_pressed or (ai_score > threshold)
        alert_id = log_alert(user_id, device_id, lat, lon) if is_incident else None
        if alert_id is None and is_incident:
            return None

        return {"alert_id": alert_id, "is_incident": is_incident, "ai_score": ai_score} if alert_id else None
    except Exception as e:
        print(f"[✗] Error deciding alert for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None