from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
from .profiling import detect_user_anomalies, build_user_profile
from .incident_prediction import predict_incident
from .threshold_adjustment import predict_threshold, load_threshold_model
from .utils import insert_location, insert_geo_data, insert_user_alert
import uuid
from .config import MONGO_URI, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]
geo_collection = db["geo_data"]

def should_retrain(user_id: str, collection=locations_collection, last_update: datetime = None) -> bool:
    """Determine if user profile needs retraining based on recent data or last update."""
    try:
        time_threshold = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_count = collection.count_documents({
            "user_id": user_id,
            "timestamp": {"$gte": time_threshold}
        })
        retrain = recent_count >= 10 or (last_update and (datetime.now(timezone.utc) - last_update) > timedelta(days=1))
        print(f"[DEBUG] Retrain check for {user_id}: recent_count={recent_count}, last_update={last_update}, retrain={retrain} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return retrain
    except Exception as e:
        print(f"[✗] Error in should_retrain for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return False

def process_capture(user_id: str, device_id: str, latitude: float, longitude: float, timestamp: datetime = None, sos_pressed: bool = False):
    """Process a user alert (SOS or periodic check), compute anomalies, predict incident probability, and store data."""
    try:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Validate inputs
        if not user_id or not device_id or latitude is None or longitude is None:
            raise ValueError("Missing required fields: user_id, device_id, latitude, or longitude")

        # Verify user exists
        user = users_collection.find_one({"user_id": user_id})#, "device_id": device_id})
        if not user:
            raise ValueError(f"User {user_id} with device {device_id} not found")

        # Update profile if needed
        last_update = user.get("profile", {}).get("last_updated") if user.get("profile") else None
        if should_retrain(user_id, last_update=last_update):
            build_user_profile(user_id, locations_collection)
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"profile.last_updated": timestamp}}
            )
            print(f"[✓] Updated profile for user {user_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Compute anomaly scores
        anomalies = detect_user_anomalies(latitude, longitude, timestamp.hour, timestamp.weekday(), timestamp.month, user_id, locations_collection)
        if not anomalies or len(anomalies) != 4:
            raise ValueError(f"Invalid anomaly scores for user {user_id}: {anomalies}")
        location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = anomalies

        # Predict incident probability
        incident_probability = predict_incident(user_id, location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly)

        # Predict threshold (use only 4 features to match training)
        features = [location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly]  # Removed timestamp.hour
        threshold_model, scaler = load_threshold_model(user_id)
        if threshold_model and scaler:
            threshold = predict_threshold(threshold_model, scaler, features)
        else:
            threshold = DEFAULT_PROB_THRESHOLD
            print(f"[DEBUG] Using default threshold {threshold} for user {user_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S CET')}")
        # Determine if incident
        is_incident = sos_pressed or (incident_probability >= threshold)
        alert_id = str(uuid.uuid4())

        # Store location data
        location_data = {
            "user_id": user_id,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp,
            "alert": {
                "alert_id": alert_id,
                "incident_probability": float(incident_probability),
                "is_incident": is_incident,
                "location_anomaly": float(location_anomaly),
                "hour_anomaly": float(hour_anomaly),
                "weekday_anomaly": float(weekday_anomaly),
                "month_anomaly": float(month_anomaly)
            }
        }
        insert_location(location_data, collection=locations_collection)

       #we used to store geo data in a separate collection, but now we store it in the same collection

        # Store user alert metadata
        insert_user_alert(user_id, alert_id, incident_probability, is_incident, timestamp, collection=users_collection)

        print(f"[✓] Processed alert for user {user_id}: alert_id={alert_id}, incident_probability={incident_probability:.2f}, threshold={threshold:.2f}, is_incident={is_incident} at {timestamp.strftime('%Y-%m-%d %H:%M:%S CET')}")

        return {
            "alert_id": alert_id,
            "incident_probability": incident_probability,
            "threshold": threshold,
            "is_incident": is_incident,
            "timestamp": timestamp,
            "location_anomaly": location_anomaly,
            "hour_anomaly": hour_anomaly,
            "weekday_anomaly": weekday_anomaly,
            "month_anomaly": month_anomaly
        }
    except Exception as e:
        print(f"[✗] Error processing alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None
    

"""
from datetime import datetime, timezone, timedelta
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
import random
import time
import argparse
from .profiling import detect_user_anomalies, build_user_profile
from .incident_prediction import predict_incident, load_incident_model
from .threshold_adjustment import predict_threshold, load_threshold_model
from .utils import insert_location, insert_geo_data, insert_user_alert
from .config import MONGO_URI, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
geo_collection = db["geo_data"]
users_collection = db["users"]

def should_retrain(ts_collection: Collection, user_id: str, last_update: datetime = None) -> bool:
    #Determine if user profile needs retraining.
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=24)
        recent_count = ts_collection.count_documents({
            "user_id": user_id,
            "timestamp": {"$gte": time_threshold}
        })
        return recent_count >= 10 or (last_update and (datetime.utcnow() - last_update) > timedelta(days=1))
    except Exception as e:
        print(f"[✗] Error in should_retrain for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return False

def capture_and_store(user_id: str, lat: float, lon: float, ts_collection: Collection, geo_collection: Collection):
    #Store GPS ping in collections.
    now = datetime.utcnow()
    data = {
        "user_id": user_id,
        "latitude": lat,
        "longitude": lon,
        "timestamp": now
    }
    insert_location(ts_collection, user_id, lat, lon, now)
    insert_geo_data(geo_collection, user_id, lat, lon)
    return now

def process_capture(user_id: str, lat: float, lon: float, sos_pressed: bool, ts_collection: Collection, geo_collection: Collection, users_collection: Collection):
    #Process GPS ping, update profile if needed, and check for alerts.
    try:
        now = capture_and_store(user_id, lat, lon, ts_collection, geo_collection)
        
        # Update profile if needed
        last_update = users_collection.find_one({"user_id": user_id}, {"profile.last_updated": 1})
        last_update = last_update.get("profile", {}).get("last_updated") if last_update else None
        if should_retrain(ts_collection, user_id, last_update):
            build_user_profile(user_id, ts_collection)
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"profile.last_updated": now}}
            )
            print(f"[✓] Updated profile for user {user_id} at {now.strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Compute anomaly scores
        threshold_model = load_threshold_model(user_id)
        df = pd.DataFrame(list(ts_collection.find({"user_id": user_id})))
        features = [
            df["timestamp"].dt.hour.std() if not df.empty else 0,
            df["timestamp"].diff().dt.total_seconds().ne(0).mean() if not df.empty and len(df) > 1 else 0,
            len(df)
        ]
        prob_threshold = predict_threshold(threshold_model, features) if threshold_model else DEFAULT_PROB_THRESHOLD
        loc_anomaly, time_anomaly = detect_user_anomalies(lat, lon, now.hour, now.weekday(), now.month, user_id, ts_collection, prob_threshold)

        # Predict incident
        incident_model, scaler = load_incident_model(user_id)
        incident_prob = predict_incident(incident_model, scaler, loc_anomaly, time_anomaly) if incident_model else None

        # Log alert
        is_incident = sos_pressed or (incident_prob is not None and incident_prob >= prob_threshold)
        alert_id = insert_user_alert(users_collection, user_id, loc_anomaly, time_anomaly, is_incident) if is_incident else None
        
        if is_incident:
            print(f"[✓] Alert {alert_id} logged for user {user_id} at {now.strftime('%Y-%m-%d %H:%M:%S CET')}: SOS={sos_pressed}, Incident Prob={incident_prob:.2f}")
        else:
            print(f"[DEBUG] No alert for user {user_id} at {now.strftime('%Y-%m-%d %H:%M:%S CET')}: Incident Prob={incident_prob:.2f}")

        return {
            "timestamp": now,
            "location_anomaly": loc_anomaly,
            "time_anomaly": time_anomaly,
            "dynamic_threshold": prob_threshold,
            "incident_probability": incident_prob,
            "anomaly_flag": is_incident,
            "alert_id": alert_id
        }
    except Exception as e:
        print(f"[✗] Error in process_capture for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def capture_location_periodically(user_id, device_id, duration_minutes=60, interval_seconds=600):
    #Periodically capture location, update profile, and check for alerts.
    try:
        base_lat, base_lon = 48.8566, 2.3522  # Paris coordinates
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now(timezone.utc) < end_time:
            lat = base_lat + random.uniform(-0.01, 0.01)
            lon = base_lon + random.uniform(-0.01, 0.01)
            result = process_capture(
                user_id, lat, lon, sos_pressed=False,
                ts_collection=locations_collection,
                geo_collection=geo_collection,
                users_collection=users_collection
            )
            if result is None:
                print(f"[✗] Failed to process location for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            else:
                print(f"[✓] Processed location for user {user_id} at {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S CET')}: Incident Prob={result['incident_probability']:.2f}")
            time.sleep(interval_seconds)
    except Exception as e:
        print(f"[✗] Error in periodic location capture for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate periodic location capture for a user.")
    parser.add_argument("--user_id", required=True, help="User ID for location capture")
    parser.add_argument("--device_id", required=True, help="Device ID for location capture")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--interval", type=int, default=600, help="Interval in seconds")
    args = parser.parse_args()
    
    capture_location_periodically(args.user_id, args.device_id, args.duration, args.interval)
"""


""" 
       # Store geo data
        geo_data = {
            "user_id": user_id,
            "alert_id": alert_id,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp,
        }
        insert_geo_data(geo_data, collection=geo_collection)
        """