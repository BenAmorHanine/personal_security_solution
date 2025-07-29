import random
import uuid  # Added for uuid4
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from behavioral_alerts.core.db_functions import create_user, register_device, update_location, log_alert
from behavioral_alerts.core.profiling import build_user_profile, detect_user_anomalies
from behavioral_alerts.core.incident_prediction import train_incident_model, save_incident_model, prepare_incident_data
from behavioral_alerts.core.config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def run_test_pipeline():
    try:
        # Step 1: Create new user (zero data)
        user_id = create_user(
            name="Test User",
            email="test@test.com",
            phone="+33612345678",
            emergency_contact_phone="+33687654321"
        )
        if user_id is None:
            raise Exception("Failed to create user")
        print(f"[✓] Created test user: {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        device_id = register_device(
            user_id=user_id,
            device_type="smartphone",
            sim_id="1234567890",
            battery_level=100
        )
        if device_id is None:
            raise Exception("Failed to register device")
        print(f"[✓] Registered device: {device_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Step 2: Test with zero data
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id, locations_collection)
        if centroids is None:
            print(f"[DEBUG] No profile built for {user_id} due to zero data, profile creation deferred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            raise Exception("Unexpected profile creation with zero data")
        
        features, labels = prepare_incident_data(user_id, locations_collection)
        model, scaler = train_incident_model(features, labels)
        if model is None:
            print(f"[WARNING] No incident model trained for {user_id} due to zero alert data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            save_incident_model(user_id, model, scaler, save_to_mongo=True, users_collection=users_collection)
            print(f"[✓] Trained and saved incident model (dummy) for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        lat_anomaly, lon_anomaly = 48.9566, 2.4522  # Slightly outside Paris
        loc_anomaly, time_anomaly = detect_user_anomalies(
            lat_anomaly, lon_anomaly, 3, 0, 7, user_id, locations_collection
        )
        print(f"[✓] Anomaly scores with zero data - Location: {loc_anomaly}, Time: {time_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        alert_id = log_alert(user_id, device_id, lat_anomaly, lon_anomaly)
        if alert_id is None:
            print(f"[WARNING] Failed to trigger SOS alert for {user_id} due to zero data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            print(f"[✓] SOS Alert with zero data: {{'alert_id': {alert_id}, 'is_incident': True, 'ai_score': <float>}} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Step 3: Simulate periodic location capture
        base_lat, base_lon = 48.8566, 2.3522  # Paris coordinates
        for i in range(15):  # Simulate 15 updates (enough for profile)
            lat = base_lat + random.uniform(-0.01, 0.01)
            lon = base_lon + random.uniform(-0.01, 0.01)
            timestamp = datetime.now(timezone.utc) - timedelta(days=30 - i % 30)
            location_id = update_location(user_id, device_id, lat, lon, timestamp=timestamp)
            if location_id is None:
                raise Exception("Failed to update location")
            print(f"[DEBUG] Inserted location {location_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        print(f"[✓] Added 15 location updates at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Step 4: Build profile with location data
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id, locations_collection)
        if centroids is None:
            raise Exception("Failed to build user profile with sufficient data")
        print(f"[✓] Built user profile: {len(centroids)} clusters at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Step 5: Add synthetic alerts
        for i in range(10):
            lat = base_lat + random.uniform(-0.05 if i % 2 else -0.01, 0.05 if i % 2 else 0.01)
            lon = base_lon + random.uniform(-0.05 if i % 2 else -0.01, 0.05 if i % 2 else 0.01)
            timestamp = datetime.now(timezone.utc) - timedelta(days=15 - i)
            loc_anomaly, time_anomaly = detect_user_anomalies(
                lat, lon, timestamp.hour, timestamp.weekday(), timestamp.month, user_id, locations_collection
            )
            alert = {
                "alert_id": str(uuid.uuid4()),  # Fixed: Use uuid.uuid4
                "alert_time": timestamp,
                "location_anomaly_score": float(loc_anomaly),
                "time_anomaly_score": float(time_anomaly),
                "is_incident": bool(i % 2),  # Alternate True/False
                "status": "pending",
                "nearest_police_station": None
            }
            locations_collection.insert_one({
                "user_id": user_id,
                "device_id": device_id,
                "location": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "timestamp": timestamp,
                "location_type": "gps",
                "alert": alert
            })
            print(f"[DEBUG] Inserted synthetic alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        print(f"[✓] Added 10 synthetic alerts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Step 6: Train incident model with alerts
        features, labels = prepare_incident_data(user_id, locations_collection)
        model, scaler = train_incident_model(features, labels)
        if model is None:
            raise Exception("Failed to train incident model")
        save_incident_model(user_id, model, scaler, save_to_mongo=True, users_collection=users_collection)
        print(f"[✓] Trained and saved incident model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Step 7: Test alert with trained models
        loc_anomaly, time_anomaly = detect_user_anomalies(
            lat_anomaly, lon_anomaly, 3, 0, 7, user_id, locations_collection
        )
        print(f"[✓] Anomaly scores with data - Location: {loc_anomaly}, Time: {time_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        alert_id = log_alert(user_id, device_id, lat_anomaly, lon_anomaly)
        if alert_id is None:
            raise Exception("Failed to trigger SOS alert")
        print(f"[✓] SOS Alert with data: {{'alert_id': {alert_id}, 'is_incident': True, 'ai_score': <float>}} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error in test pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

if __name__ == "__main__":
    run_test_pipeline()
