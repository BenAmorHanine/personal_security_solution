
from datetime import datetime, timedelta, timezone
import numpy as np
from .db_functions import create_user, register_device, update_location
from .profiling import build_user_profile, detect_user_anomalies
from .risk_assessment import trigger_alert, periodic_risk_check
from .alert_decision import decide_alert
from pymongo import MongoClient
from .config import MONGO_URI
import time

def test_pipeline():
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]
        users_collection = db["users"]
        locations_collection = db["locations"]

        # Create user with unique email
        email = f"john{str(uuid.uuid4())[:8]}@example.com"
        user_id = create_user("John Doe", email, "+1234567890", "+0987654321")
        if not user_id:
            raise ValueError("Failed to create user")
        print(f"[✓] Created test user: {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Clear previous location data
        locations_collection.delete_many({"user_id": user_id})

        # Register device
        device_id = register_device(user_id, "smartphone", "SIM123456")
        if not device_id:
            raise ValueError("Failed to register device")
        print(f"[✓] Registered device: {device_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Simulate location updates with varied timestamps
        np.random.seed(42)
        lat_base, lon_base = 48.8566, 2.3522
        base_time = datetime.now(timezone.utc) - timedelta(days=29)  # Start 29 days ago
        for i in range(100):
            lat = lat_base + np.random.normal(0, 0.01)
            lon = lon_base + np.random.normal(0, 0.01)
            timestamp = base_time + timedelta(hours=i * 6)  # Every 6 hours
            location_id = update_location(user_id, device_id, lat, lon, timestamp=timestamp)
            if not location_id:
                raise ValueError("Failed to update location")
        print(f"[✓] Added 100 location updates at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        # Debug: Verify location count
        time.sleep(1)  # Wait for writes to settle
        location_count = locations_collection.count_documents({"user_id": user_id})
        valid_count = locations_collection.count_documents({
            "user_id": user_id,
            "location": {"$exists": True},
            "alert": {"$exists": False},
            "timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(days=30)}
        })
        print(f"[DEBUG] Total locations for user {user_id}: {location_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        print(f"[DEBUG] Valid locations for user {user_id}: {valid_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Build user profile
        centroids, _, _, _, _ = build_user_profile(user_id, locations_collection)
        if not centroids:
            raise ValueError("Failed to build user profile")
        print(f"[✓] Built user profile: {len(centroids)} clusters at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Test anomaly detection
        lat_anomaly, lon_anomaly = 48.0, 2.0
        loc_anomaly, time_anomaly = detect_user_anomalies(lat_anomaly, lon_anomaly, 3, 0, 7, user_id, locations_collection)
        print(f"[✓] Anomaly scores - Location: {loc_anomaly}, Time: {time_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Test SOS alert
        result = decide_alert(user_id, device_id, lat_anomaly, lon_anomaly, sos_pressed=True)
        if result and result["alert_id"]:
            print(f"[✓] SOS Alert: {result} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            raise ValueError("Failed to trigger SOS alert")

        # Test periodic risk check
        result = periodic_risk_check(user_id, device_id, lat_anomaly, lon_anomaly)
        if result and result[0]:
            print(f"[✓] Periodic Alert: ID={result[0]}, Scores={result[1:4]}, Threshold={result[4]} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            print(f"[✗] Periodic alert not triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

    except Exception as e:
        print(f"[✗] Error in test pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    import uuid
    test_pipeline()