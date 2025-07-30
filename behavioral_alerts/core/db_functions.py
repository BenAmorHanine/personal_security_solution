from pymongo import MongoClient
from datetime import datetime, timezone
import os
import uuid
from .config import MONGO_URI, LOG_DIR

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def create_user(name, email, phone, emergency_contact_phone, collection=users_collection):
    """Create a new user in MongoDB."""
    try:
        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "emergency_contact_phone": emergency_contact_phone,
            "devices": [],
            "created_at": datetime.now(timezone.utc)
        }
        collection.insert_one(user)
        print(f"[✓] Created user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return user_id
    except Exception as e:
        print(f"[✗] Error creating user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def register_device(user_id, device_type, sim_id, battery_level, collection=users_collection):
    """Register a device for a user."""
    try:
        device_id = str(uuid.uuid4())
        collection.update_one(
            {"user_id": user_id},
            {"$push": {
                "devices": {
                    "device_id": device_id,
                    "device_type": device_type,
                    "sim_id": sim_id,
                    "battery_level": battery_level,
                    "registered_at": datetime.now(timezone.utc)
                }
            }},
            upsert=True
        )
        print(f"[✓] Registered device {device_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return device_id
    except Exception as e:
        print(f"[✗] Error registering device for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def update_location(user_id, device_id, latitude, longitude, timestamp=None, collection=locations_collection):
    """Update user location in MongoDB."""
    try:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        location_id = str(uuid.uuid4())
        location = {
            "location_id": location_id,
            "user_id": user_id,
            "device_id": device_id,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp
        }
        collection.insert_one(location)
        print(f"[✓] Updated location for user {user_id}, device {device_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return location_id
    except Exception as e:
        print(f"[✗] Error updating location for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def log_alert(user_id, device_id, latitude, longitude, timestamp=None, ai_score=0.5, is_incident=None, collection=locations_collection):
    """Log an alert for a user with anomaly scores."""
    try:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        alert_id = str(uuid.uuid4())
        alert_data = {
            "alert_id": alert_id,
            "ai_score": ai_score,
            "location_anomaly_score": 0.0,  # Will be updated after anomaly detection
            "time_anomaly_score": 0.0      # Will be updated after anomaly detection
        }
        if is_incident is not None:
            alert_data["is_incident"] = int(is_incident)
        
        location = {
            "location_id": alert_id,
            "user_id": user_id,
            "device_id": device_id,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp,
            "alert": alert_data
        }
        collection.insert_one(location)
        
        # Write to log file
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"alerts_{datetime.now().strftime('%Y-%m-%d')}.log")
        log_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')} - Alert {alert_id} for user {user_id}: ai_score={ai_score}"
        if is_incident is not None:
            log_message += f", is_incident={is_incident}"
        log_message += "\n"
        with open(log_file, "a") as f:
            f.write(log_message)
        
        print(f"[✓] Logged alert {alert_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return alert_id
    except Exception as e:
        print(f"[✗] Error logging alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None