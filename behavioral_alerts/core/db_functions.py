import uuid
from datetime import datetime, timezone
from pymongo import MongoClient
from config import MONGO_URI, LOG_DIR
import os, json
import numpy as np

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def create_user(name, email, phone, emergency_contact_phone, collection=users_collection):
    """Create a new user with a unique email."""
    try:
        user_id = str(uuid.uuid4())
        # Check if email already exists
        existing_email_doc = collection.find_one({"email": email})
        if existing_email_doc:
            # Generate a unique email by appending a random suffix
            email = f"{email.split('@')[0]}_{uuid.uuid4().hex[:8]}@example.com"
            print(f"[DEBUG] Email {email} already exists, using unique email {email} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        user_doc = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "emergency_contact_phone": emergency_contact_phone,
            "created_at": datetime.now(timezone.utc),
            "devices": []
        }
        collection.insert_one(user_doc)
        print(f"[✓] Created user {user_id} with email {email} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return user_id
    except Exception as e:
        print(f"[✗] Error creating user with email {email} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        raise

def drop_user(user_id: str):
    """
    Deletes the user and all associated data from the database.
    """
    result = users_collection.delete_one({"user_id": user_id})
    return result.deleted_count

def register_device(user_id, device_type, sim_id, battery_level):
    """Register a device for a user and return the device_id."""
    try:
        device_id = str(uuid.uuid4())
        device = {
            "device_id": device_id,
            "user_id": user_id,
            "device_type": device_type,
            "sim_id": sim_id,
            "battery_level": battery_level,
            "registered_at": datetime.now(timezone.utc)
        }
        users_collection.update_one(
            {"user_id": user_id},
            {"$push": {"devices": device}}
        )
        print(f"[✓] Registered device {device_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return device_id
    except Exception as e:
        print(f"[✗] Error registering device for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def update_location(user_id, device_id, latitude, longitude, timestamp=None):
    """Update user location and return the location_id."""
    try:
        location_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        location = {
            "location_id": location_id,
            "user_id": user_id,
            "device_id": device_id,
            "location": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            },
            "timestamp": timestamp
        }
        locations_collection.insert_one(location)
        print(f"[✓] Updated location {location_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return location_id
    except Exception as e:
        print(f"[✗] Error updating location for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def log_alert(
    user_id: str,
    device_id: str,
    latitude: float,
    longitude: float,
    *,
    incident_probability: float,
    is_incident: bool,
    location_anomaly: float,
    hour_anomaly: float,
    weekday_anomaly: float,
    month_anomaly: float,
    timestamp: datetime = None,
    save_locally: bool = True
) -> str:
    """
    Log an alert for a user. All anomaly scores and probability must be provided.
    Returns the generated alert_id, or None on failure.
    """
    try:
        alert_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Convert NumPy types to Python types
        incident_probability = float(incident_probability) if isinstance(incident_probability, (np.floating, np.integer)) else incident_probability
        is_incident = bool(is_incident) if isinstance(is_incident, np.bool_) else is_incident
        location_anomaly = float(location_anomaly) if isinstance(location_anomaly, (np.floating, np.integer)) else location_anomaly
        alert_doc = {
            "user_id": user_id,
            "device_id": device_id,
            "location": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            },
            "timestamp": timestamp,
            "alert": {
                "alert_id": alert_id,
                "incident_probability": incident_probability,
                "is_incident": is_incident,
                "location_anomaly": location_anomaly,
                "hour_anomaly": hour_anomaly,
                "weekday_anomaly": weekday_anomaly,
                "month_anomaly": month_anomaly
            }
        }

        locations_collection.insert_one(alert_doc)
        print(f"[✓] Logged alert {alert_id} for user {user_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Optionally write to local file in LOG_DIR
        if save_locally:
            os.makedirs(LOG_DIR, exist_ok=True)
            local_path = os.path.join(LOG_DIR, f"alert_{alert_id}.json")
            # Prepare JSON-serializable version
            serializable = {
                "user_id": alert_doc["user_id"],
                "device_id": alert_doc["device_id"],
                "location": alert_doc["location"],
                "timestamp": alert_doc["timestamp"].isoformat(),
                "alert": alert_doc["alert"]
            }
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            print(f"[✓] Saved alert {alert_id} locally at {local_path}")

        return alert_id

    except Exception as e:
        print(f"[✗] Error logging alert for user {user_id}: {e}")
        return None

