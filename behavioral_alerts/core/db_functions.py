import uuid
from datetime import datetime, timezone
from pymongo import MongoClient
from .config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def create_user(name, email, phone, emergency_contact_phone):
    """Create a new user and return their user_id."""
    try:
        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "emergency_contact_phone": emergency_contact_phone,
            "created_at": datetime.now(timezone.utc)
        }
        users_collection.insert_one(user)
        print(f"[✓] Created user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return user_id
    except Exception as e:
        print(f"[✗] Error creating user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

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

def log_alert(user_id, device_id, latitude, longitude, timestamp=None, ai_score=None, is_incident=None):
    """Log an alert for a user and return the alert_id."""
    try:
        latitude = float(latitude)
        longitude = float(longitude)
        alert_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        alert = {
            "location_id": alert_id,
            "user_id": user_id,
            "device_id": device_id,
            "location": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            },
            "timestamp": timestamp,
            "alert": {
                "alert_id": alert_id,
                "ai_score": ai_score,
                "is_incident": is_incident
            }
        }
        locations_collection.insert_one(alert)
        print(f"[✓] Logged alert {alert_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return alert_id
    except Exception as e:
        print(f"[✗] Error logging alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None