import uuid
from datetime import datetime, timezone
from pymongo import MongoClient, errors
import os
from .config import MONGO_URI, MODEL_DIR  # Added MODEL_DIR for log path
from .profiling import detect_user_anomalies
from .incident_prediction import predict_incident

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]
devices_collection = db["devices"]

# Define log directory
LOG_DIR = os.path.join(os.path.dirname(MODEL_DIR), "logs")  # E:\Solution_securite_perso\logs

def create_user(name, email, phone, emergency_contact_phone):
    """Create a new user in the database."""
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
    except errors.DuplicateKeyError:
        print(f"[✗] Error: User with email {email} already exists at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return None
    except Exception as e:
        print(f"[✗] Error creating user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def register_device(user_id, device_type, sim_id, battery_level):
    """Register a device for a user."""
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
        devices_collection.insert_one(device)
        print(f"[✓] Registered device {device_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return device_id
    except Exception as e:
        print(f"[✗] Error registering device for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def update_location(user_id, device_id, lat, lon, location_type="gps", timestamp=None):
    """Update user location in the database."""
    try:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        location = {
            "type": "Point",
            "coordinates": [float(lon), float(lat)]
        }
        location_id = str(uuid.uuid4())
        location_doc = {
            "location_id": location_id,
            "user_id": user_id,
            "device_id": device_id,
            "location": location,
            "location_type": location_type,
            "timestamp": timestamp
        }
        locations_collection.insert_one(location_doc)
        print(f"[✓] Updated location {location_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return location_id
    except Exception as e:
        print(f"[✗] Error updating location for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def log_alert(user_id, device_id, lat, lon, timestamp=None):
    """Log an SOS alert to MongoDB and a local log file."""
    try:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        hour = timestamp.hour
        weekday = timestamp.weekday()
        month = timestamp.month
        loc_anomaly, time_anomaly = detect_user_anomalies(lat, lon, hour, weekday, month, user_id)
        ai_score = predict_incident(user_id, loc_anomaly, time_anomaly)
        alert_id = str(uuid.uuid4())
        alert = {
            "alert_id": alert_id,
            "alert_time": timestamp,
            "location_anomaly_score": float(loc_anomaly),
            "time_anomaly_score": float(time_anomaly),
            "is_incident": True,
            "ai_score": float(ai_score),
            "status": "pending",
            "nearest_police_station": None
        }
        location = {
            "type": "Point",
            "coordinates": [float(lon), float(lat)]
        }
        alert_doc = {
            "location_id": str(uuid.uuid4()),
            "user_id": user_id,
            "device_id": device_id,
            "location": location,
            "location_type": "gps",
            "timestamp": timestamp,
            "alert": alert
        }
        locations_collection.insert_one(alert_doc)
        
        # Log to local file
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"alerts_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log")
        user = users_collection.find_one({"user_id": user_id})
        user_name = user.get("name", "Unknown") if user else "Unknown"
        log_message = (
            f"[ALERT] user_id={user_id}, name={user_name}, alert_id={alert_id}, "
            f"timestamp={timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}, "
            f"location=({lat}, {lon}), loc_anomaly={loc_anomaly:.2f}, "
            f"time_anomaly={time_anomaly:.2f}, ai_score={ai_score:.2f}, status=pending\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message)
        print(f"[✓] Logged alert {alert_id} for user {user_id} with AI score {ai_score:.2f} to DB and {log_file} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return alert_id
    except Exception as e:
        print(f"[✗] Error logging alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None
