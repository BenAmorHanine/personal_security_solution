from pymongo import MongoClient
from datetime import datetime, timezone
import uuid
from .config import MONGO_URI, TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE
import numpy as np
import os

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def create_user(name, email, phone, emergency_contact_phone, emergency_contacts=None):
    try:
        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "emergency_contact_phone": emergency_contact_phone,
            "emergency_contacts": emergency_contacts or [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        users_collection.insert_one(user)
        return user_id
    except Exception as e:
        print(f"[✗] Error creating user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def sign_in(email, password_hash):
    try:
        user = users_collection.find_one({"email": email, "password_hash": password_hash})
        if not user:
            raise ValueError("Invalid credentials")
        return user["user_id"]
    except Exception as e:
        print(f"[✗] Error signing in at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def register_device(user_id, device_type, sim_id, battery_level=100):
    try:
        device_id = str(uuid.uuid4())
        device = {
            "device_id": device_id,
            "user_id": user_id,
            "device_type": device_type,
            "sim_id": sim_id,
            "battery_level": battery_level,
            "last_seen": datetime.now(timezone.utc),
            "timestamp": datetime.now(timezone.utc)
        }
        locations_collection.insert_one(device)
        return device_id
    except Exception as e:
        print(f"[✗] Error registering device at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def log_alert(user_id, device_id, latitude, longitude, audio_file=None, health_data=None):
    from .profiling import detect_user_anomalies
    from .incident_prediction import predict_incident, load_incident_model
    try:
        alert_id = str(uuid.uuid4())
        police_station = find_nearest_police_station(latitude, longitude)
        loc_anomaly, time_anomaly = detect_user_anomalies(
            latitude, longitude, datetime.now(timezone.utc).hour,
            datetime.now(timezone.utc).weekday(), datetime.now(timezone.utc).month,
            user_id, locations_collection
        )
        model, scaler = load_incident_model(user_id)
        ai_score = predict_incident(model, scaler, loc_anomaly, time_anomaly) if model else 0.0

        alert = {
            "alert_id": alert_id,
            "alert_time": datetime.now(timezone.utc),
            "audio_file": audio_file,
            "health_data": health_data or {"heart_rate": None, "stress": None},
            "location_anomaly_score": float(loc_anomaly),
            "time_anomaly_score": float(time_anomaly),
            "ai_score": float(ai_score),
            "is_incident": None,
            "status": "pending",
            "nearest_police_station": police_station
        }
        locations_collection.insert_one({
            "user_id": user_id,
            "device_id": device_id,
            "location": {"type": "Point", "coordinates": [float(longitude), float(latitude)]},
            "timestamp": datetime.now(timezone.utc),
            "location_type": "gps",
            "alert": alert
        })

        user = users_collection.find_one({"user_id": user_id})
        send_emergency_notification(user["emergency_contact_phone"], latitude, longitude, alert_id)
        if ai_score > 0.7:
            send_police_notification(police_station, latitude, longitude, alert_id)
            locations_collection.update_one(
                {"alert.alert_id": alert_id},
                {"$set": {"alert.status": "sent_to_police"}}
            )

        return alert_id
    except Exception as e:
        print(f"[✗] Error logging alert at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def find_nearest_police_station(latitude, longitude):
    return None

def send_emergency_notification(phone, latitude, longitude, alert_id):
    pass

def send_police_notification(police_station, latitude, longitude, alert_id):
    print(f"[✓] Notifying police station at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    pass

def update_location(user_id, device_id, latitude, longitude, location_type="gps", timestamp=None):
    try:
        location_id = str(uuid.uuid4())
        location_doc = {
            "location_id": location_id,
            "user_id": user_id,
            "device_id": device_id,
            "location": {"type": "Point", "coordinates": [float(longitude), float(latitude)]},
            "timestamp": timestamp if timestamp else datetime.now(timezone.utc),
            "location_type": location_type
        }
        result = locations_collection.insert_one(location_doc)
        print(f"[DEBUG] Inserted location {location_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return location_id
    except Exception as e:
        print(f"[✗] Error updating location at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def update_device_status(device_id, battery_level):
    try:
        locations_collection.update_one(
            {"device_id": device_id},
            {"$set": {"battery_level": battery_level, "last_seen": datetime.now(timezone.utc)}}
        )
        print(f"[✓] Updated status for device {device_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error updating device status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
