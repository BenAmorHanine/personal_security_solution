from pymongo import MongoClient
from datetime import datetime
import uuid
from config import MONGO_URI, TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE
from twilio.rest import Client
import requests
import numpy as np
import os

client = MongoClient(MONGO_URI)
db = client["safety_db"]
users_collection = db["users"]
locations_collection = db["locations"]

def create_user(name, email, phone, emergency_contact_phone, emergency_contacts=None, subscription_status="free"):
    try:
        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "emergency_contact_phone": emergency_contact_phone,
            "emergency_contacts": emergency_contacts or [],
            "subscription_status": subscription_status,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        users_collection.insert_one(user)
        return user_id
    except Exception as e:
        print(f"Error creating user: {e}")
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
            "last_seen": datetime.utcnow()
        }
        locations_collection.insert_one(device)
        return device_id
    except Exception as e:
        print(f"Error registering device: {e}")
        return None

def sign_in(email, password_hash):
    try:
        user = users_collection.find_one({"email": email, "password_hash": password_hash})
        if not user:
            raise ValueError("Invalid credentials")
        return user["user_id"]
    except Exception as e:
        print(f"Error signing in: {e}")
        return None


def log_alert(user_id, device_id, latitude, longitude, audio_file=None, health_data=None):
    from profiling import detect_user_anomalies
    from incident_prediction import predict_incident, load_incident_model
    try:
        alert_id = str(uuid.uuid4())
        police_station = find_nearest_police_station(latitude, longitude)
        loc_anomaly, time_anomaly = detect_user_anomalies(
            latitude, longitude, datetime.utcnow().hour,
            datetime.utcnow().weekday(), datetime.utcnow().month,
            user_id, locations_collection
        )
        model, scaler = load_incident_model(user_id)
        ai_score = predict_incident(model, scaler, loc_anomaly, time_anomaly) if model else 0.0

        alert = {
            "alert_id": alert_id,
            "alert_time": datetime.utcnow(),
            "audio_file": audio_file,
            "health_data": health_data,
            "location_anomaly_score": loc_anomaly,
            "time_anomaly_score": time_anomaly,
            "ai_score": ai_score,
            "is_incident": None,  # To be updated later
            "status": "pending",
            "nearest_police_station": police_station
        }
        locations_collection.insert_one({
            "user_id": user_id,
            "device_id": device_id,
            "location": {"type": "Point", "coordinates": [longitude, latitude]},
            "timestamp": datetime.utcnow(),
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
        print(f"Error logging alert: {e}")
        return None

def find_nearest_police_station(latitude, longitude):
    # Replace with real API (e.g., OpenStreetMap Overpass API)
    try:
        # Mock implementation
        return {
            "name": "Nearest Police Station",
            "address": "123 Rue de la Paix",
            "distance_m": 500
        }
    except Exception as e:
        print(f"Error finding police station: {e}")
        return None

def send_emergency_notification(phone, latitude, longitude, alert_id):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Emergency Alert {alert_id} at ({latitude}, {longitude})",
            from_=TWILIO_PHONE,
            to=phone
        )
        print(f"[✓] Sent SMS to {phone} for alert {alert_id}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

def send_police_notification(police_station, latitude, longitude, alert_id):
    # Replace with real police API
    print(f"Notifying {police_station['name']}: Alert {alert_id} at ({latitude}, {longitude})")

def update_location(user_id, device_id, latitude, longitude, location_type="gps"):
    try:
        location_id = str(uuid.uuid4())
        locations_collection.insert_one({
            "location_id": location_id,
            "user_id": user_id,
            "device_id": device_id,
            "location": {"type": "Point", "coordinates": [longitude, latitude]},
            "timestamp": datetime.utcnow(),
            "location_type": location_type
        })
        # Update zones
        user = users_collection.find_one({"user_id": user_id})
        centroids = user.get("behavior_profile", {}).get("centroids", [])
        for zone in centroids:
            dist = np.sqrt((latitude - zone["center"]["coordinates"][1])**2 + (longitude - zone["center"]["coordinates"][0])**2)
            if dist < zone["radius"] / 6378100:  # Convert meters to degrees
                # Update visit count (simplified; consider a separate zones collection if needed)
                break
        else:
            locations_collection.insert_one({
                "user_id": user_id,
                "location": {"type": "Point", "coordinates": [longitude, latitude]},
                "timestamp": datetime.utcnow(),
                "location_type": location_type
            })
        return location_id
    except Exception as e:
        print(f"Error updating location: {e}")
        return None