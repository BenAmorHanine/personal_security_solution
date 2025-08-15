from pymongo import MongoClient
from datetime import datetime
from .config import MONGO_URI, DB_NAME
import pandas as pd
import base64
import joblib
import io

# behavioral_alerts/core/data_preprocessing.py

import pandas as pd
from datetime import datetime


def preprocess_user_data(user_id, collection):
    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    if df.empty or len(df) < 10:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0) / 3600

    return df

def serialize_model(obj):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def deserialize_model(encoded_str):
    buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
    return joblib.load(buffer)

from pymongo import MongoClient, GEOSPHERE
from .config import MONGO_URI
from datetime import datetime, timezone

def setup_timeseries_collection():
    """Set up the device_logs time-series collection."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]
        collections = db.list_collection_names()
        if "device_logs" not in collections:
            db.create_collection(
                "device_logs",
                timeseries={
                    "timeField": "timestamp",
                    "metaField": "device_id",
                    "granularity": "seconds"
                }
            )
        db.device_logs.create_index("device_id")
        db.device_logs.create_index("timestamp")
        print(f"[✓] Initialized device_logs collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return db.device_logs
    except Exception as e:
        print(f"[✗] Error setting up device_logs collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        raise

def setup_geospatial_collection():
    """Set up the locations collection with geospatial index."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]
        collections = db.list_collection_names()
        if "locations" not in collections:
            db.create_collection("locations")
        db.locations.create_index("user_id")
        db.locations.create_index("device_id")
        db.locations.create_index([("location", GEOSPHERE)])
        print(f"[✓] Initialized locations collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return db.locations
    except Exception as e:
        print(f"[✗] Error setting up locations collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        raise

def setup_users_collection():
    """Set up the users collection."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]
        if "users" not in db.list_collection_names():
            db.create_collection("users")
        db.users.create_index("user_id", unique=True)
        print(f"[✓] Initialized users collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return db.users
    except Exception as e:
        print(f"[✗] Error setting up users collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        raise



###############################################################################
from pymongo import MongoClient
from datetime import datetime, timezone

def insert_location(location_data, collection):
    """Insert or update location data in the specified collection."""
    try:
        collection.update_one(
            {"user_id": location_data["user_id"], "alert.alert_id": location_data["alert"]["alert_id"]},
            {"$set": location_data},
            upsert=True
        )
        print(f"[✓] Inserted location data for user {location_data['user_id']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error inserting location data for user {location_data['user_id']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

def insert_geo_data(geo_data, collection):
    """Insert or update geo data in the specified collection."""
    try:
        collection.update_one(
            {"user_id": geo_data["user_id"], "alert_id": geo_data["alert_id"]},
            {"$set": geo_data},
            upsert=True
        )
        print(f"[✓] Inserted geo data for user {geo_data['user_id']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error inserting geo data for user {geo_data['user_id']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

def insert_user_alert(user_id, alert_id, incident_probability, is_incident, timestamp, collection):
    """Insert or update user alert metadata in the specified collection."""
    try:
        collection.update_one(
            {"user_id": user_id, "alert_id": alert_id},
            {
                "$set": {
                    "alert_id": alert_id,
                    "incident_probability": float(incident_probability),
                    "is_incident": is_incident,
                    "last_alert_timestamp": timestamp
                }
            },
            upsert=True
        )
        print(f"[✓] Inserted user alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error inserting user alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")