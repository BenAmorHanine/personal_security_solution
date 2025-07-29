from pymongo import MongoClient
from datetime import datetime
from .config import MONGO_URI, DB_NAME
import pandas as pd

# behavioral_alerts/core/data_preprocessing.py

import pandas as pd
from datetime import datetime

"""def preprocess_user_data(user_id, collection, retrain_check_func, profile_builder):
    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    
    if df.empty or len(df) < 10:
        return None
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0) / 3600

    # Use injected retraining check and profile builder
    if 'cluster' not in df.columns or retrain_check_func(collection, user_id, None):
        df, _, _ = profile_builder(user_id, collection, save_to_mongo=True)
    
    return df
#WHEN CALLING: preprocess_user_data(user_id, collection, should_retrain, build_user_profile)
"""

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


# model_utils.py
import base64
import joblib
import io

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