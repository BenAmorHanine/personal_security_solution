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


def setup_timeseries_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    try:
        db.create_collection(
            "user_locations_ts",
            timeseries={
                "timeField": "timestamp",
                "metaField": "user_id",
                "granularity": "seconds"
            }
        )
    except Exception as e:
        print(f"Collection exists or error: {e}")
    return db["user_locations_ts"]

def setup_geospatial_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db["user_locations_geo"]
    collection.create_index([("location_index", "2dsphere")])
    return collection

def setup_users_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db["users"]

def insert_location(collection, user_id, lat, lon, timestamp):
    collection.insert_one({
        "user_id": user_id,
        "timestamp": timestamp,
        "latitude": lat,
        "longitude": lon
    })

def insert_geo_data(collection, user_id, lat, lon):
    collection.insert_one({
        "user_id": user_id,
        "location_index": {
            "type": "Point",
            "coordinates": [lon, lat]
        }
    })

def insert_user_alert(collection, user_id, location_anomaly, time_anomaly, is_incident):
    collection.update_one(
        {"user_id": user_id},
        {
            "$push": {
                "alert_history": {
                    "timestamp": datetime.now(),
                    "location_anomaly_score": location_anomaly,
                    "time_anomaly_score": time_anomaly,
                    "is_incident": is_incident
                }
            },
            "$set": {
                "model_metadata": {
                    "last_trained": datetime.now()
                }
            }
        },
        upsert=True
    )