from pymongo import MongoClient
from datetime import datetime
from .config import MONGO_URI, DB_NAME

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