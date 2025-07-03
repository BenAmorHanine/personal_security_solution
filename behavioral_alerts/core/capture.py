from datetime import datetime
import pandas as pd
from pymongo.collection import Collection
from .anomalies import detect_user_anomalies

def capture_and_store(user_id: str, lat: float, lon: float, collection: Collection):
    now = datetime.now()
    data = {
        'user_id': user_id,
        'latitude': lat,
        'longitude': lon,
        'timestamp': now,
        'hour': now.hour,
        'weekday': now.weekday(),
        'month': now.month
    }
    collection.insert_one(data)
    return now

def process_capture(user_id: str, lat: float, lon: float, collection: Collection):
    now = capture_and_store(user_id, lat, lon, collection)
    return detect_user_anomalies(lat, lon, now.hour, now.weekday(), now.month, user_id, collection)
