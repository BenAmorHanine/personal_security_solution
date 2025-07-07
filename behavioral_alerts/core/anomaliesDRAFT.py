from .profiling import build_user_profile
from .config import DISTANCE_THRESHOLD, PROB_THRESHOLD, LATE_NIGHT_HOURS , CLUSTERING_METHOD
from pymongo.collection import Collection
import numpy as np
from datetime import datetime

"""This file detects whether a new user activity is unusual, by comparing it to their profile built by profiling.py."""





def detect_user_anomalies(lat, lon, hour, weekday, month, user_id, collection: Collection):
    profile = build_user_profile(user_id, collection)
    if profile[0] is None:
        return 0.0, 0.0

    centroids, hour_freq, weekday_freq, month_freq, _ = profile

    # Location anomaly
    loc_anomaly = 0.0
    for _, zone in centroids.iterrows():
        dist = np.sqrt((lat - zone['latitude'])**2 + (lon - zone['longitude'])**2)
        if dist < DISTANCE_THRESHOLD:
            break
    else:
        loc_anomaly = 1.0

    # Time anomaly
    time_anomaly = 0.0
    hour_prob = hour_freq.get(hour, 0.01)
    weekday_prob = weekday_freq.get(weekday, 0.01)
    month_prob = month_freq.get(month, 0.01)

    if hour_prob < PROB_THRESHOLD:
        time_anomaly += 0.5
    if weekday_prob < PROB_THRESHOLD:
        time_anomaly += 0.3
    if month_prob < PROB_THRESHOLD:
        time_anomaly += 0.2
    """if hour in LATE_NIGHT_HOURS:
        time_anomaly += 0.5"""
    # Replace hardcoded late-night logic with user-specific rarity
    if hour_freq.get(hour, 0) < hour_freq.quantile(0.05):  # rare hour for this user
        time_anomaly += 0.5

    return loc_anomaly, min(1.0, time_anomaly)


from datetime import datetime, timedelta

def should_retrain(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id})
    if data_count > 1000 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False

import joblib

def save_profile(user_id, optics, scaler):
    joblib.dump(optics, f"models/{user_id}_optics.pkl")
    joblib.dump(scaler, f"models/{user_id}_scaler.pkl")

def load_profile(user_id):
    optics = joblib.load(f"models/{user_id}_optics.pkl")
    scaler = joblib.load(f"models/{user_id}_scaler.pkl")
    return optics, scaler

"""from .autoencoder_module import compute_anomaly_score_autoencoder
score = compute_anomaly_score_autoencoder(user_id, lat, lon, datetime.now())
if score > YOUR_THRESHOLD:
    trigger_alert()
"""