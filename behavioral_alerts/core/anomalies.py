from .profiling import build_user_profile
from pymongo.collection import Collection
import numpy as np
from datetime import datetime

""""DISTANCE_THRESHOLD = 0.05
PROB_THRESHOLD = 0.05
LATE_NIGHT_HOURS = list(range(22, 24)) + list(range(0, 5))
""""

from ../config import DISTANCE_THRESHOLD, PROB_THRESHOLD, LATE_NIGHT_HOURS

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
    if hour in LATE_NIGHT_HOURS:
        time_anomaly += 0.5

    return loc_anomaly, min(1.0, time_anomaly)
