from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import joblib
from src.config import DISTANCE_THRESHOLD, DEFAULT_PROB_THRESHOLD
from src.threshold_adjustment import predict_threshold, load_threshold_model

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

@lru_cache(maxsize=1000)
def build_user_profile(user_id, collection, last_trained=None):
    df = preprocess_user_data(user_id, collection)
    if df is None:
        return None, None, None, None, None
    coords = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    optics = OPTICS(min_samples=5, xi=0.1) #tnjm tzid if clustering method is optics for these 2 lines
    labels = optics.fit_predict(coords_scaled)
    df["cluster"] = labels
    centroids = df[df["cluster"] != -1].groupby("cluster")[["latitude", "longitude"]].mean()
    hour_freq = df["hour"].value_counts(normalize=True)
    weekday_freq = df["weekday"].value_counts(normalize=True)
    month_freq = df["month"].value_counts(normalize=True)
    return centroids, hour_freq, weekday_freq, month_freq, scaler

def save_profile(user_id, optics, scaler):
    joblib.dump(optics, f"models/{user_id}_optics.pkl")
    joblib.dump(scaler, f"models/{user_id}_scaler.pkl")

def load_profile(user_id):
    try:
        optics = joblib.load(f"models/{user_id}_optics.pkl")
        scaler = joblib.load(f"models/{user_id}_scaler.pkl")
        return optics, scaler
    except FileNotFoundError:
        return None, None

def detect_user_anomalies(lat, lon, hour, weekday, month, user_id, collection, prob_threshold=None):
    profile = build_user_profile(user_id, collection)
    if profile[0] is None:
        return 0.0, 0.0
    centroids, hour_freq, weekday_freq, month_freq, scaler = profile
    if prob_threshold is None:
        threshold_model = load_threshold_model(user_id)
        if threshold_model:
            df = preprocess_user_data(user_id, collection)
            features = [df["hour"].std(), df["cluster"].diff().ne(0).mean(), len(df)]
            prob_threshold = predict_threshold(threshold_model, features)
        else:
            prob_threshold = DEFAULT_PROB_THRESHOLD
    loc_anomaly = 0.0
    coords = scaler.transform([[lat, lon]])
    for _, zone in centroids.iterrows():
        dist = np.sqrt((coords[0][0] - zone["latitude"])**2 + (coords[0][1] - zone["longitude"])**2)
        if dist < DISTANCE_THRESHOLD:
            break
        else:
            loc_anomaly = 1.0
    time_anomaly = 0.0
    hour_prob = hour_freq.get(hour, 0.01)
    weekday_prob = weekday_freq.get(weekday, 0.01)
    month_prob = month_freq.get(month, 0.01)
    if hour_prob < prob_threshold:
        time_anomaly += 0.5
    if weekday_prob < prob_threshold:
        time_anomaly += 0.3
    if month_prob < prob_threshold:
        time_anomaly += 0.2
    if hour_prob < hour_freq.quantile(0.05):  # Rare hour
        time_anomaly += 0.5
    return loc_anomaly, min(1.0, time_anomaly)

def should_retrain(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id})
    if last_trained is None or data_count > 1000 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False