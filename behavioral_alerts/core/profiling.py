from pymongo.collection import Collection
from sklearn.cluster import OPTICS, DBSCAN
from config import CLUSTERING_METHOD
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

"""This file builds a profile of each user based on their location and time history (e.g., usual zones, hours, weekdays...).
It uses clustering algorithms (OPTICS or DBSCAN) to identify common locations and temporal patterns."""



def preprocess_user_data(user_id, collection):
    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    if df.empty or len(df) < 10:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    return df

def extract_temporal_features(df):
    hour_freq = df["hour"].value_counts(normalize=True)
    weekday_freq = df["weekday"].value_counts(normalize=True)
    month_freq = df["month"].value_counts(normalize=True)
    return hour_freq, weekday_freq, month_freq

def build_user_profile_optics(user_id, collection):
    df = preprocess_user_data(user_id, collection)
    if df is None:
        return None, None, None, None, None

    coords = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    optics = OPTICS(min_samples=5, xi=0.1)
    labels = optics.fit_predict(coords_scaled)
    df["cluster"] = labels

    centroids = df[df["cluster"] != -1].groupby("cluster")[["latitude", "longitude"]].mean()
    hour_freq, weekday_freq, month_freq = extract_temporal_features(df)

    return centroids, hour_freq, weekday_freq, month_freq, scaler

def build_user_profile_dbscan(user_id, collection):
    df = preprocess_user_data(user_id, collection)
    if df is None:
        return None, None, None, None, None

    coords = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(coords_scaled)
    df["cluster"] = labels

    centroids = df[df["cluster"] != -1].groupby("cluster")[["latitude", "longitude"]].mean()
    hour_freq, weekday_freq, month_freq = extract_temporal_features(df)

    return centroids, hour_freq, weekday_freq, month_freq, scaler

def build_user_profile(user_id, collection, clustering_method=CLUSTERING_METHOD):
    if CLUSTERING_METHOD == "dbscan":
        return build_user_profile_dbscan(user_id, collection)
    elif CLUSTERING_METHOD == "optics":
        return build_user_profile_optics(user_id, collection)
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}, Use dbscan or optics.")


# In-memory cache for user profiles
user_profiles_cache = {}
cache_duration = timedelta(minutes=5)

@lru_cache(maxsize=1000)
def build_user_profile(user_id, data):
    """Build a user profile with location clusters and temporal features."""
    user_data = data[data['user_id'] == user_id]
    user_data = preprocess_user_data(user_data)
    
    # Cluster locations
    coords = user_data[['latitude', 'longitude']].values
    clustering = DBSCAN(eps=0.01, min_samples=5).fit(coords)
    user_data['location_cluster'] = clustering.labels_
    
    # Extract temporal features
    hour_freq = user_data['hour'].value_counts(normalize=True).to_dict()
    weekday_freq = user_data['weekday'].value_counts(normalize=True).to_dict()
    month_freq = user_data['month'].value_counts(normalize=True).to_dict()
    
    return {
        'clusters': user_data.groupby('location_cluster')[['latitude', 'longitude']].mean().to_dict(),
        'hour_freq': hour_freq,
        'weekday_freq': weekday_freq,
        'month_freq': month_freq
    }