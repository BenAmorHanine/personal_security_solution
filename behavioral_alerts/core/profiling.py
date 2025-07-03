from pymongo.collection import Collection
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def build_user_profile_optics(user_id, collection):
    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    if df.empty or len(df) < 10:
        return None, None, None, None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    locations = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(locations)

    optics = OPTICS(min_samples=5, xi=0.1)
    labels = optics.fit_predict(X_scaled)
    df["cluster"] = labels

    centroids = df[df["cluster"] != -1].groupby("cluster")[["latitude", "longitude"]].mean()
    hour_freq = df["hour"].value_counts(normalize=True)
    weekday_freq = df["weekday"].value_counts(normalize=True)
    month_freq = df["month"].value_counts(normalize=True)

    return centroids, hour_freq, weekday_freq, month_freq, scaler


def build_user_profile_dbscan(user_id, collection):
    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    if df.empty or len(df) < 10:
        return None, None, None, None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    locations = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(locations)

    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    df["cluster"] = labels

    centroids = df[df["cluster"] != -1].groupby("cluster")[["latitude", "longitude"]].mean()
    hour_freq = df["hour"].value_counts(normalize=True)
    weekday_freq = df["weekday"].value_counts(normalize=True)
    month_freq = df["month"].value_counts(normalize=True)

    return centroids, hour_freq, weekday_freq, month_freq, scaler
