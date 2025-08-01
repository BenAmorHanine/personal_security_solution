from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from .config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def preprocess_data(user_id, collection=locations_collection):
    """Preprocess location data for profiling."""
    try:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=35)
        locations = list(collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": one_month_ago}
        }))
        print(f"[DEBUG] Found {len(locations)} records for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        if locations:
            print(f"[DEBUG] Location timestamps: {[loc['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z') for loc in locations]}")
        
        if len(locations) < 5:
            print(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: Insufficient data for user {user_id}: {len(locations)} records")
            return None, None, None, None, None
        
        df = pd.DataFrame(locations)
        # Extract latitude and longitude from GeoJSON location field
        df["latitude"] = df["location"].apply(lambda x: x["coordinates"][1] if isinstance(x, dict) and "coordinates" in x else None)
        df["longitude"] = df["location"].apply(lambda x: x["coordinates"][0] if isinstance(x, dict) and "coordinates" in x else None)
        
        # Check for missing coordinates
        if df["latitude"].isnull().any() or df["longitude"].isnull().any():
            print(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: Missing latitude or longitude values")
            return None, None, None, None, None
        
        df["hour"] = df["timestamp"].apply(lambda x: x.hour)
        df["weekday"] = df["timestamp"].apply(lambda x: x.weekday())
        df["month"] = df["timestamp"].apply(lambda x: x.month)
        
        X = df[["latitude", "longitude", "hour", "weekday", "month"]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        hour_freq = df["hour"].value_counts(normalize=True).to_dict()
        weekday_freq = df["weekday"].value_counts(normalize=True).to_dict()
        month_freq = df["month"].value_counts(normalize=True).to_dict()
        
        print(f"[DEBUG] Preprocessed {len(X)} records for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return df, X_scaled, hour_freq, weekday_freq, month_freq, scaler
    except Exception as e:
        print(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None, None, None, None



def build_user_profile(user_id, collection=locations_collection, users_collection=users_collection):
    """Build user profile using OPTICS clustering."""
    try:
        result = preprocess_data(user_id, collection)
        if len(result) == 5:  # Handle insufficient data case
            print(f"[DEBUG] No profile built for user {user_id} due to insufficient data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None, None, None, None
        df, X_scaled, hour_freq, weekday_freq, month_freq, scaler = result
        print(f"[DEBUG] Data points for user {user_id}: {len(df)}")
        
        # OPTICS clustering
        coords_km = df[["latitude", "longitude"]] * 111
        std_km = np.std(coords_km.values, axis=0).mean()
        #max_eps = max(0.5, min(1.0, std_km * 1.5))  # Cap at 1.0 km
        max_eps = 0.3 if len(df) < 5 else max(0.3, min(2.0, std_km * 1.2))
        print(f"[DEBUG] Computed max_eps={max_eps:.2f} km based on std={std_km:.2f} km")
        optics = OPTICS(min_samples=2, max_eps=max_eps)
        labels = optics.fit_predict(X_scaled)
        print(f"[DEBUG] Cluster labels: {labels.tolist()}")
        
        unique_labels = set(labels) - {-1}
        centroids = []
        for cluster_id in unique_labels:
            cluster_points = X_scaled[labels == cluster_id]
            cluster_df = df[labels == cluster_id]
            centroid = {
                "cluster_id": int(cluster_id),
                "center": cluster_points[:, :2].mean(axis=0).tolist(),
                "size": len(cluster_points),
                "hour_mean": cluster_df["hour"].mean(),
                "weekday_mean": cluster_df["weekday"].mean(),
                "month_mean": cluster_df["month"].mean()
            }
            centroids.append(centroid)
        
        print(f"[DEBUG] Found {len(centroids)} clusters for user {user_id}")
        for c in centroids:
            print(f"[DEBUG] Cluster {c['cluster_id']}: size={c['size']}, center={c['center']}, hour_mean={c['hour_mean']:.2f}")
        # Convert numeric keys to strings for MongoDB
        hour_freq_str = {str(k): float(v) for k, v in hour_freq.items()}
        weekday_freq_str = {str(k): float(v) for k, v in weekday_freq.items()}
        month_freq_str = {str(k): float(v) for k, v in month_freq.items()}
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": 
             {"profile": 
                {"centroids": centroids, 
                 "hour_freq": hour_freq_str, 
                 "weekday_freq": weekday_freq_str, 
                 "month_freq": month_freq_str}}}
        )
        print(f"[✓] Built user profile: {len(centroids)} clusters at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        return centroids, hour_freq, weekday_freq, month_freq, scaler
    except Exception as e:
        print(f"[✗] Error building profile for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None, None, None, None

def detect_user_anomalies(latitude, longitude, hour, weekday, month, user_id, collection=locations_collection):
    """Detect anomalies in user location and time."""
    try:
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id, collection)
        if centroids is None:
            print(f"[DEBUG] No behavior profile for user {user_id}, deferring profile creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return 1.0, 1.0
        
        input_data = np.array([[latitude, longitude, hour, weekday, month]])
        input_scaled = scaler.transform(input_data)
        
        min_distance = float("inf")
        for centroid in centroids:
            centroid_scaled = scaler.transform([[centroid["center"][0], centroid["center"][1], centroid["hour_mean"], centroid["weekday_mean"], centroid["month_mean"]]])
            distance = np.linalg.norm(input_scaled[:, :2] - centroid_scaled[:, :2])
            min_distance = min(min_distance, distance)
        
        location_anomaly = min(min_distance / 1.0, 1.0)
        time_anomaly = 1.0 - (hour_freq.get(hour, 0) + weekday_freq.get(weekday, 0) + month_freq.get(month, 0)) / 3
        print(f"[✓] Detected anomalies for user {user_id}: location={location_anomaly}, time={time_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return location_anomaly, time_anomaly
    except Exception as e:
        print(f"[✗] Error detecting anomalies for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return 1.0, 1.0