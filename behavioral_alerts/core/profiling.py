
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .config import MONGO_URI, MODEL_DIR, DISTANCE_THRESHOLD
from .threshold_adjustment import adjust_threshold


client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def preprocess_user_data(user_id):
    """Extract and preprocess user location data from locations collection."""
    try:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        data = list(locations_collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": one_month_ago},
            "alert": {"$exists": False},
            "location": {"$exists": True}
        }))
        print(f"[DEBUG] Found {len(data)} records for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Filter valid location documents
        valid_data = []
        for doc in data:
            loc = doc.get("location")
            if isinstance(loc, dict) and loc.get("type") == "Point" and isinstance(loc.get("coordinates"), list) and len(loc["coordinates"]) == 2:
                try:
                    lat, lon = float(loc["coordinates"][1]), float(loc["coordinates"][0])
                    valid_data.append(doc)
                except (TypeError, ValueError):
                    print(f"[DEBUG] Invalid coordinates in document for user {user_id}: {doc}")
            else:
                print(f"[DEBUG] Invalid location document for user {user_id}: {doc}")
        
        print(f"[DEBUG] Valid records after filtering: {len(valid_data)}")
        if valid_data:
            print(f"[DEBUG] Sample valid document: {valid_data[0]}")
        if not valid_data or len(valid_data) < 10:
            raise ValueError(f"Insufficient data for user {user_id}: {len(valid_data)} records")
        
        df = pd.DataFrame(valid_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Invalid timestamps in data")
        df["latitude"] = df["location"].apply(lambda x: float(x["coordinates"][1]))
        df["longitude"] = df["location"].apply(lambda x: float(x["coordinates"][0]))
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0) / 3600
        return df
    except Exception as e:
        print(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def build_user_profile(user_id, locations_collection):
    """Build user behavior profile using clustering and frequency analysis."""
    try:
        df = preprocess_user_data(user_id)
        if df is None:
            return None, None, None, None, None
        coordinates = df[["latitude", "longitude"]].values
        scaler = StandardScaler()
        scaled_coordinates = scaler.fit_transform(coordinates)
        # Use max_eps suitable for scaled data
        clustering = OPTICS(max_eps=DISTANCE_THRESHOLD, min_samples=5).fit(scaled_coordinates)
        labels = clustering.labels_
        print(f"[DEBUG] Cluster labels: {np.unique(labels)}")
        centroids = []
        for cluster_id in np.unique(labels[labels != -1]):
            cluster_points = coordinates[labels == cluster_id]  # Use original coordinates for radius
            center = cluster_points.mean(axis=0)
            # Calculate radius in meters using Haversine distance approximation
            radius = np.max(np.sqrt(((cluster_points - center) ** 2).sum(axis=1))) * 111320  # Approx. degrees to meters
            centroids.append({
                "cluster_id": int(cluster_id),
                "center": {"type": "Point", "coordinates": [float(center[1]), float(center[0])]},
                "radius": float(radius)
            })
        print(f"[DEBUG] Found {len(centroids)} clusters for user {user_id}")
        hour_freq = df["hour"].value_counts(normalize=True).to_dict()
        weekday_freq = df["weekday"].value_counts(normalize=True).to_dict()
        month_freq = df["month"].value_counts(normalize=True).to_dict()
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "behavior_profile": {
                        "centroids": centroids,
                        "hour_freq": {str(k): float(v) for k, v in hour_freq.items()},
                        "weekday_freq": {str(k): float(v) for k, v in weekday_freq.items()},
                        "month_freq": {str(k): float(v) for k, v in month_freq.items()},
                        "last_updated": datetime.now(timezone.utc)
                    }
                }
            },
            upsert=True
        )
        user_dir = os.path.join(MODEL_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        joblib.dump(clustering, os.path.join(user_dir, f"{user_id}_optics_model.pkl"))
        joblib.dump(scaler, os.path.join(user_dir, f"{user_id}_optics_scaler.pkl"))
        return centroids, hour_freq, weekday_freq, month_freq, scaler
    except Exception as e:
        print(f"[✗] Error building profile for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None, None, None, None

def detect_user_anomalies(lat, lon, hour, weekday, month, user_id, collection=None, prob_threshold=0.5):
    """Detect anomalies in location and time for a user."""
    try:
        user = users_collection.find_one({"user_id": user_id})
        if not user or "behavior_profile" not in user:
            return 1.0, 1.0
        centroids = user["behavior_profile"].get("centroids", [])
        hour_freq = user["behavior_profile"].get("hour_freq", {})
        weekday_freq = user["behavior_profile"].get("weekday_freq", {})
        month_freq = user["behavior_profile"].get("month_freq", {})
        loc_anomaly = 1.0
        for zone in centroids:
            # Distance in degrees, converted to meters
            dist = np.sqrt((lat - zone["center"]["coordinates"][1])**2 + (lon - zone["center"]["coordinates"][0])**2) * 111320
            if dist < zone["radius"]:
                loc_anomaly = 0.0
                break
        time_anomaly = 1.0 - max(
            hour_freq.get(str(hour), 0.0),
            weekday_freq.get(str(weekday), 0.0),
            month_freq.get(str(month), 0.0)
        )
        return loc_anomaly, time_anomaly
    except Exception as e:
        print(f"[✗] Error detecting anomalies for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return 1.0, 1.0
