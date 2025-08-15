from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from config import MONGO_URI
from geopy.distance import geodesic

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


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
        logger.debug(f"Found {len(locations)} records for user {user_id}")
        if locations:
            #logger.debug(f"Location timestamps: {[loc['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z') for loc in locations[:5]]}")
            logger.debug(f"Sample locations: {[loc.get('location', {}) for loc in locations[:5]]}")        
        
        if len(locations) < 5:
            logger.error(f"Insufficient data for user {user_id}: {len(locations)} records")
            return None, None, None, None, None, None
        
        df = pd.DataFrame(locations)

        # Extract latitude and longitude from GeoJSON location field
        df["latitude"] = df["location"].apply(lambda x: x["coordinates"][1] if isinstance(x, dict) and "coordinates" in x and len(x["coordinates"]) == 2 else None)
        df["longitude"] = df["location"].apply(lambda x: x["coordinates"][0] if isinstance(x, dict) and "coordinates" in x and len(x["coordinates"]) == 2 else None)        
        
        # Check for missing or invalid coordinates
        invalid_coords = df["latitude"].isnull() | df["longitude"].isnull()
        if invalid_coords.any():
            #logger.error(f"Invalid coordinates for user {user_id}: {df[invalid_coords][['location']].to_dict('records')}")
            return None, None, None, None, None, None

        # Debug: sample coordinates before scaling (degrees)
        #print("[DEBUG] Sample coordinates before scaling (degrees):")
        #print(df[["latitude", "longitude"]].dropna().sample(min(20, len(df)), random_state=42))

        df["hour"] = df["timestamp"].apply(lambda x: x.hour)
        df["weekday"] = df["timestamp"].apply(lambda x: x.weekday())
        df["month"] = df["timestamp"].apply(lambda x: x.month)
        
        X = df[["latitude", "longitude", "hour", "weekday", "month"]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert integer keys to strings for MongoDB compatibility
        hour_freq = {str(k): v for k, v in df["hour"].value_counts(normalize=True).to_dict().items()}
        weekday_freq = {str(k): v for k, v in df["weekday"].value_counts(normalize=True).to_dict().items()}
        month_freq = {str(k): v for k, v in df["month"].value_counts(normalize=True).to_dict().items()}

        logger.debug(f"Preprocessed {len(X)} records for user {user_id}, coords range: lat={df['latitude'].min():.2f}-{df['latitude'].max():.2f}, lon={df['longitude'].min():.2f}-{df['longitude'].max():.2f}")
        return df, X_scaled, hour_freq, weekday_freq, month_freq, scaler
    
    except Exception as e:
        logger.error(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None, None, None, None, None

def build_user_profile(user_id, collection=locations_collection):
    """Build user movement profile using clustering (degrees-only pipeline)."""
    df, X_scaled, hour_freq, weekday_freq, month_freq, scaler = preprocess_data(user_id, collection)
    if df is None:
        return None, None, None, None, None

    # OPTICS clustering (eps in degrees)
    eps_degrees = 2.0 / 111  # ~2 km radius
    clustering = OPTICS(max_eps=eps_degrees, min_samples=3, metric="euclidean")
    clustering.fit(df[["latitude", "longitude"]].values)

    df["cluster"] = clustering.labels_

    centroids = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:  # Noise
            continue
        cluster_points = df[df["cluster"] == cluster_id]
        center_lat = cluster_points["latitude"].mean()
        center_lon = cluster_points["longitude"].mean()
        hour_mean = cluster_points["hour"].mean()
        weekday_mean = cluster_points["weekday"].mean()
        month_mean = cluster_points["month"].mean()

        centroids.append({
            "cluster_id": int(cluster_id),
            "center": (center_lat, center_lon),
            "hour_mean": hour_mean,
            "weekday_mean": weekday_mean,
            "month_mean": month_mean
        })

    logger.debug(f"Built {len(centroids)} centroids for user {user_id}")
    return centroids, hour_freq, weekday_freq, month_freq, scaler


def detect_user_anomalies(latitude, longitude, hour, weekday, month, user_id, collection=locations_collection):
    """Detect anomalies in user location and time."""
    try:
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id, collection)
        if centroids is None:
            print(f"[DEBUG] No behavior profile for user {user_id}, deferring profile creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return 1.0, 1.0, 1.0, 1.0
        
        input_data = np.array([[latitude, longitude, hour, weekday, month]])
        input_scaled = scaler.transform(input_data)
        
        min_distance_km = float("inf")
        print(f"[DEBUG] Number of centroids: {len(centroids)}")
        for centroid in centroids:
            # Keep centroid in scaled space for consistency
            centroid_scaled = scaler.transform([[centroid["center"][0], centroid["center"][1], centroid["hour_mean"], centroid["weekday_mean"], centroid["month_mean"]]])
            
            # Invert scaling to get raw lat/lon back
            raw_point = scaler.inverse_transform(input_scaled)[0][:2]
            raw_centroid = scaler.inverse_transform(centroid_scaled)[0][:2]
            
            # Compute geodesic distance in kilometers
            distance_km = geodesic((raw_point[0], raw_point[1]),
                                   (raw_centroid[0], raw_centroid[1])).kilometers
            
            #print(f"[DEBUG] Distance from centroid {centroid['cluster_id']}: {distance_km:.2f} km")
            min_distance_km = min(min_distance_km, distance_km)
        
        # Now threshold is in real kilometers
        location_anomaly = min(min_distance_km / 2.0, 1.0)  # 2.0 km threshold
        
        hour_anomaly = 1.0 - hour_freq.get(str(hour), 0)
        weekday_anomaly = 1.0 - weekday_freq.get(str(weekday), 0)
        month_anomaly = 1.0 - month_freq.get(str(month), 0)
        
        logger.info(f"[✓] Detected anomalies for user {user_id}: location={location_anomaly}, hour={hour_anomaly}, weekday={weekday_anomaly}, month={month_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        #logger.debug(f"Centroids length: {len(centroids)}")
        #logger.debug(f"Input coords: {input_data}, Input scaled: {input_scaled}")
        #logger.debug(f"Min distance (km): {min_distance_km}, Location anomaly: {location_anomaly}")
        
        return location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly
    
    except Exception as e:
        logger.error(f"[✗] Error detecting anomalies for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return (1.0, 1.0, 1.0, 1.0)



def load_scaler_from_profile(profile_doc):
    scaler = StandardScaler()
    scaler.mean_ = np.array(profile_doc["scaler_mean"])
    scaler.scale_ = np.array(profile_doc["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


