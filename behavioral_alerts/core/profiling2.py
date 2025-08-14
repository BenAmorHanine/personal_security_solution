import numpy as np
import logging
from datetime import datetime, timezone
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from .config import MONGO_URI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def build_user_profile(user_id, collection=locations_collection):
    """Build a user profile using historical location and time data."""
    try:
        locations = list(collection.find({"user_id": user_id}))
        if not locations:
            raise ValueError("No location data for user")
        
        # Extract coordinates and times
        coords = np.array([[loc["latitude"] * 111, loc["longitude"] * 111] for loc in locations])
        times = [loc.get("timestamp", datetime.now(timezone.utc)) for loc in locations]
        
        # OPTICS clustering for locations
        max_eps = max(2.0, np.std(coords, axis=0).mean() * 1.5)  # Dynamic: at least 2 km
        optics = OPTICS(max_eps=max_eps, min_samples=5)
        clusters = optics.fit_predict(coords)
        centroids = []
        for label in set(clusters) - {-1}:
            cluster_points = coords[clusters == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append({
                "center": centroid.tolist(),
                "count": len(cluster_points),
                "hour_mean": np.mean([t.hour for t, c in zip(times, clusters) if c == label]),
                "weekday_mean": np.mean([t.weekday() for t, c in zip(times, clusters) if c == label]),
                "month_mean": np.mean([t.month for t, c in zip(times, clusters) if c == label])
            })
        
        # Coordinate scaler
        coord_scaler = StandardScaler()
        coord_scaler.fit(coords)
        
        # Time frequency distributions
        hour_freq = {}
        weekday_freq = {}
        month_freq = {}
        for t in times:
            hour_freq[t.hour] = hour_freq.get(t.hour, 0) + 1
            weekday_freq[t.weekday()] = weekday_freq.get(t.weekday(), 0) + 1
            month_freq[t.month] = month_freq.get(t.month, 0) + 1
        
        # Save to MongoDB
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "centroids": centroids,
                    "max_eps": max_eps,
                    "hour_freq": hour_freq,
                    "weekday_freq": weekday_freq,
                    "month_freq": month_freq,
                    "coord_scaler": coord_scaler.__dict__,
                    "last_updated": datetime.now(timezone.utc)
                }
            },
            upsert=True
        )
        
        logger.info(f"Built user profile for {user_id} with {len(centroids)} clusters, max_eps={max_eps}")
        return centroids, hour_freq, weekday_freq, month_freq, coord_scaler
    
    except Exception as e:
        logger.error(f"Error building profile for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return [], {}, {}, {}, None

def detect_user_anomalies(latitude, longitude, hour, weekday, month, user_id, collection=locations_collection):
    """Detect anomalies in user location and time."""
    try:
        user_doc = users_collection.find_one({"user_id": user_id})
        if not user_doc or "last_updated" not in user_doc or (datetime.now(timezone.utc) - user_doc["last_updated"] > timedelta(hours=24)):
            logger.info(f"User profile missing or outdated for {user_id}, rebuilding")
            centroids, hour_freq, weekday_freq, month_freq, coord_scaler = build_user_profile(user_id, collection)
            if not centroids:
                raise ValueError("Failed to build user profile")
        else:
            centroids = user_doc.get("centroids", [])
            max_eps = user_doc.get("max_eps", 2.0)
            hour_freq = user_doc.get("hour_freq", {})
            weekday_freq = user_doc.get("weekday_freq", {})
            month_freq = user_doc.get("month_freq", {})
            coord_scaler = StandardScaler()
            if "coord_scaler" in user_doc:
                coord_scaler.__dict__.update(user_doc["coord_scaler"])
        
        if not centroids:
            logger.warning(f"No centroids for user {user_id}, assuming anomaly")
            return 1.0, 1.0, 1.0, 1.0
        
        # Scale input coordinates
        input_coords = np.array([[latitude * 111, longitude * 111]])
        input_scaled = coord_scaler.transform(input_coords)
        
        # Calculate location anomaly
        min_distance = float("inf")
        for centroid in centroids:
            centroid_scaled = coord_scaler.transform([[centroid["center"][0], centroid["center"][1]]])
            distance = np.linalg.norm(input_scaled[:, :2] - centroid_scaled[:, :2])
            min_distance = min(min_distance, distance)
        location_anomaly = 1.0 - min(min_distance / max_eps, 1.0)  # Reversed: 1.0 = anomalous, 0.0 = normal
        
        # Time-based anomalies
        hour_anomaly = 1.0 - (hour_freq.get(hour, 0) / max(hour_freq.values(), 1.0))
        weekday_anomaly = 1.0 - (weekday_freq.get(weekday, 0) / max(weekday_freq.values(), 1.0))
        month_anomaly = 1.0 - (month_freq.get(month, 0) / max(month_freq.values(), 1.0))
        
        logger.debug(f"min_distance: {min_distance}, location_anomaly: {location_anomaly}")
        logger.info(f"Detected anomalies for user {user_id}: location={location_anomaly}, hour={hour_anomaly}, weekday={weekday_anomaly}, month={month_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly
    
    except Exception as e:
        logger.error(f"Error detecting anomalies for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return (1.0, 1.0, 1.0, 1.0)  # Anomalous for safety