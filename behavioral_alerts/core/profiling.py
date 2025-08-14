from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from .config import MONGO_URI

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def preprocess_data_origin(user_id, collection=locations_collection):
    """Preprocess location data for profiling."""
    try:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=35)
        locations = list(collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": one_month_ago}
        }))
        logger.debug(f"Found {len(locations)} records for user {user_id}")
        if locations:
            logger.debug(f"Location timestamps: {[loc['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z') for loc in locations[:5]]}")
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
            logger.error(f"Invalid coordinates for user {user_id}: {df[invalid_coords][['location']].to_dict('records')}")
            return None, None, None, None, None, None
                


        # After extracting lat/lon but before scaling
        print("[DEBUG] Sample coordinates before scaling:")
        print(df[["latitude", "longitude"]].dropna().sample(20, random_state=42))

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
        print(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None, None, None, None, None


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
            logger.debug(f"Location timestamps: {[loc['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z') for loc in locations[:5]]}")
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
            logger.error(f"Invalid coordinates for user {user_id}: {df[invalid_coords][['location']].to_dict('records')}")
            return None, None, None, None, None, None

        # Debug: sample coordinates before scaling (degrees)
        print("[DEBUG] Sample coordinates before scaling (degrees):")
        print(df[["latitude", "longitude"]].dropna().sample(min(20, len(df)), random_state=42))

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
        print(f"[✗] Error preprocessing data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
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


def build_user_profile_before(user_id, collection=locations_collection):
    """Build a user profile using historical location and time data."""
    if not client:
        logger.error(f"No MongoDB connection for user {user_id}")
        return [], {}, {}, {}, None
    
    try:
        # Use preprocess_data to get coordinates and time features
        df, X_scaled, hour_freq, weekday_freq, month_freq, scaler = preprocess_data(user_id, collection)
        if df is None:
            logger.error(f"Failed to preprocess data for user {user_id}")
            return [], {}, {}, {}, None
        
        # Extract coordinates for clustering (latitude, longitude only)
        coords = X_scaled[:, :2]  # First two columns are latitude, longitude (scaled)
        coords_raw = df[["latitude", "longitude"]].values * 111  # Raw coords scaled by 111 for centroid centers
        times = df["timestamp"].tolist()
        
        logger.debug(f"Coords shape: {coords.shape}, Coords variance: {np.std(coords_raw, axis=0).mean()}")
        
        # OPTICS clustering on scaled coordinates
        min_samples = max(5, len(coords) // 10)
        max_eps = max(5.0, np.std(coords_raw, axis=0).mean() * 2.0)
        optics = OPTICS(max_eps=max_eps, min_samples=min_samples)
        clusters = optics.fit_predict(coords)
        
        # Check for valid clusters
        unique_labels = set(clusters) - {-1}
        if not unique_labels:
            logger.warning(f"No valid clusters for user {user_id}, using mean as single centroid")
            centroid = np.mean(coords_raw, axis=0)
            centroids = [{
                "center": centroid.tolist(),
                "count": len(coords),
               "hour_mean": float(np.mean([t.hour for t in times])),
                "weekday_mean": float(np.mean([t.weekday() for t in times])),
                "month_mean": float(np.mean([t.month for t in times]))            }]
        else:
            centroids = []
            for label in unique_labels:
                cluster_points = coords_raw[clusters == label]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append({
                    "center": centroid.tolist(),
                    "count": len(cluster_points),
                     "hour_mean": float(np.mean([t.hour for t, c in zip(times, clusters) if c == label])),
                    "weekday_mean": float(np.mean([t.weekday() for t, c in zip(times, clusters) if c == label])),
                    "month_mean": float(np.mean([t.month for t, c in zip(times, clusters) if c == label]))
                })
        
        logger.debug(f"Clusters: {clusters.tolist()}, Unique labels: {unique_labels}, Centroids length: {len(centroids)}, Centroid centers: {[c['center'] for c in centroids]}")
        print("[DEBUG] OPTICS labels (first 50):", unique_labels[:50])
        print("[DEBUG] Unique cluster labels:", set(unique_labels))
        print("[DEBUG] Centroids before saving:", centroids)

        # Save to MongoDB
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "centroids": centroids,
                    "max_eps": float(max_eps),
                    "hour_freq": hour_freq,
                    "weekday_freq": weekday_freq,
                    "month_freq": month_freq,
                    #"coord_scaler": scaler.__dict__,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "last_updated": datetime.now(timezone.utc)
                }
            },
            upsert=True
        )
        
        logger.info(f"Built user profile for {user_id} with {len(centroids)} clusters, max_eps={max_eps}, min_samples={min_samples}")
        return centroids, hour_freq, weekday_freq, month_freq, scaler
    
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection error for {user_id}: {e}")
        return [], {}, {}, {}, None
    except Exception as e:
        logger.error(f"Error building profile for {user_id} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}: {e}")
        return [], {}, {}, {}, None
    

def build_user_profile_original(user_id, collection=locations_collection, users_collection=users_collection):
    """Build user profile using OPTICS clustering."""
    try:
        result = preprocess_data(user_id, collection)
        if len(result) == 5:  # Handle insufficient data case
            print(f"[DEBUG] No profile built for user {user_id} due to insufficient data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None, None, None, None,None
        df, X_scaled, hour_freq, weekday_freq, month_freq, scaler = result
        print(f"[DEBUG] Data points for user {user_id}: {len(df)}")
        
        # OPTICS clustering
        coords_km = df[["latitude", "longitude"]] * 111
        std_km = np.std(coords_km.values, axis=0).mean()
        #max_eps = max(0.5, min(1.0, std_km * 1.5))  # Cap at 1.0 km
        #max_eps = 0.3 if len(df) < 5 else max(0.3, min(2.0, std_km * 1.2))
        max_eps = max(2.0, np.std(coords_km, axis=0).mean() * 1.5)  # At least 2 km
        print(f"[DEBUG] Computed max_eps={max_eps:.2f} km based on std={std_km:.2f} km")
        min_samples = max(3, len(coords_km) // 20) 
        optics = OPTICS(max_eps=max_eps, min_samples=min_samples)
        #optics = OPTICS(min_samples=2, max_eps=max_eps)
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
    """Detect anomalies in user location and time (degrees-only pipeline)."""
    try:
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id, collection)
        if centroids is None:
            print(f"[DEBUG] No behavior profile for user {user_id}, deferring profile creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return 1.0, 1.0, 1.0, 1.0

        input_data = np.array([[latitude, longitude, hour, weekday, month]])
        input_scaled = scaler.transform(input_data)

        min_distance_deg = float("inf")
        print(f"[DEBUG] Number of centroids: {len(centroids)}")
        for centroid in centroids:
            centroid_scaled = scaler.transform([[centroid["center"][0], centroid["center"][1],
                                                 centroid["hour_mean"], centroid["weekday_mean"], centroid["month_mean"]]])
            distance_deg = np.linalg.norm(input_scaled[:, :2] - centroid_scaled[:, :2])
            min_distance_deg = min(min_distance_deg, distance_deg)

            distance_km = distance_deg * 111
            print(f"[DEBUG] Distance from centroid {centroid['cluster_id']}: {distance_km:.2f} km")

        location_anomaly = min(min_distance_deg / (2.0 / 111), 1.0)  # normalized by ~2 km radius in degrees
        hour_anomaly = 1.0 - hour_freq.get(str(hour), 0)
        weekday_anomaly = 1.0 - weekday_freq.get(str(weekday), 0)
        month_anomaly = 1.0 - month_freq.get(str(month), 0)

        print(f"[✓] Detected anomalies for user {user_id}: location={location_anomaly}, hour={hour_anomaly}, weekday={weekday_anomaly}, month={month_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        logger.debug(f"Centroids length: {len(centroids)}")
        logger.debug(f"Input coords: {input_data}, Input scaled: {input_scaled}")
        logger.debug(f"Min distance (deg): {min_distance_deg}, Location anomaly: {location_anomaly}")

        return location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly

    except Exception as e:
        print(f"[✗] Error detecting anomalies for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return (1.0, 1.0, 1.0, 1.0)



def detect_user_anomalies_before(latitude, longitude, hour, weekday, month, user_id, collection=locations_collection):
    """Detect anomalies in user location and time."""
    try:
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id, collection)
        if centroids is None:
            print(f"[DEBUG] No behavior profile for user {user_id}, deferring profile creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return 1.0, 1.0, 1.0, 1.0
        
        input_data = np.array([[latitude, longitude, hour, weekday, month]])
        input_scaled = scaler.transform(input_data)
        
        min_distance = float("inf")
        print(f"[DEBUG] Number of centroids: {len(centroids)}")
        for centroid in centroids:
            centroid_scaled = scaler.transform([[centroid["center"][0], centroid["center"][1], centroid["hour_mean"], centroid["weekday_mean"], centroid["month_mean"]]])
            distance = np.linalg.norm(input_scaled[:, :2] - centroid_scaled[:, :2])
            print(f"[DEBUG] Distance from centroid {centroid['cluster_id']}: {distance:.2f} km")
            min_distance = min(min_distance, distance)
        
        location_anomaly = min(min_distance / 1.0, 1.0)
        #time_anomaly = 1.0 - (hour_freq.get(hour, 0) + weekday_freq.get(weekday, 0) + month_freq.get(month, 0)) / 3
        hour_anomaly = 1.0 - hour_freq.get(str(hour), 0)
        weekday_anomaly = 1.0 - weekday_freq.get(str(weekday), 0)
        month_anomaly = 1.0 - month_freq.get(str(month), 0)
        print(f"[✓] Detected anomalies for user {user_id}: location={location_anomaly}, hour={hour_anomaly}, weekday= {weekday_anomaly}, month= {month_anomaly} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        logger.debug(f"Centroids length: {len(centroids)}")#, Max_eps: {max_eps}, Coord_scaler n_features_in_: {getattr(coord_scaler, 'n_features_in_', 'unknown')}")
        logger.debug(f"Input coords: {input_data}, Input scaled: {input_scaled}")
        logger.debug(f"Min distance: {min_distance}, Location anomaly: {location_anomaly}")
        return location_anomaly,hour_anomaly, weekday_anomaly, month_anomaly #,time_anomaly
    except Exception as e:
        print(f"[✗] Error detecting anomalies for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return (1.0, 1.0, 1.0, 1.0)
    
def detect_user_anomalies_latest(latitude, longitude, hour, weekday, month, user_id, collection=locations_collection):
    """Detect anomalies in user location and time."""
    if not client:
        logger.error(f"No MongoDB connection for user {user_id}")
        return 1.0, 1.0, 1.0, 1.0
    try:
        user_doc = users_collection.find_one({"user_id": user_id})
        if not user_doc or "last_updated" not in user_doc or (datetime.now(timezone.utc) - user_doc["last_updated"].replace(tzinfo=timezone.utc) > timedelta(hours=24)):
            logger.info(f"User profile missing or outdated for {user_id}, rebuilding")
            centroids, hour_freq, weekday_freq, month_freq, coord_scaler = build_user_profile(user_id, collection)
            if not centroids:
                raise ValueError("Failed to build user profile")
        else:
            centroids = user_doc.get("centroids", [])
            max_eps = user_doc.get("max_eps", 5.0)  # Default to 5 km
            hour_freq = user_doc.get("hour_freq", {})
            weekday_freq = user_doc.get("weekday_freq", {})
            month_freq = user_doc.get("month_freq", {})
            coord_scaler = StandardScaler()
            """ 
           if "coord_scaler" in user_doc:
                coord_scaler.__dict__.update(user_doc["coord_scaler"])
                """
        
            if user_doc.get("coord_scaler"):
                scaler_dict = user_doc["coord_scaler"]
                coord_scaler.__dict__.update({
                    "with_mean": scaler_dict["with_mean"],
                    "with_std": scaler_dict["with_std"],
                    "copy": scaler_dict["copy"],
                    "n_features_in_": scaler_dict["n_features_in_"],
                    "n_samples_seen_": scaler_dict["n_samples_seen_"],
                    "mean_": np.array(scaler_dict["mean_"]),
                    "var_": np.array(scaler_dict["var_"]),
                    "scale_": np.array(scaler_dict["scale_"])
                })
            else:
                logger.warning(f"No scaler data for user {user_id}, rebuilding profile")
                centroids, hour_freq, weekday_freq, month_freq, coord_scaler = build_user_profile(user_id, collection)
                if not centroids:
                    raise ValueError("Failed to build user profile")
        
        if not centroids:
            logger.warning(f"No centroids for user {user_id}, assuming anomaly")
            return 1.0, 1.0, 1.0, 1.0
        
        # Scale input coordinates (2D: latitude, longitude)
        input_coords = np.array([[latitude * 111, longitude * 111]])
        try:
            input_scaled = coord_scaler.transform(input_coords)
        except ValueError as e:
            logger.error(f"Scaler error for {user_id}: {e}, rebuilding profile")
            centroids, hour_freq, weekday_freq, month_freq, coord_scaler = build_user_profile(user_id, collection)
            if not centroids:
                return 1.0, 1.0, 1.0, 1.0
            input_scaled = coord_scaler.transform(input_coords)
        
        # Calculate location anomaly
        min_distance = float("inf")
        for centroid in centroids:
            centroid_scaled = coord_scaler.transform([[centroid["center"][0], centroid["center"][1]]])
            distance = np.linalg.norm(input_scaled - centroid_scaled)
            min_distance = min(min_distance, distance)
        location_anomaly = 1.0 - min(min_distance / max_eps, 1.0)  # Reversed: 1.0 = anomalous, 0.0 = normal
        
        # Time-based anomalies
        max_hour = max(hour_freq.values(), default=1.0)
        max_weekday = max(weekday_freq.values(), default=1.0)
        max_month = max(month_freq.values(), default=1.0)
        hour_anomaly = 1.0 - (hour_freq.get(hour, 0) / max_hour)
        weekday_anomaly = 1.0 - (weekday_freq.get(weekday, 0) / max_weekday)
        month_anomaly = 1.0 - (month_freq.get(month, 0) / max_month)
        
        return location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly
    
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection error for {user_id}: {e}")
        return 1.0, 1.0, 1.0, 1.0
    except Exception as e:
        logger.error(f"Error detecting anomalies for {user_id} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}: {e}")
        return 1.0, 1.0, 1.0, 1.0

def detect_user_anomalies_the_new_one(latitude, longitude, hour, weekday, month, user_id, collection=locations_collection):
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
    z


from sklearn.preprocessing import StandardScaler
import numpy as np

def load_scaler_from_profile(profile_doc):
    scaler = StandardScaler()
    scaler.mean_ = np.array(profile_doc["scaler_mean"])
    scaler.scale_ = np.array(profile_doc["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler



"""
# Check for valid clusters
        unique_labels = set(clusters) - {-1}
        if not unique_labels:
            logger.warning(f"No valid clusters for user {user_id}, using mean as single centroid")
            centroid = np.mean(coords, axis=0)
            centroids = [{
                "center": centroid.tolist(),
                "count": len(coords),
                "hour_mean": np.mean([t.hour for t in times]),
                "weekday_mean": np.mean([t.weekday() for t in times]),
                "month_mean": np.mean([t.month for t in times])
            }]
        else:
            centroids = []
            for label in unique_labels:
                cluster_points = coords[clusters == label]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append({
                    "center": centroid.tolist(),
                    "count": len(cluster_points),
                    "hour_mean": np.mean([t.hour for t, c in zip(times, clusters) if c == label]),
                    "weekday_mean": np.mean([t.weekday() for t, c in zip(times, clusters) if c == label]),
                    "month_mean": np.mean([t.month for t, c in zip(times, clusters) if c == label])
                })
                
"""