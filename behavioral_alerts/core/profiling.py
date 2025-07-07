from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
import base64
import io
from pymongo.collection import Collection
from .config import DISTANCE_THRESHOLD, DEFAULT_PROB_THRESHOLD, CLUSTERING_METHOD
from functools import lru_cache
from .threshold_adjustment import load_threshold_model, predict_threshold


# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_user_data(user_id, collection: Collection):
    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    if df.empty or len(df) < 10:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0) / 3600
    return df

# ------------------------------
# Build Profile
# ------------------------------
@lru_cache(maxsize=1000)
def build_user_profile(user_id, collection: Collection, clustering_method=CLUSTERING_METHOD, last_trained=None, save_to_mongo=False):
    df = preprocess_user_data(user_id, collection)
    if df is None:
        return None, None, None, None, None

    coords = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    if clustering_method.lower() == "optics":
        clustering_model = OPTICS(min_samples=5, xi=0.1)
    else:
        raise ValueError("Unsupported clustering method")

    labels = clustering_model.fit_predict(coords_scaled)
    df["cluster"] = labels

    centroids = df[df["cluster"] != -1].groupby("cluster")[["latitude", "longitude"]].mean()
    hour_freq = df["hour"].value_counts(normalize=True).to_dict()
    weekday_freq = df["weekday"].value_counts(normalize=True).to_dict()
    month_freq = df["month"].value_counts(normalize=True).to_dict()

    save_profile(user_id, clustering_model, scaler, save_to_mongo, collection.database["users"])
    save_profile_to_db(user_id, centroids, hour_freq, weekday_freq, month_freq, collection.database["users"])

    return centroids, hour_freq, weekday_freq, month_freq, scaler

# ------------------------------
# Save/Load Profile to Disk or MongoDB
# ------------------------------
def save_profile(user_id, model, scaler, save_to_mongo=False, users_collection=None, save_local=True):
    if save_to_mongo and users_collection is not None:
        save_model_to_mongodb(user_id, model, scaler, users_collection)
    if save_local:
        user_dir = os.path.join("behavioral_alerts", "models", user_id)
        os.makedirs(user_dir, exist_ok=True)
        joblib.dump(model, os.path.join(user_dir, f"{user_id}_optics.pkl"))
        joblib.dump(scaler, os.path.join(user_dir, f"{user_id}_scaler.pkl"))
        print(f"[✓] Saved model locally for {user_id}")


def load_profile(user_id):
    user_dir = os.path.join("behavioral_alerts", "models", user_id)
    try:
        model = joblib.load(os.path.join(user_dir, f"{user_id}_optics.pkl"))
        scaler = joblib.load(os.path.join(user_dir, f"{user_id}_scaler.pkl"))
        return model, scaler
    except FileNotFoundError:
        print(f"[✗] Profile not found for {user_id}")
        return None, None

# ------------------------------
# Save Profile Summary to DB
# ------------------------------
def save_profile_to_db(user_id, centroids, hour_freq, weekday_freq, month_freq, users_collection: Collection):
    hour_freq_str = {str(k): v for k, v in hour_freq.items()}
    weekday_freq_str = {str(k): v for k, v in weekday_freq.items()}
    month_freq_str = {str(k): v for k, v in month_freq.items()}

    users_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "behavior_profile": {
                    "centroids": centroids.reset_index().to_dict(orient="records"),
                    "hour_freq": hour_freq_str,
                    "weekday_freq": weekday_freq_str,
                    "month_freq": month_freq_str,
                    "last_updated": datetime.utcnow()
                }
            }
        },
        upsert=True
    )
    print(f"[✓] Cached profile in DB for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

# ------------------------------
# Load Profile Summary from DB
# ------------------------------
def load_profile_from_db(user_id, users_collection):
    doc = users_collection.find_one({"user_id": user_id})
    if not doc or "behavior_profile" not in doc:
        return None

    profile = doc["behavior_profile"]
    hour_freq = {int(k): v for k, v in profile["hour_freq"].items()}
    weekday_freq = {int(k): v for k, v in profile["weekday_freq"].items()}
    month_freq = {int(k): v for k, v in profile["month_freq"].items()}
    centroids = pd.DataFrame(profile["centroids"])

    return centroids, hour_freq, weekday_freq, month_freq

# ------------------------------
# Save/Load Full Model to MongoDB (binary)
# ------------------------------
def save_model_to_mongodb(user_id: str, model, scaler, users_collection: Collection):
    def serialize(obj):
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    model_blob = serialize(model)
    scaler_blob = serialize(scaler)

    users_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "ml_model": {
                    "optics_model": model_blob,
                    "scaler": scaler_blob,
                    "saved_at": datetime.utcnow()
                }
            }
        },
        upsert=True
    )
    print(f"[✓] Saved ML model for {user_id} in MongoDB")

def load_model_from_mongodb(user_id: str, users_collection: Collection):
    doc = users_collection.find_one({"user_id": user_id})
    if not doc or "ml_model" not in doc:
        print(f"[✗] No ML model found in DB for {user_id}")
        return None, None

    def deserialize(encoded_str):
        buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
        return joblib.load(buffer)

    try:
        model = deserialize(doc["ml_model"]["optics_model"])
        scaler = deserialize(doc["ml_model"]["scaler"])
        return model, scaler
    except Exception as e:
        print(f"[✗] Failed to load model for {user_id}: {e}")
        return None, None

# ------------------------------
# Anomaly Detection
# ------------------------------
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


# ------------------------------
# Retraining Check
# ------------------------------
def should_retrain(collection, user_id, last_trained):
  data_count = collection.count_documents({"user_id": user_id})
  recent_data = collection.count_documents({"user_id": user_id, "timestamp": {"$gt": datetime.now() - timedelta(days=7)}})
  return (last_trained is None or data_count > 1000 or 
          datetime.now() - last_trained > timedelta(days=30) or 
          recent_data > 0.2 * data_count)  # Retrain if >20% new data
