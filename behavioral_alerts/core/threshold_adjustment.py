from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
from .config import MONGO_URI, MODEL_DIR, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def adjust_threshold(user_id, collection=locations_collection, users_collection=users_collection):
    """Adjust anomaly detection threshold using RandomForestRegressor."""
    try:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        alerts = list(collection.find({
            "user_id": user_id,
            "alert": {"$exists": True},
            "timestamp": {"$gte": one_month_ago}
        }))
        print(f"[DEBUG] Preprocessed {len(alerts)} alerts for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        if len(alerts) < 2:
            print(f"[DEBUG] Insufficient alerts for user {user_id}: {len(alerts)} records, using default threshold {DEFAULT_PROB_THRESHOLD} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return DEFAULT_PROB_THRESHOLD
        
        features = []
        target = []
        for alert in alerts:
            features.append([
                alert["alert"]["location_anomaly_score"],
                alert["alert"]["time_anomaly_score"],
                alert["alert"]["ai_score"]
            ])
            target.append(alert["alert"]["ai_score"])
        
        # Convert to DataFrame for feature names
        feature_df = pd.DataFrame(features, columns=["location_anomaly", "time_anomaly", "ai_score"])
        target = np.array(target)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(feature_df, target)
        
        mse = mean_squared_error(target, rf_model.predict(feature_df))
        print(f"[DEBUG] Threshold model MSE: {mse} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Save model locally
        user_dir = os.path.join(MODEL_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        model_path = os.path.join(user_dir, f"{user_id}_rf_threshold.pkl")
        joblib.dump(rf_model, model_path)
        print(f"[✓] Saved threshold model locally for {user_id} at {model_path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Save to MongoDB
        model_bytes = joblib.dump(rf_model, "memory")[0]
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"ml_threshold_model": model_bytes}},
            upsert=True
        )
        print(f"[✓] Saved threshold model for {user_id} to MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Predict threshold
        latest_alert = alerts[-1]
        input_data = pd.DataFrame([[
            latest_alert["alert"]["location_anomaly_score"],
            latest_alert["alert"]["time_anomaly_score"],
            latest_alert["alert"]["ai_score"]
        ]], columns=["location_anomaly", "time_anomaly", "ai_score"])
        threshold = rf_model.predict(input_data)[0]
        threshold = max(0.1, min(threshold, 0.9))  # Bound between 0.1 and 0.9
        print(f"[✓] Predicted threshold for user {user_id}: {threshold} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return threshold
    except Exception as e:
        print(f"[✗] Error adjusting threshold for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return DEFAULT_PROB_THRESHOLD