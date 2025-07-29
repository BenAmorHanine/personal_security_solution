
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import joblib
import os
import io
import base64
from .config import MODEL_DIR, DEFAULT_PROB_THRESHOLD

def extract_threshold_features(df):
    return {
        "hour_std": df["hour"].std(),
        "location_transition_freq": df["cluster"].diff().ne(0).mean() if "cluster" in df else 0,
        "data_volume": len(df),
        "alert_frequency": df["alert"].notnull().mean() if "alert" in df else 0.0
    }

def prepare_threshold_data(collection, user_id):
    from .profiling import preprocess_user_data
    df = preprocess_user_data(user_id, collection)
    if df is None:
        return None, None
    features = extract_threshold_features(df)
    feature_array = np.array([[features["hour_std"], features["location_transition_freq"], features["data_volume"]]])
    target_threshold = 0.05 + features["hour_std"] * 0.01 + features["alert_frequency"] * 0.1
    return feature_array, [target_threshold]

def train_threshold_model(features, targets):
    try:
        model = RandomForestRegressor(random_state=42)
        model.fit(features, targets)
        from sklearn.metrics import mean_squared_error
        predictions = model.predict(features)
        mse = mean_squared_error(targets, predictions)
        print(f"[DEBUG] Threshold model MSE: {mse} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return model
    except Exception as e:
        print(f"[✗] Error training threshold model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None

def save_threshold_model(user_id, model, save_to_mongo=True, users_collection=None, save_local=True):
    try:
        if save_to_mongo and users_collection is not None:
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            encoded_model = base64.b64encode(buffer.read()).decode('utf-8')
            users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "threshold_model": {
                            "model": encoded_model,
                            "saved_at": datetime.now(timezone.utc)
                        }
                    }
                },
                upsert=True
            )
            print(f"[✓] Saved threshold model for {user_id} to MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        if save_local:
            user_dir = os.path.join(MODEL_DIR, user_id)
            os.makedirs(user_dir, exist_ok=True)
            joblib.dump(model, os.path.join(user_dir, f"{user_id}_threshold_model.pkl"))
            print(f"[✓] Saved threshold model locally for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error saving threshold model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

def load_threshold_model(user_id):
    try:
        user_dir = os.path.join(MODEL_DIR, user_id)
        model = joblib.load(os.path.join(user_dir, f"{user_id}_threshold_model.pkl"))
        print(f"[✓] Loaded threshold model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return model
    except FileNotFoundError:
        print(f"[✗] Threshold model not found for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return None

def predict_threshold(model, features):
    try:
        return model.predict([features])[0]
    except Exception as e:
        print(f"[✗] Error predicting threshold at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return DEFAULT_PROB_THRESHOLD

def should_retrain_threshold(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id})
    now = datetime.now(timezone.utc)
    # Ensure last_trained is offset-aware
    if last_trained and last_trained.tzinfo is None:
        last_trained = last_trained.replace(tzinfo=timezone.utc)
    if last_trained is None or data_count > 1000 or (now - last_trained > timedelta(days=30)):
        print(f"[DEBUG] Retraining threshold for {user_id}: data_count={data_count}, last_trained={last_trained} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return True
    return False

def adjust_threshold(user_id, collection):
    """Adjust threshold for a user based on historical data."""
    try:
        user = collection.database["users"].find_one({"user_id": user_id})
        last_trained = user.get("threshold_model", {}).get("saved_at") if user else None
        if should_retrain_threshold(collection, user_id, last_trained):
            features, targets = prepare_threshold_data(collection, user_id)
            if features is None or targets is None:
                print(f"[✗] Failed to prepare threshold data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return DEFAULT_PROB_THRESHOLD
            model = train_threshold_model(features, targets)
            if model is None:
                print(f"[✗] Failed to train threshold model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return DEFAULT_PROB_THRESHOLD
            save_threshold_model(user_id, model, save_to_mongo=True, users_collection=collection.database["users"])
        else:
            model = load_threshold_model(user_id)
            if model is None:
                print(f"[✗] No threshold model available for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return DEFAULT_PROB_THRESHOLD
        features, _ = prepare_threshold_data(collection, user_id)
        if features is None:
            print(f"[✗] Failed to prepare features for prediction for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return DEFAULT_PROB_THRESHOLD
        threshold = predict_threshold(model, features[0])
        print(f"[DEBUG] Predicted threshold for {user_id}: {threshold} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return threshold
    except Exception as e:
        print(f"[✗] Error adjusting threshold for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return DEFAULT_PROB_THRESHOLD