from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from .utils import preprocess_user_data
from .utils import serialize_model, deserialize_model

def extract_threshold_features(df):
    return {
        "hour_std": df["hour"].std(),
        "location_transition_freq": df["cluster"].diff().ne(0).mean() if "cluster" in df else 0,
        "data_volume": len(df)
    }

def prepare_threshold_data(collection, user_id):
    df = preprocess_user_data(user_id, collection)
    if df is None:
        return None, None
    features = extract_threshold_features(df)
    feature_array = np.array([[features["hour_std"], features["location_transition_freq"], features["data_volume"]]])
    target_threshold = 0.05 + features["hour_std"] * 0.01  # Placeholder rule
    return feature_array, [target_threshold]

def train_threshold_model(features, targets):
    model = RandomForestRegressor(random_state=42)
    model.fit(features, targets)
    return model

import joblib
import os
import base64
import io
from datetime import datetime

def save_threshold_model(user_id, model, save_to_mongo=False, users_collection=None, save_local=True):
    if save_to_mongo and users_collection is not None:
        # Serialize and save to MongoDB
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
                        "saved_at": datetime.utcnow()
                    }
                }
            },
            upsert=True
        )
        print(f"[✓] Saved threshold model for {user_id} to MongoDB")

    if save_local:
        user_dir = os.path.join("behavioral_alerts", "models", user_id)
        os.makedirs(user_dir, exist_ok=True)
        joblib.dump(model, os.path.join(user_dir, f"{user_id}_threshold_model.pkl"))
        print(f"[✓] Saved threshold model locally for {user_id}")

def load_threshold_model(user_id):
    try:
        return joblib.load(f"models/{user_id}_threshold_model.pkl")
    except FileNotFoundError:
        return None

def predict_threshold(model, features):
    return model.predict([features])[0]

def should_retrain_threshold(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id})
    if last_trained is None or data_count > 1000 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False