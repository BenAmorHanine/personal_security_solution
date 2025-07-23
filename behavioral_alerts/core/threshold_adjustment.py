from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pymongo.collection import Collection
from .config import MODEL_DIR, DEFAULT_PROB_THRESHOLD
def extract_threshold_features(df):
    return {
        "hour_std": df["hour"].std(),
        "location_transition_freq": df["cluster"].diff().ne(0).mean() if "cluster" in df else 0,
        "data_volume": len(df),
        "alert_frequency": df["alert"].notnull().mean()
    }

def prepare_threshold_data(collection, user_id):
    from .profiling import preprocess_user_data
    #No retraining inside it — 
    # if the model is outdated, you should call maybe_retrain_user_profile() 
    # explicitly before.
    df = preprocess_user_data(user_id, collection)#, should_retrain, build_user_profile)
    if df is None:
        return None, None
    features = extract_threshold_features(df)
    feature_array = np.array([[features["hour_std"], features["location_transition_freq"], features["data_volume"]]])
    target_threshold = 0.05 + features["hour_std"] * 0.01 + features["alert_frequency"] * 0.1  # Placeholder rule
    return feature_array, [target_threshold]

def train_threshold_model(features, targets):
    try:
        model = RandomForestRegressor(random_state=42)
        model.fit(features, targets)
        # Validate model performance
        from sklearn.metrics import mean_squared_error
        predictions = model.predict(features)
        mse = mean_squared_error(targets, predictions)
        print(f"Threshold model MSE: {mse}")
        return model
    except Exception as e:
        print(f"Error training threshold model: {e}")
        return None

import joblib
import os
import base64
import io
from datetime import datetime

def save_threshold_model(user_id, model, save_to_mongo=True, users_collection=None, save_local=True):
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
        #user_dir = os.path.join("behavioral_alerts", "models", user_id)
        #user_dir = os.path.join("..","SOLUTION_SECURITE_PERSO" ,"models", user_id)
        user_dir = os.path.join(MODEL_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        joblib.dump(model, os.path.join(user_dir, f"{user_id}_threshold_model.pkl"))
        print(f"[✓] Saved threshold model locally for {user_id}")


def load_threshold_model(user_id):
    try:
        user_dir = os.path.join(MODEL_DIR, user_id)
        return joblib.load(os.path.join(user_dir, f"{user_id}_threshold_model.pkl"))
        #return joblib.load(f"models/{user_id}_threshold_model.pkl")
    except FileNotFoundError:
        print(f"[✗] Threshold model not found for {user_id}")
        return None


def predict_threshold(model, features):
    try:
        return model.predict([features])[0]
    except Exception as e:
        print(f"Error predicting threshold: {e}")
        return DEFAULT_PROB_THRESHOLD

def should_retrain_threshold(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id})
    if last_trained is None or data_count > 1000 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False