from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from src.profiling import preprocess_user_data

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

def save_threshold_model(user_id, model):
    joblib.dump(model, f"models/{user_id}_rf_threshold.pkl")

def load_threshold_model(user_id):
    try:
        return joblib.load(f"models/{user_id}_rf_threshold.pkl")
    except FileNotFoundError:
        return None

def predict_threshold(model, features):
    return model.predict([features])[0]

def should_retrain_threshold(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id})
    if last_trained is None or data_count > 1000 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False