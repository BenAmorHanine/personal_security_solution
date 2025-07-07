from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from .utils import serialize_model, deserialize_model

def prepare_incident_data(collection, user_id):
    user_data = collection.find_one({"user_id": user_id})
    if not user_data or "alert_history" not in user_data:
        return None, None
    df = pd.DataFrame(user_data["alert_history"])
    if df.empty or len(df) < 10:
        return None, None
    features = df[["location_anomaly_score", "time_anomaly_score"]]
    labels = df["is_incident"]
    return features, labels

def train_incident_model(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = XGBClassifier(random_state=42, eval_metric="logloss")
    model.fit(features_scaled, labels)
    return model, scaler

import base64
import io
import os

def save_incident_model(user_id, model, scaler, save_to_mongo=False, users_collection=None, save_local=True):
    if save_to_mongo and users_collection is not None:
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
                    "ml_incident_model": {
                        "xgboost_model": model_blob,
                        "scaler": scaler_blob,
                        "saved_at": datetime.utcnow()
                    }
                }
            },
            upsert=True
        )
        print(f"[✓] Saved incident model for {user_id} to MongoDB")

    if save_local:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{user_id}_xgboost.pkl")
        joblib.dump(scaler, f"models/{user_id}_xgboost_scaler.pkl")
        print(f"[✓] Saved incident model locally for {user_id}")


def load_incident_model(user_id):
    try:
        model = joblib.load(f"models/{user_id}_xgboost.pkl")
        scaler = joblib.load(f"models/{user_id}_xgboost_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

def predict_incident(model, scaler, location_anomaly, time_anomaly):
    features = np.array([[location_anomaly, time_anomaly]])
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    return probability

def should_retrain_incident(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id, "alert_history.is_incident": {"$exists": True}})
    if last_trained is None or data_count > 100 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False

def load_incident_model_from_db(user_id, users_collection):
    doc = users_collection.find_one({"user_id": user_id})
    if not doc or "ml_incident_model" not in doc:
        print(f"[✗] No incident model found in DB for {user_id}")
        return None, None

    def deserialize(encoded_str):
        buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
        return joblib.load(buffer)

    try:
        model = deserialize(doc["ml_incident_model"]["xgboost_model"])
        scaler = deserialize(doc["ml_incident_model"]["scaler"])
        return model, scaler
    except Exception as e:
        print(f"[✗] Failed to load incident model for {user_id}: {e}")
        return None, None


