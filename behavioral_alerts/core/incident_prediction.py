from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from .utils import serialize_model, deserialize_model
from pymongo.collection import Collection
from .config import MODEL_DIR, DEFAULT_PROB_THRESHOLD

def prepare_incident_data(collection, user_id):
    try:
        data = list(collection.find({"user_id": user_id, "alert": {"$exists": True}}))
        if not data:
            raise ValueError(f"No alert data for user {user_id}")
        df = pd.DataFrame([d["alert"] for d in data])
        if df.empty or len(df) < 10:
            raise ValueError(f"Insufficient alert data for user {user_id}")
        features = df[["location_anomaly_score", "time_anomaly_score"]]
        labels = df["is_incident"]
        return features, labels
    except Exception as e:
        print(f"Error preparing incident data for {user_id}: {e}")
        return None, None

def train_incident_model(features, labels):
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        model = XGBClassifier(random_state=42, eval_metric="logloss")
        model.fit(features_scaled, labels)
        from sklearn.metrics import accuracy_score
        predictions = model.predict(features_scaled)
        accuracy = accuracy_score(labels, predictions)
        print(f"Incident model accuracy: {accuracy}")
        return model, scaler
    except Exception as e:
        print(f"Error training incident model: {e}")
        return None, None

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
        #user_dir = os.path.join("behavioral_alerts", "models", user_id)
        #user_dir = os.path.join("..","SOLUTION_SECURITE_PERSO" ,"models", user_id)
        user_dir = os.path.join(MODEL_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        joblib.dump(model, os.path.join(user_dir, f"{user_id}_xgboost_incident_pred.pkl"))
        joblib.dump(scaler, os.path.join(user_dir, f"{user_id}_xgboost_incident_pred_scaler.pkl"))
        print(f"[✓] Saved incident model locally for {user_id} at {user_dir}" )
        print("Saving incident model to:", os.path.abspath(user_dir))


import os
import joblib

def load_incident_model(user_id, path=None):
    if path is None:
        try:
            #return joblib.load(f"models/{user_id}_xgboost_incident_pred.pkl") , joblib.load(f"models/{user_id}_xgboost_incident_pred_scaler.pkl")
            user_dir = os.path.join(MODEL_DIR, user_id)
            model = joblib.load(os.path.join(user_dir, f"{user_id}_xgboost_incident_pred.pkl"))
            scaler = joblib.load(os.path.join(user_dir, f"{user_id}_xgboost_incident_pred_scaler.pkl"))
            return model, scaler
        except FileNotFoundError:
            print(f"[✗] Incident model not found for {user_id}")
            return None
    else:
        try:
            model = joblib.load(os.path.join(path, f"{user_id}_xgboost_incident_pred.pkl"))
            scaler = joblib.load(os.path.join(path, f"{user_id}_xgboost_incident_pred_scaler.pkl"))
            return model, scaler
        except FileNotFoundError:
            print(f"[✗] Incident model not found for {user_id}")
            return None, None


def predict_incident(model, scaler, location_anomaly, time_anomaly):
    try:
        features = pd.DataFrame([[location_anomaly, time_anomaly]], columns=["location_anomaly_score", "time_anomaly_score"])
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        return probability
    except Exception as e:
        print(f"Error predicting incident: {e}")
        return 0.0

"""def should_retrain_incident(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id, "alert_history.is_incident": {"$exists": True}})
    if last_trained is None or data_count > 100 or (datetime.now() - last_trained > timedelta(days=30)):
        return True
    return False"""
def should_retrain_incident(collection, user_id, last_trained):
    data_count = collection.count_documents({"user_id": user_id, "alert": {"$exists": True}})
    recent_data = collection.count_documents({"user_id": user_id, "alert": {"$exists": True}, "timestamp": {"$gt": datetime.now() - timedelta(days=7)}})
    return (last_trained is None or data_count > 100 or
            datetime.now() - last_trained > timedelta(days=30) or
            recent_data > 0.2 * data_count)

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

