from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
from .config import MONGO_URI, MODEL_DIR, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def prepare_incident_data(user_id, collection=locations_collection):
    """Prepare data for incident prediction model."""
    try:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        alerts = list(collection.find({
            "user_id": user_id,
            "alert": {"$exists": True},
            "timestamp": {"$gte": one_month_ago}
        }))
        print(f"[DEBUG] Preprocessed {len(alerts)} alerts for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        if len(alerts) < 2:
            print(f"[DEBUG] Insufficient alert data for user {user_id}: {len(alerts)} records, supplementing with dummy data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            features = np.array([[0.0, 0.0], [1.0, 1.0]])
            labels = np.array([0, 1])
            return features, labels
        
        features = []
        labels = []
        for alert in alerts:
            features.append([
                alert["alert"]["location_anomaly_score"],
                alert["alert"]["time_anomaly_score"]
            ])
            labels.append(int(alert["alert"]["is_incident"]))
        print(f"[DEBUG] Label values: {labels}")
        
        features = np.array(features)
        labels = np.array(labels)
        print(f"[DEBUG] Prepared {len(features)} records for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return features, labels
    except Exception as e:
        print(f"[✗] Error preparing incident data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1])

def train_incident_model(features, labels):
    """Train logistic regression model for incident prediction."""
    try:
        if len(features) < 2 or len(np.unique(labels)) < 2:
            print(f"[✗] Error training incident model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: Insufficient data or single class")
            return None, None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, labels)
        
        predictions = model.predict(X_scaled)
        accuracy = accuracy_score(labels, predictions)
        print(f"[✓] Incident model accuracy: {accuracy} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return model, scaler
    except Exception as e:
        print(f"[✗] Error training incident model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None

def save_incident_model(user_id, model, scaler, users_collection=users_collection):
    """Save incident model and scaler to disk and MongoDB."""
    try:
        if model is None or scaler is None:
            print(f"[✗] Error saving incident model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: Model or scaler is None")
            return
        
        # Verify types
        if not isinstance(model, LogisticRegression):
            print(f"[✗] Error saving incident model for {user_id}: Expected LogisticRegression, got {type(model)}")
            return
        if not isinstance(scaler, StandardScaler):
            print(f"[✗] Error saving incident scaler for {user_id}: Expected StandardScaler, got {type(scaler)}")
            return
        
        # Save to disk
        user_dir = os.path.join(MODEL_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        model_path = os.path.join(user_dir, f"{user_id}_incident_model.pkl")
        scaler_path = os.path.join(user_dir, f"{user_id}_incident_scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"[✓] Saved incident model locally for {user_id} at {model_path} and scaler at {scaler_path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Save to MongoDB
        model_bytes = joblib.dump(model, "memory")[0]
        scaler_bytes = joblib.dump(scaler, "memory")[0]
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "ml_incident_model": model_bytes,
                "ml_incident_scaler": scaler_bytes
            }},
            upsert=True
        )
        print(f"[✓] Saved incident model for {user_id} to MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error saving incident model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

def predict_incident(user_id, loc_anomaly, time_anomaly):
    """Predict incident probability using trained model from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, user_id, f"{user_id}_incident_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, user_id, f"{user_id}_incident_scaler.pkl")
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"[DEBUG] No incident model or scaler found for user {user_id} at {model_path}, using default score {DEFAULT_PROB_THRESHOLD} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return DEFAULT_PROB_THRESHOLD
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Verify types
        if not isinstance(model, LogisticRegression):
            print(f"[✗] Error loading incident model for {user_id}: Expected LogisticRegression, got {type(model)}")
            return DEFAULT_PROB_THRESHOLD
        if not isinstance(scaler, StandardScaler):
            print(f"[✗] Error loading incident scaler for {user_id}: Expected StandardScaler, got {type(scaler)}")
            return DEFAULT_PROB_THRESHOLD
        
        features = np.array([[loc_anomaly, time_anomaly]])
        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        print(f"[✓] Predicted incident probability for {user_id}: {prob} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return prob
    except Exception as e:
        print(f"[✗] Error predicting incident for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return DEFAULT_PROB_THRESHOLD