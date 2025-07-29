from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import os
import base64
import io
from .config import MONGO_URI, MODEL_DIR, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def prepare_incident_data(user_id, collection=locations_collection):
    """Prepare data for incident prediction model, using dummy data if insufficient."""
    try:
        data = list(collection.find({"user_id": user_id, "alert": {"$exists": True}}))
        if not data:
            print(f"[DEBUG] No alert data for user {user_id}, using dummy data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            dummy_data = [
                {"alert": {"location_anomaly_score": 0.0, "time_anomaly_score": 0.0, "is_incident": False}},
                {"alert": {"location_anomaly_score": 1.0, "time_anomaly_score": 1.0, "is_incident": True}}
            ]
            df = pd.DataFrame([d["alert"] for d in dummy_data])
        else:
            df = pd.DataFrame([doc["alert"] for doc in data])
        if len(df) < 10:
            print(f"[DEBUG] Insufficient alert data for user {user_id}: {len(df)} records, supplementing with dummy data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            dummy_data = [
                {"alert": {"location_anomaly_score": 0.0, "time_anomaly_score": 0.0, "is_incident": False}},
                {"alert": {"location_anomaly_score": 1.0, "time_anomaly_score": 1.0, "is_incident": True}}
            ]
            dummy_df = pd.DataFrame([d["alert"] for d in dummy_data])
            df = pd.concat([df, dummy_df], ignore_index=True)
        df["is_incident"].fillna(False, inplace=True)
        features = df[["location_anomaly_score", "time_anomaly_score"]]
        labels = df["is_incident"]
        return features, labels
    except Exception as e:
        print(f"[✗] Error preparing incident data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None

def train_incident_model(features, labels):
    """Train incident prediction model and log accuracy."""
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        model = XGBClassifier(random_state=42, eval_metric="logloss")
        model.fit(features_scaled, labels)
        predictions = model.predict(features_scaled)
        accuracy = accuracy_score(labels, predictions)
        print(f"[✓] Incident model accuracy: {accuracy} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return model, scaler
    except Exception as e:
        print(f"[✗] Error training incident model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None

def save_incident_model(user_id, model, scaler, save_to_mongo=True, users_collection=users_collection, save_local=True):
    """Save incident model and scaler to disk and optionally to MongoDB."""
    try:
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
                            "saved_at": datetime.now(timezone.utc)
                        }
                    }
                },
                upsert=True
            )
            print(f"[✓] Saved incident model for {user_id} to MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        if save_local:
            user_dir = os.path.join(MODEL_DIR, user_id)
            os.makedirs(user_dir, exist_ok=True)
            model_path = os.path.join(user_dir, f"{user_id}_xgboost_incident_pred.pkl")
            scaler_path = os.path.join(user_dir, f"{user_id}_xgboost_incident_pred_scaler.pkl")
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"[✓] Saved incident model locally for {user_id} at {os.path.abspath(user_dir)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error saving incident model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

def load_incident_model(user_id, path=None, users_collection=users_collection):
    """Load incident model and scaler from MongoDB or disk."""
    try:
        if path is None:
            # Try MongoDB first
            doc = users_collection.find_one({"user_id": user_id})
            if doc and "ml_incident_model" in doc:
                def deserialize(encoded_str):
                    buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
                    return joblib.load(buffer)
                
                model = deserialize(doc["ml_incident_model"]["xgboost_model"])
                scaler = deserialize(doc["ml_incident_model"]["scaler"])
                print(f"[✓] Loaded incident model for {user_id} from MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return model, scaler
            
            # Fall back to disk
            user_dir = os.path.join(MODEL_DIR, user_id)
            model_path = os.path.join(user_dir, f"{user_id}_xgboost_incident_pred.pkl")
            scaler_path = os.path.join(user_dir, f"{user_id}_xgboost_incident_pred_scaler.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print(f"[✓] Loaded incident model for {user_id} from disk at {os.path.abspath(user_dir)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return model, scaler
            print(f"[✗] Incident model not found for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None
        else:
            model_path = os.path.join(path, f"{user_id}_xgboost_incident_pred.pkl")
            scaler_path = os.path.join(path, f"{user_id}_xgboost_incident_pred_scaler.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print(f"[✓] Loaded incident model for {user_id} from path {path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return model, scaler
            print(f"[✗] Incident model not found for {user_id} at {path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None
    except Exception as e:
        print(f"[✗] Error loading incident model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None

def predict_incident(user_id, location_anomaly, time_anomaly, collection=locations_collection):
    """Predict if an alert is an incident based on anomaly scores."""
    try:
        model, scaler = load_incident_model(user_id)
        if model is None or scaler is None:
            print(f"[WARNING] No incident model for {user_id}, using default probability at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return (location_anomaly + time_anomaly) / 2
        features = pd.DataFrame([[location_anomaly, time_anomaly]], columns=["location_anomaly_score", "time_anomaly_score"])
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        print(f"[✓] Predicted incident probability for {user_id}: {probability} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return probability
    except Exception as e:
        print(f"[✗] Error predicting incident for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return (location_anomaly + time_anomaly) / 2  # Fallback to average anomaly scores

def should_retrain_incident(user_id, last_trained, collection=locations_collection):
    """Determine if the incident model should be retrained."""
    try:
        data_count = collection.count_documents({"user_id": user_id, "alert": {"$exists": True}})
        recent_data = collection.count_documents({
            "user_id": user_id,
            "alert": {"$exists": True},
            "timestamp": {"$gt": datetime.now(timezone.utc) - timedelta(days=7)}
        })
        if (last_trained is None or
                data_count > 100 or
                datetime.now(timezone.utc) - last_trained > timedelta(days=30) or
                (data_count > 0 and recent_data > 0.2 * data_count)):
            print(f"[DEBUG] Retraining required for {user_id}: last_trained={last_trained}, data_count={data_count}, recent_data={recent_data} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return True
        print(f"[DEBUG] No retraining needed for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return False
    except Exception as e:
        print(f"[✗] Error checking retrain condition for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return True

def load_incident_model_from_db(user_id, users_collection=users_collection):
    """Load incident model and scaler from MongoDB."""
    try:
        doc = users_collection.find_one({"user_id": user_id})
        if not doc or "ml_incident_model" not in doc:
            print(f"[✗] No incident model found in DB for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None
        
        def deserialize(encoded_str):
            buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
            return joblib.load(buffer)
        
        model = deserialize(doc["ml_incident_model"]["xgboost_model"])
        scaler = deserialize(doc["ml_incident_model"]["scaler"])
        print(f"[✓] Loaded incident model for {user_id} from MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return model, scaler
    except Exception as e:
        print(f"[✗] Failed to load incident model for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None
