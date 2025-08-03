from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pickle
import os
from .config import MONGO_URI, MODEL_DIR
from .profiling import detect_user_anomalies #we will call it in the capture file and pass the results as parameters
import io
import base64
import joblib

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

def prepare_incident_data(user_id, collection=locations_collection):
    """Prepare data for incident prediction model with enriched features."""
    try:
        three_month_ago = datetime.now(timezone.utc) - timedelta(days=90)
        alerts = list(collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": three_month_ago},
            "alert.is_incident": {"$exists": True}
        }))
        print(f"[DEBUG] Preprocessed {len(alerts)} alerts for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        if len(alerts) < 20:
            print(f"[DEBUG] Insufficient alerts for user {user_id}: {len(alerts)} records, skipping model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None

        anomaly_features = []
        incident_labels = []
        for alert in alerts:
            location_anomaly = alert.get("alert", {}).get("location_anomaly", 1.0)
            hour_anomaly = alert.get("alert", {}).get("hour_anomaly", 1.0)
            weekday_anomaly = alert.get("alert", {}).get("weekday_anomaly", 1.0)
            month_anomaly = alert.get("alert", {}).get("month_anomaly", 1.0)
            is_incident = alert.get("alert", {}).get("is_incident", False)
            anomaly_features.append([location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly])
            incident_labels.append(1 if is_incident else 0)

        print(f"[DEBUG] Incident labels: {incident_labels}")
        return np.array(anomaly_features), np.array(incident_labels)
    except Exception as e:
        print(f"[✗] Error preparing incident data for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None

def train_incident_model(anomaly_features, incident_labels):
    """Train the incident prediction model with cross-validation and threshold optimization."""
    try:
        if anomaly_features is None or incident_labels is None:
            print(f"[DEBUG] No data to train incident model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return None, None, None

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(anomaly_features)

        #model = RandomForestClassifier(n_estimators=100, random_state=42)
        model = XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=(len(incident_labels) - sum(incident_labels)) / sum(incident_labels) if sum(incident_labels) > 0 else 1)
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_probs = []
        val_labels = []

        for train_idx, val_idx in kf.split(scaled_features):
            X_train, X_val = scaled_features[train_idx], scaled_features[val_idx]
            y_train, y_val = incident_labels[train_idx], incident_labels[val_idx]

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            val_probs.extend(probs)
            val_labels.extend(y_val)

        # Find optimal threshold using F1-score
        thresholds = np.arange(0.1, 1.0, 0.05, 0.5, 0.7)
        f1_scores = []
        for thresh in thresholds:
            preds = [1 if p >= thresh else 0 for p in val_probs]
            f1 = f1_score(val_labels, preds)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"[✓] Optimal threshold: {optimal_threshold:.2f} with F1-score: {f1_scores[optimal_idx]:.2f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Train final model on all data
        model.fit(scaled_features, incident_labels)

        return model, scaler, optimal_threshold
    except Exception as e:
        print(f"[✗] Error training incident model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return None, None, None

def save_incident_model(
    user_id,
    model,
    scaler,
    threshold,
    name="Test User",
    email=None,
    phone="+1234567890",
    emergency_contact_phone="+0987654321",
    collection=users_collection,
    save_to_db=False
):
    """Save the incident model, scaler, and threshold to local storage (mandatory) and optionally to MongoDB."""
    try:
        if model is None or scaler is None or threshold is None:
            print(f"[DEBUG] No incident model to save for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return

        # Mandatory: Save to local storage
        os.makedirs(os.path.join(MODEL_DIR, user_id), exist_ok=True)
        model_path = os.path.join(MODEL_DIR, user_id, f"{user_id}_incident_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, user_id, f"{user_id}_incident_scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"[✓] Saved incident model locally for {user_id} at {model_path} and scaler at {scaler_path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Optional: Save to MongoDB
        if save_to_db:
            if email is None:
                email = f"test_{user_id}@example.com"

            try:
                collection.update_one(
                    {"user_id": user_id},
                    {"$set": {
                        "user_id": user_id,
                        "name": name,
                        "email": email,
                        "phone": phone,
                        "emergency_contact_phone": emergency_contact_phone,
                        "created_at": datetime.now(timezone.utc),
                        "incident_model": pickle.dumps(model),
                        "incident_scaler": pickle.dumps(scaler),
                        "optimal_threshold": threshold
                    }},
                    upsert=True
                )
                print(f"[✓] Saved incident model and threshold for {user_id} to MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            except Exception as e:
                print(f"[✗] Failed to save incident model to MongoDB for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

    except Exception as e:
        print(f"[✗] Error saving incident model locally for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")


def load_incident_model(user_id, path=None, users_collection=users_collection):
    """Load incident model and scaler from MongoDB or disk."""
    try:
        if path is None:
            doc = users_collection.find_one({"user_id": user_id})
            if doc and "ml_incident_model" in doc:
                def deserialize(encoded_str):
                    buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
                    return joblib.load(buffer)
                model = deserialize(doc["ml_incient_model"]["xgboost_model"])
                scaler = deserialize(doc["ml_incident_model"]["scaler"])
                print(f"[✓] Loaded incident model for {user_id} from MongoDB at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                return model, scaler
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

def predict_incident(user_id, location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly):#, collection=locations_collection):
    """Predict if an alert is an incident based on anomaly scores."""
    try:
        model, scaler = load_incident_model(user_id)
        if model is None or scaler is None:
            print(f"[WARNING] No incident model for {user_id}, using default probability at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return (location_anomaly + hour_anomaly + weekday_anomaly + month_anomaly) / 4
        features = pd.DataFrame(
            [[location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly]],
            columns=["location_anomaly", "hour_anomaly", "weekday_anomaly", "month_anomaly"]
        )
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        print(f"[✓] Predicted incident probability for {user_id}: {probability} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return probability
    except Exception as e:
        print(f"[✗] Error predicting incident for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return (location_anomaly + hour_anomaly + weekday_anomaly + month_anomaly) / 4


"""def predict_incident(user_id, latitude, longitude, hour, weekday, month)#, collection=locations_collection):
    #Predict the probability of an incident using enriched features.
    try:
        
        user = users_collection.find_one({"user_id": user_id})
        if user is None or "incident_model" not in user or "incident_scaler" not in user:
            print(f"[DEBUG] No incident model found for {user_id}, returning default probability 1.0 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return 1.0

        model = pickle.loads(user["incident_model"])
        scaler = pickle.loads(user["incident_scaler"])

        # Assuming detect_user_anomalies returns [location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly]
        anomalies = detect_user_anomalies(latitude, longitude, hour, weekday, month, user_id, collection)
        anomaly_features = np.array([anomalies])
        scaled_anomaly_features = scaler.transform(anomaly_features)
        incident_probability = model.predict_proba(scaled_anomaly_features)[0][1]

        print(f"[✓] Predicted incident probability for {user_id}: {incident_probability:.2f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        return incident_probability
    except Exception as e:
        print(f"[✗] Error predicting incident for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return 1.0
"""
