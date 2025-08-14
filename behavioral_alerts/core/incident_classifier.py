from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import numpy as np
import os, joblib, base64, io, pickle
import joblib
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from .config import MONGO_URI, MODEL_DIR, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]


# --- 1. Data Preparation ---
def prepare_incident_classif_data(user_id, collection=locations_collection):
    three_month_ago = datetime.now(timezone.utc) - timedelta(days=90)
    alerts = list(collection.find({
        "user_id": user_id,
        "timestamp": {"$gte": three_month_ago},
        "alert.is_incident": {"$exists": True}
    }))

    if len(alerts) < 20:
        print(f"[DEBUG] Not enough data to train threshold model for {user_id}")
        return None, None

    features, labels = [], []
    for alert in alerts:
        loc_anomaly = alert.get("alert", {}).get("location_anomaly", 1.0)
        hour = alert["timestamp"].hour

        hour_anomaly = alert.get("alert", {}).get("hour_anomaly", 1.0)
        weekday_anomaly = alert.get("alert", {}).get("weekday_anomaly", 1.0)
        month_anomaly = alert.get("alert", {}).get("month_anomaly", 1.0)
        is_incident = alert.get("alert", {}).get("is_incident", False)

        features.append([loc_anomaly, hour_anomaly, weekday_anomaly, month_anomaly])
        labels.append(1 if is_incident else 0)

    return np.array(features), np.array(labels)

# --- 2. Model Training ---
def train_incident_classifier_before_flip_score(features, labels):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaled, labels)
    return model, scaler

def train_incident_classifier(features, labels):
    """Train a RandomForestClassifier on anomaly scores."""
    try:
        features = 1.0 - np.array(features)  # Flip scores: 1.0 = anomalous
        model = RandomForestClassifier(random_state=42)
        model.fit(features, labels)
        scaler = StandardScaler()
        scaler.fit(features)
        logger.info("Trained incident classifier")
        return model, scaler
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        return None, None

# --- 3. F1-based Threshold Optimization ---
def optimize_incident_threshold(user_id):
    features, labels = prepare_incident_classif_data(user_id)
    if features is None:
            logger.warning(f"Using default threshold {DEFAULT_PROB_THRESHOLD} for {user_id}")
            return DEFAULT_PROB_THRESHOLD

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    features = 1.0 - np.array(features)  # Flip scores

    val_probs, val_labels = [], []
    for train_idx, val_idx in kf.split(scaled):
        model.fit(scaled[train_idx], labels[train_idx])
        probs = model.predict_proba(scaled[val_idx])[:, 1]
        val_probs.extend(probs)
        val_labels.extend(labels[val_idx])

    thresholds = np.arange(0.1, 1.0, 0.05)
    f1_scores = [f1_score(val_labels, [1 if p >= t else 0 for p in val_probs]) for t in thresholds]
    #best_threshold = thresholds[np.argmax(f1_scores)]
    best_threshold = max(thresholds[np.argmax(f1_scores)], 0.2)  # Enforce minimum 0.2
    print(f"[✓] Best threshold for user {user_id}: {best_threshold:.2f}")
    return best_threshold

# --- 4. Save Model Locally + Optionally in MongoDB ---
def save_incident_classifier(user_id, model, scaler, threshold, save_to_db=False):
    os.makedirs(os.path.join(MODEL_DIR, user_id), exist_ok=True)
    model_path = os.path.join(MODEL_DIR, user_id, "threshold_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, user_id, "threshold_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[✓] Saved threshold model locally for user {user_id} at {os.path.abspath(model_path)} and scaler at {os.path.abspath(scaler_path)}")

    if save_to_db:
        try:
            
            #model_blob = serialize(model)
            #scaler_blob = serialize(scaler)
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "threshold_model": pickle.dumps(model),
                    "threshold_scaler": pickle.dumps(scaler),
                    "threshold_value": threshold,
                    "threshold_updated_at": datetime.now(timezone.utc)
                }},
                upsert=True
            )
            print(f"[✓] Also saved threshold model to MongoDB for user {user_id}")
        except Exception as e:
            print(f"[✗] Failed to save threshold model to DB for {user_id}: {e}")

# --- 5. Load Model (DB → fallback to local → fallback to train) ---

def load_incident_classifier(user_id, fallback_to_train=True):
    model_path = os.path.join(MODEL_DIR, user_id, "threshold_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, user_id, "threshold_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print(f"[✓] Loaded threshold model from local for {user_id} at {os.path.abspath(model_path)}")
            return model, scaler
        except Exception as e:
            print(f"[✗] Failed to load local threshold model , trying from DB, for {user_id}: {e}")
    user = users_collection.find_one({"user_id": user_id})
    if user and "threshold_model" in user and "threshold_scaler" in user:
        try:
            model = pickle.loads(user["threshold_model"])
            scaler = pickle.loads(user["threshold_scaler"])
            print(f"[✓] Loaded threshold model from MongoDB for {user_id}")
            return model, scaler
        except Exception as e:
            print(f"[✗] Failed to load threshold model from DB for {user_id}: {e}")
    if fallback_to_train:
        print(f"[INFO] No threshold model found for {user_id}. Training new one...")
        features, labels = prepare_incident_classif_data(user_id)
        if features is None:
            return None, None
        model, scaler = train_incident_classifier(features, labels)
        threshold = optimize_incident_threshold(user_id)
        save_incident_classifier(user_id, model, scaler, threshold, save_to_db=True)
        print(f"[info] Retrained model and saved for user {user_id}")
        return model, scaler
    print(f"[✗] Failed to load and retrain threshold model for user {user_id}")
    return None, None


def predict_threshold(model, scaler, features):
    features = np.array(features).reshape(1, -1)  # Ensure 2D: (1, 4)
    features_scaled = scaler.transform(features)
    return model.predict_proba(features_scaled)[0][1]

# --- 6. Predict Using Threshold Model ---
def predict_incident_probability(model, scaler, features):
    
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    

    #features_scaled = scaler.transform([features])
    return model.predict_proba(features_scaled)[0][1]

def serialize(obj):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

