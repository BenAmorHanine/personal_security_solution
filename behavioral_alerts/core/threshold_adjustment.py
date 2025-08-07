from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import numpy as np
import os, joblib, base64, io, pickle
from sklearn.externals import joblib

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
def prepare_threshold_data(user_id, collection=locations_collection):
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
def train_threshold_model(features, labels):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaled, labels)
    return model, scaler

# --- 3. F1-based Threshold Optimization ---
def adjust_threshold(user_id):
    features, labels = prepare_threshold_data(user_id)
    if features is None:
        return 0.5

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    val_probs, val_labels = [], []
    for train_idx, val_idx in kf.split(scaled):
        model.fit(scaled[train_idx], labels[train_idx])
        probs = model.predict_proba(scaled[val_idx])[:, 1]
        val_probs.extend(probs)
        val_labels.extend(labels[val_idx])

    thresholds = np.arange(0.1, 1.0, 0.05)
    f1_scores = [f1_score(val_labels, [1 if p >= t else 0 for p in val_probs]) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"[✓] Best threshold for user {user_id}: {best_threshold:.2f}")
    return best_threshold

# --- 4. Save Model Locally + Optionally in MongoDB ---
def save_threshold_model(user_id, model, scaler, threshold, save_to_db=False):
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

def load_threshold_model(user_id, fallback_to_train=True):
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
        features, labels = prepare_threshold_data(user_id)
        if features is None:
            return None, None
        model, scaler = train_threshold_model(features, labels)
        threshold = adjust_threshold(user_id)
        save_threshold_model(user_id, model, scaler, threshold, save_to_db=True)
        print(f"[info] Retrained model and saved for user {user_id}")
        return model, scaler
    print(f"[✗] Failed to load and retrain threshold model for user {user_id}")
    return None, None




# --- 6. Predict Using Threshold Model ---
def predict_threshold(model, scaler, features):
    """
    features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
    """
    features_scaled = scaler.transform([features])
    return model.predict_proba(features_scaled)[0][1]

def serialize(obj):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

"""
PREVIOUS VERSION:
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from .config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]

def adjust_threshold(user_id, collection=locations_collection):
    #Adjust threshold based on historical alerts using cross-validation and RandomForest.
    try:
        three_month_ago = datetime.now(timezone.utc) - timedelta(days=90)
        alerts = list(collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": three_month_ago},
            "alert.is_incident": {"$exists": True}
        }))
        print(f"[DEBUG] Preprocessed {len(alerts)} alerts for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        if len(alerts) < 20:
            print(f"[DEBUG] Insufficient alerts for user {user_id}: {len(alerts)} records, using default threshold 0.5 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return 0.5

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

        anomaly_features = np.array(anomaly_features)
        incident_labels = np.array(incident_labels)

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_probs = []
        val_labels = []

        for train_idx, val_idx in kf.split(anomaly_features):
            X_train, X_val = anomaly_features[train_idx], anomaly_features[val_idx]
            y_train, y_val = incident_labels[train_idx], incident_labels[val_idx]

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            val_probs.extend(probs)
            val_labels.extend(y_val)

        # Find optimal threshold using F1-score
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        for thresh in thresholds:
            preds = [1 if p >= thresh else 0 for p in val_probs]
            f1 = f1_score(val_labels, preds)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"[✓] Adjusted threshold: {optimal_threshold:.2f} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        return optimal_threshold
    except Exception as e:
        print(f"[✗] Error adjusting threshold for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")
        return 0.5
        
        
def load_threshold_model(user_id, fallback_to_train=True):
    # 1. Try MongoDB
    user = users_collection.find_one({"user_id": user_id})
    if user and "threshold_model" in user and "threshold_scaler" in user:
        try:
            def deserialize(encoded_str):
                buffer = io.BytesIO(base64.b64decode(encoded_str.encode('utf-8')))
                return joblib.load(buffer)
            model = deserialize(user["threshold_model"])
            scaler = deserialize(user["threshold_scaler"])
            print(f"[✓] Loaded threshold model from MongoDB for {user_id}")
            return model, scaler
        except Exception as e:
            print(f"[✗] Failed to load threshold model from DB for {user_id}: {e}")

    # 2. Try local files
    model_path = os.path.join(MODEL_DIR, user_id, "threshold_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, user_id, "threshold_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print(f"[✓] Loaded threshold model from local for {user_id} at {os.path.abspath(model_path)}")
            return model, scaler
        except Exception as e:
            print(f"[✗] Failed to load local threshold model for {user_id}: {e}")

    # 3. Optional: Train from scratch
    if fallback_to_train:
        print(f"[INFO] No threshold model found for {user_id}. Training new one...")
        features, labels = prepare_threshold_data(user_id)
        if features is None:
            return None, None
        model, scaler = train_threshold_model(features, labels)
        threshold = adjust_threshold(user_id)
        save_threshold_model(user_id, model, scaler, threshold, save_to_db=True)
        return model, scaler

    return None, None
        
        
        """

