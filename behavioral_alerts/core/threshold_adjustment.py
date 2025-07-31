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
    """Adjust threshold based on historical alerts using cross-validation and RandomForest."""
    try:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=35)
        alerts = list(collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": one_month_ago},
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