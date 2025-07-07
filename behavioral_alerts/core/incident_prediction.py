from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

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

def save_incident_model(user_id, model, scaler):
    joblib.dump(model, f"models/{user_id}_xgboost.pkl")
    joblib.dump(scaler, f"models/{user_id}_xgboost_scaler.pkl")

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