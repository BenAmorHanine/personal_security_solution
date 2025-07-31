from pymongo import MongoClient
from .profiling import detect_user_anomalies
from .threshold_adjustment import adjust_threshold
from .incident_prediction import prepare_incident_data, train_incident_model, save_incident_model, predict_incident
from .db_functions import create_user, register_device, update_location, log_alert
from .config import MONGO_URI
import uuid
import random
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

# Function to generate synthetic data for a user
def generate_synthetic_data(user_id, num_locations=30, num_alerts=20):
    # Generate synthetic locations with varying times and locations
    for _ in range(num_locations):
        latitude = random.uniform(48.7, 49.0)
        longitude = random.uniform(2.2, 2.5)
        timestamp = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        update_location(user_id, device_id, latitude, longitude, timestamp)

    # Generate synthetic alerts with varying anomaly scores and incident labels
    for _ in range(num_alerts):
        latitude = random.uniform(48.7, 49.0)
        longitude = random.uniform(2.2, 2.5)
        hour = random.randint(0, 23)
        weekday = random.randint(0, 6)
        month = random.randint(1, 12)
        location_anomaly, time_anomaly = detect_user_anomalies(latitude, longitude, hour, weekday, month, user_id, locations_collection)
        ai_score = predict_incident(user_id, latitude, longitude, hour, weekday, month, locations_collection)
        is_incident = random.choice([True, False])  # Randomly assign incident label
        log_alert(user_id, device_id, latitude, longitude, None, ai_score, is_incident)



# Function to evaluate model performance
def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)
    return np.mean(f1_scores)

# Function to optimize threshold
def optimize_threshold(model, X, y):
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_threshold = 0.5
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
        f1 = f1_score(y, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

# Simulated users and devices
user_ids = [str(uuid.uuid4()) for _ in range(3)]  # Create 3 users
device_ids = [str(uuid.uuid4()) for _ in range(3)]  # Create 3 devices

try:
    for user_id, device_id in zip(user_ids, device_ids):
        # Create user with unique email
        unique_email = f"test_{user_id}@eexample.com"
        existing_user = users_collection.find_one({"user_id": user_id})
        if not existing_user:
            create_user("Test User", unique_email, "+1234567890", "+0987654321")
        else:
            users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "name": "Test User",
                        "email": unique_email,
                        "phone": "+1234567890",
                        "emergency_contact_phone": "+0987654321",
                        "created_at": datetime.now(timezone.utc)
                    }
                }
            )
            print(f"[✓] Updated user {user_id} with email {unique_email} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Register device
        existing_device = users_collection.find_one({"user_id": user_id, "devices.device_id": device_id})
        if not existing_user:
            register_device(user_id, "smartphone", "SIM123456", 100)
        else:
            users_collection.update_one(
                {"user_id": user_id, "devices.device_id": device_id},
                {
                    "$set": {
                        "devices.$.device_type": "smartphone",
                        "devices.$.sim_id": "SIM123456",
                        "devices.$.battery_level": 100,
                        "devices.$.registered_at": datetime.now(timezone.utc)
                    }
                }
            )
            print(f"[✓] Updated device {device_id} for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

        # Generate synthetic data for the user
        generate_synthetic_data(user_id)

        # Train incident model
        anomaly_features, incident_labels = prepare_incident_data(user_id, locations_collection)
        if anomaly_features is not None and incident_labels is not None:
            model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
            if model is not None and scaler is not None:
                # Evaluate model performance
                f1_score_val = evaluate_model(model, anomaly_features, incident_labels)
                print(f"[✓] Model F1-score for user {user_id}: {f1_score_val:.2f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

                # Use the optimal_threshold from train_incident_model
                print(f"[✓] Optimal threshold for user {user_id}: {optimal_threshold:.2f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

                # Save model and threshold (local mandatory, MongoDB optional)
                save_incident_model(
                    user_id,
                    model,
                    scaler,
                    optimal_threshold,
                    name="Test User",
                    email=unique_email,
                    phone="+1234567890",
                    emergency_contact_phone="+0987654321",
                    collection=users_collection,
                    save_to_db=True  # Set to True to enable MongoDB saving
                )

        # Simulate SOS alerts
        for _ in range(5):
            ai_score = random.uniform(0.0, 1.0)
            is_incident = random.choice([True, False])
            log_alert(user_id, device_id, float(0), float(0), None, ai_score, is_incident)
            print(f"[✓] Logged SOS alert for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

except Exception as e:
    print(f"[✗] Error in test pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")