from pymongo import MongoClient
from .profiling import detect_user_anomalies, build_user_profile
from .threshold_adjustment import adjust_threshold
from .incident_prediction import prepare_incident_data, train_incident_model, save_incident_model, predict_incident
from .db_functions import create_user, register_device, update_location, log_alert
from .config import MONGO_URI
import uuid
import random
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
"""
Use cases tested:
User with Zero Data: Tests build_user_profile and pipeline with no location or alert data, ensuring graceful handling (skips profiling and model training).
User with Zero Alert History: Tests profiling with location data (30 points) but no alerts, verifying clustering works and model training is skipped.
User with Sufficient Data (Real-Time): Tests full pipeline (profiling, model training, real-time predictions) with 100 locations and 50 alerts, simulating production-like conditions.
User with Invalid Data: Tests update_location and build_user_profile with invalid inputs (e.g., latitude=999, invalid timestamp), ensuring validation prevents crashes.
Normal User (Baseline): Tests standard pipeline with 30 locations and 20 alerts, validating typical user behavior and model performance.
"""
client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]
users_collection = db["users"]

# Function to generate synthetic data for a user
def generate_synthetic_data(user_id, device_id, num_locations=30, num_alerts=20):
    for _ in range(num_locations):
        latitude = random.uniform(48.7, 49.0)
        longitude = random.uniform(2.2, 2.5)
        timestamp = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        update_location(user_id, device_id, latitude, longitude, timestamp)

    for _ in range(num_alerts):
        latitude = random.uniform(48.7, 49.0)
        longitude = random.uniform(2.2, 2.5)
        hour = random.randint(0, 23)
        weekday = random.randint(0, 6)
        month = random.randint(1, 12)
        location_anomaly, time_anomaly = detect_user_anomalies(latitude, longitude, hour, weekday, month, user_id, locations_collection)
        ai_score = predict_incident(user_id, latitude, longitude, hour, weekday, month, locations_collection)
        is_incident = location_anomaly > 0.7 or time_anomaly > 0.7
        log_alert(user_id, device_id, latitude, longitude, None, ai_score, is_incident)

# Function to evaluate model performance
def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores, precisions, recalls, aucs, cms = [], [], [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        cms.append(confusion_matrix(y_val, y_pred))
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, y_pred_proba))
    # Average confusion matrix
    cm_avg = np.mean(cms, axis=0).astype(int)
    metrics = {
        "f1_score": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "roc_auc": np.mean(aucs) if aucs else None,
        "confusion_matrix": cm_avg.tolist()
    }
    return metrics

# Simulated users and devices
user_ids = [str(uuid.uuid4()) for _ in range(5)]  # 5 users for different test cases
device_ids = [str(uuid.uuid4()) for _ in range(5)]

try:
    # Initialize collections if they don't exist
    if "users" not in db.list_collection_names():
        db.create_collection("users", validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["user_id", "name", "email", "phone", "emergency_contact_phone", "created_at"],
                "properties": {
                    "user_id": {"bsonType": "string"},
                    "name": {"bsonType": "string"},
                    "email": {"bsonType": "string"},
                    "phone": {"bsonType": "string"},
                    "emergency_contact_phone": {"bsonType": "string"},
                    "created_at": {"bsonType": "date"}
                }
            }
        })
        users_collection.create_index("email", unique=True)
        print(f"[✓] Created users collection with schema and email index at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

    if "locations" not in db.list_collection_names():
        db.create_collection("locations")
        print(f"[✓] Created locations collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

    # Test Case 1: User with zero data
    print(f"[TEST] Testing user with zero data: {user_ids[0]}")
    unique_email = f"test_{user_ids[0]}@example.com"
    try:
        create_user("Zero Data User", unique_email, "+1234567890", "+0987654321")
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_ids[0], locations_collection)
        if centroids is None:
            print(f"[DEBUG] No data for user {user_ids[0]}, skipping model training")
        else:
            anomaly_features, incident_labels = prepare_incident_data(user_ids[0], locations_collection)
            if anomaly_features is not None and incident_labels is not None:
                model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
                if model is not None and scaler is not None:
                    metrics = evaluate_model(model, anomaly_features, incident_labels)
                    print(f"[✓] Model metrics for user {user_ids[0]}: F1={metrics['f1_score']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, ROC-AUC={metrics['roc_auc']:.2f}, CM={metrics['confusion_matrix']}")
                    save_incident_model(user_ids[0], model, scaler, optimal_threshold, name="Zero Data User", email=unique_email, save_to_db=True)
    except Exception as e:
        print(f"[✗] Error in zero data test for user {user_ids[0]}: {e}")

    # Test Case 2: User with zero alert history
    print(f"[TEST] Testing user with zero alert history: {user_ids[1]}")
    unique_email = f"test_{user_ids[1]}@example.com"
    try:
        create_user("No Alert User", unique_email, "+1234567890", "+0987654321")
        generate_synthetic_data(user_ids[1], device_ids[1], num_locations=30, num_alerts=0)
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_ids[1], locations_collection)
        if centroids is None:
            print(f"[DEBUG] No alert data for user {user_ids[1]}, skipping model training")
        else:
            anomaly_features, incident_labels = prepare_incident_data(user_ids[1], locations_collection)
            if anomaly_features is not None and incident_labels is not None:
                model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
                if model is not None and scaler is not None:
                    metrics = evaluate_model(model, anomaly_features, incident_labels)
                    print(f"[✓] Model metrics for user {user_ids[1]}: F1={metrics['f1_score']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, ROC-AUC={metrics['roc_auc']:.2f}, CM={metrics['confusion_matrix']}")
                    save_incident_model(user_ids[1], model, scaler, optimal_threshold, name="No Alert User", email=unique_email, save_to_db=True)
            else:
                print(f"[DEBUG] No alert data for user {user_ids[1]}, skipping model training")
    except Exception as e:
        print(f"[✗] Error in zero alert test for user {user_ids[1]}: {e}")

    # Test Case 3: User with sufficient data for real-time scenario
    print(f"[TEST] Testing user with sufficient data: {user_ids[2]}")
    unique_email = f"test_{user_ids[2]}@example.com"
    try:
        create_user("Real-Time User", unique_email, "+1234567890", "+0987654321")
        generate_synthetic_data(user_ids[2], device_ids[2], num_locations=100, num_alerts=50)
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_ids[2], locations_collection)
        if centroids is not None:
            anomaly_features, incident_labels = prepare_incident_data(user_ids[2], locations_collection)
            if anomaly_features is not None and incident_labels is not None:
                model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
                if model is not None and scaler is not None:
                    metrics = evaluate_model(model, anomaly_features, incident_labels)
                    print(f"[✓] Model metrics for user {user_ids[2]}: F1={metrics['f1_score']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, ROC-AUC={metrics['roc_auc']:.2f}, CM={metrics['confusion_matrix']}")
                    save_incident_model(user_ids[2], model, scaler, optimal_threshold, name="Real-Time User", email=unique_email, save_to_db=True)
                    # Simulate real-time predictions
                    for _ in range(5):
                        latitude = random.uniform(48.7, 49.0)
                        longitude = random.uniform(2.2, 2.5)
                        hour = random.randint(0, 23)
                        weekday = random.randint(0, 6)
                        month = random.randint(1, 12)
                        ai_score = predict_incident(user_ids[2], latitude, longitude, hour, weekday, month, locations_collection)
                        print(f"[✓] Real-time prediction for user {user_ids[2]}: AI Score={ai_score:.2f}")
        else:
            print(f"[DEBUG] No profile built for user {user_ids[2]}, skipping model training")
    except Exception as e:
        print(f"[✗] Error in real-time test for user {user_ids[2]}: {e}")

    # Test Case 4: User with invalid data
    print(f"[TEST] Testing user with invalid data: {user_ids[3]}")
    unique_email = f"test_{user_ids[3]}@example.com"
    try:
        create_user("Invalid Data User", unique_email, "+1234567890", "+0987654321")
        register_device(user_ids[3], "smartphone", "SIM123456", 100)
        # Insert invalid location data
        update_location(user_ids[3], device_ids[3], latitude=999, longitude=999, timestamp=datetime.now(timezone.utc))
        update_location(user_ids[3], device_ids[3], latitude=-90, longitude=-180, timestamp=datetime.now(timezone.utc))
        # Insert invalid time data
        update_location(user_ids[3], device_ids[3], latitude=48.8, longitude=2.3, timestamp=datetime.now(timezone.utc) - timedelta(hours=25))
        # Insert valid data
        update_location(user_ids[3], device_ids[3], latitude=48.8, longitude=2.3, timestamp=datetime.now(timezone.utc))
        centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_ids[3], locations_collection)
        if centroids is None:
            print(f"[DEBUG] Invalid data handled correctly for user {user_ids[3]} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            print(f"[WARNING] Invalid data not handled correctly for user {user_ids[3]}: {len(centroids)} clusters found")
        # Try model training
        anomaly_features, incident_labels = prepare_incident_data(user_ids[3], locations_collection)
        if anomaly_features is not None and incident_labels is not None:
            model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
            if model is not None and scaler is not None:
                metrics = evaluate_model(model, anomaly_features, incident_labels)
                print(f"[✓] Model metrics for user {user_ids[3]}: F1={metrics['f1_score']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, ROC-AUC={metrics['roc_auc']:.2f}, CM={metrics['confusion_matrix']}")
    except Exception as e:
        print(f"[✗] Error in invalid data test for user {user_ids[3]} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

    # Test Case 5: Normal user
    print(f"[TEST] Testing normal user: {user_ids[4]}")
    unique_email = f"test_{user_ids[4]}@example.com"
    try:
        if users_collection.find_one({"user_id": user_ids[4]}):
            print(f"[DEBUG] User {user_ids[4]} already exists, skipping at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        else:
            create_user("Normal User", unique_email, "+1234567890", "+0987654321")
            register_device(user_ids[4], "smartphone", "SIM123456", 100)
            generate_synthetic_data(user_ids[4], device_ids[4])
            centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_ids[4], locations_collection)
            if centroids is not None:
                anomaly_features, incident_labels = prepare_incident_data(user_ids[4], locations_collection)
                if anomaly_features is not None and incident_labels is not None:
                    model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
                    if model is not None and scaler is not None:
                        metrics = evaluate_model(model, anomaly_features, incident_labels)
                        print(f"[✓] Model metrics for user {user_ids[4]}: F1={metrics['f1_score']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, ROC-AUC={metrics['roc_auc']:.2f}, CM={metrics['confusion_matrix']}")
                        save_incident_model(user_ids[4], model, scaler, optimal_threshold, name="Normal User", email=unique_email, save_to_db=True)
                for _ in range(5):
                    latitude = random.uniform(48.7, 49.0)
                    longitude = random.uniform(2.2, 2.5)
                    hour = random.randint(0, 23)
                    weekday = random.randint(0, 6)
                    month = random.randint(1, 12)
                    ai_score = predict_incident(user_ids[4], latitude, longitude, hour, weekday, month, locations_collection)
                    is_incident = random.choice([True, False])
                    log_alert(user_ids[4], device_ids[4], latitude, longitude, None, ai_score, is_incident)
                    print(f"[✓] Logged SOS alert for user {user_ids[4]} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            else:
                print(f"[DEBUG] No profile built for user {user_ids[4]}, skipping model training")
    except Exception as e:
        print(f"[✗] Error in normal user test for user {user_ids[4]}: {e}")

except Exception as e:
    print(f"[✗] Error in test pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")