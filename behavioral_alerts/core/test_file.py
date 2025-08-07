# test_pipeline.py
import pytest
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
import os
from .db_functions import create_user, register_device, update_location, log_alert
from .capture import process_capture
from .incident_prediction import prepare_incident_data, train_incident_model, save_incident_model, load_incident_model, predict_incident
from .threshold_adjustment import prepare_threshold_data, train_threshold_model, save_threshold_model, load_threshold_model, predict_threshold
from .profiling import preprocess_data, build_user_profile, detect_user_anomalies
from .config import MONGO_URI, MODEL_DIR, DEFAULT_PROB_THRESHOLD

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

# Test fixtures
@pytest.fixture(scope="module")
def setup_database():
    """Set up test database with a user and location data."""
    # Clean database
    users_collection.drop()
    locations_collection.drop()
    
    # Create test user
    user_id = create_user(
        name="Test User",
        email="test@example.com",
        phone="+1234567890",
        emergency_contact_phone="+0987654321",
        collection=users_collection
    )
    device_id = register_device(
        user_id=user_id,
        device_type="phone",
        sim_id="SIM123",
        battery_level=80
    )
    
    # Populate location data
    start_date = datetime.now(timezone.utc) - timedelta(days=60)
    for i in range(200):
        timestamp = start_date + timedelta(hours=i)
        is_night = timestamp.hour >= 22 or timestamp.hour < 4
        update_location(
            user_id=user_id,
            device_id=device_id,
            latitude=48.8 + (i % 10) * 0.01,
            longitude=2.3 + (i % 10) * 0.01,
            timestamp=timestamp
        )
        log_alert(
            user_id=user_id,
            device_id=device_id,
            latitude=48.8 + (i % 10) * 0.01,
            longitude=2.3 + (i % 10) * 0.01,
            incident_probability=0.7 if is_night else 0.3,
            is_incident=is_night and (i % 10 == 0),
            location_anomaly=0.5 + (i % 10) * 0.05,
            hour_anomaly=0.7 if is_night else 0.3,
            weekday_anomaly=0.4 + (i % 7) * 0.05,
            month_anomaly=0.2 + (i % 12) * 0.05,
            timestamp=timestamp,
            save_locally=False
        )
    
    yield user_id, device_id
    # Teardown
    users_collection.drop()
    locations_collection.drop()

@pytest.fixture(scope="module")
def train_models(setup_database):
    """Train incident and threshold models."""
    user_id, _ = setup_database
    # Train incident model
    features, labels = prepare_incident_data(user_id)
    assert features is not None, "Failed to prepare incident data"
    model, scaler, threshold = train_incident_model(features, labels)
    assert model is not None, "Failed to train incident model"
    save_incident_model(user_id, model, scaler, threshold, save_to_db=True)
    
    # Train threshold model
    features, labels = prepare_threshold_data(user_id)
    assert features is not None, "Failed to prepare threshold data"
    model, scaler = train_threshold_model(features, labels)
    threshold = predict_threshold(model, scaler, features[-1])
    save_threshold_model(user_id, model, scaler, threshold, save_to_db=True)
    
    return user_id

def test_preprocess_data(setup_database):
    """Test data preprocessing."""
    user_id, _ = setup_database
    df, X_scaled, hour_freq, weekday_freq, month_freq, scaler = preprocess_data(user_id)
    assert df is not None, "Failed to preprocess data"
    assert len(df) <= 100, f"Expected at least 100 records, got {len(df)}"
    assert X_scaled.shape[0] == len(df), "Mismatch in scaled data size"
    assert hour_freq, "Hour frequency dictionary is empty"

def test_build_user_profile(setup_database):
    """Test building user profile."""
    user_id, _ = setup_database
    centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id)
    assert centroids is not None, "Failed to build user profile"
    assert len(centroids) > 0, "No clusters found"
    assert hour_freq, "Hour frequency dictionary is empty"
    user = users_collection.find_one({"user_id": user_id})
    assert user["profile"]["centroids"], "Profile not saved to MongoDB"

def test_detect_user_anomalies(setup_database):
    """Test anomaly detection."""
    user_id, _ = setup_database
    timestamp = datetime.now(timezone.utc)
    anomalies = detect_user_anomalies(
        latitude=48.8,
        longitude=2.3,
        hour=timestamp.hour,
        weekday=timestamp.weekday(),
        month=timestamp.month,
        user_id=user_id,
        collection=locations_collection
    )
    assert anomalies is not None, "Failed to detect anomalies"
    assert len(anomalies) == 4, f"Expected 4 anomalies, got {len(anomalies)}"
    assert all(0 <= a <= 1 for a in anomalies), "Anomaly scores out of range"

def test_process_capture_sos(setup_database, train_models):
    """Test SOS alert processing."""
    user_id, device_id = setup_database
    timestamp = datetime.now(timezone.utc)
    result = process_capture(
        user_id=user_id,
        device_id=device_id,
        latitude=48.8,
        longitude=2.3,
        timestamp=timestamp,
        sos_pressed=True
    )
    assert result is not None, "Failed to process SOS alert"
    assert result["is_incident"] is True, "SOS alert not marked as incident"
    assert result["alert_id"], "Alert ID not generated"
    assert 0 <= result["incident_probability"] <= 1, "Invalid incident probability"
    assert locations_collection.find_one({"alert.alert_id": result["alert_id"]}), "Alert not saved to locations"
    assert users_collection.find_one({"user_id": user_id, "alerts.alert_id": result["alert_id"]}), "Alert not saved to users"

def test_process_capture_periodic(setup_database, train_models):
    """Test periodic check processing."""
    user_id, device_id = setup_database
    timestamp = datetime.now(timezone.utc)
    result = process_capture(
        user_id=user_id,
        device_id=device_id,
        latitude=48.9,
        longitude=2.4,
        timestamp=timestamp,
        sos_pressed=False
    )
    assert result is not None, "Failed to process periodic check"
    assert isinstance(result["is_incident"], bool), "Invalid is_incident value"
    assert result["alert_id"], "Alert ID not generated"
    assert 0 <= result["incident_probability"] <= 1, "Invalid incident probability"
    assert locations_collection.find_one({"alert.alert_id": result["alert_id"]}), "Alert not saved to locations"
    assert users_collection.find_one({"user_id": user_id, "alerts.alert_id": result["alert_id"]}), "Alert not saved to users"

def test_insufficient_data(setup_database):
    """Test handling of insufficient data."""
    user_id, _ = setup_database
    # Clear locations to simulate insufficient data
    locations_collection.delete_many({"user_id": user_id})
    timestamp = datetime.now(timezone.utc)
    result = process_capture(
        user_id=user_id,
        device_id="6c58c3d4-f9f7-4298-8c33-8dc5abc5442b",
        latitude=48.8,
        longitude=2.3,
        timestamp=timestamp,
        sos_pressed=False
    )
    assert result is not None, "Failed to process with insufficient data"
    assert result["incident_probability"] == 1.0, "Expected default probability of 1.0"
    assert locations_collection.find_one({"alert.alert_id": result["alert_id"]}), "Alert not saved to locations"

def test_missing_model(setup_database):
    """Test handling of missing incident/threshold models."""
    user_id, device_id = setup_database
    # Remove models
    user_dir = os.path.join(MODEL_DIR, user_id)
    for file in os.listdir(user_dir):
        os.remove(os.path.join(user_dir, file))
    users_collection.update_one({"user_id": user_id}, {"$unset": {"incident_model": "", "incident_scaler": "", "threshold_model": "", "threshold_scaler": ""}})
    
    timestamp = datetime.now(timezone.utc)
    result = process_capture(
        user_id=user_id,
        device_id=device_id,
        latitude=48.8,
        longitude=2.3,
        timestamp=timestamp,
        sos_pressed=False
    )
    assert result is not None, "Failed to process with missing models"
    assert result["threshold"] == DEFAULT_PROB_THRESHOLD, f"Expected default threshold {DEFAULT_PROB_THRESHOLD}"
    assert 0 <= result["incident_probability"] <= 1, "Invalid default probability"

if __name__ == "__main__":
    pytest.main(["-v", "behavioral_alerts/core/test_file.py"])