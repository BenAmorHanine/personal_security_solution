import pytest
import random
import uuid
from datetime import datetime, timezone, timedelta

from .db_functions import (
    create_user,
    register_device,
    update_location,
    log_alert,
    drop_user,
)
from .profiling import (
    build_user_profile,
    detect_user_anomalies,
    preprocess_data,
)
from .incident_prediction import predict_incident
from .threshold_adjustment import predict_threshold
from .capture import process_capture

# === Helper: Generate realistic data for user ===
def generate_synthetic_data(user_id: str, device_id: str, num_points=30, num_alerts=10):
    for _ in range(num_points):
        lat = random.uniform(48.7, 49.0)
        lon = random.uniform(2.2, 2.5)
        timestamp = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
        update_location(user_id, device_id, lat, lon, timestamp)

    for _ in range(num_alerts):
        lat = random.uniform(48.7, 49.0)
        lon = random.uniform(2.2, 2.5)
        timestamp = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
        log_alert(
            user_id=user_id,
            device_id=device_id,
            latitude=lat,
            longitude=lon,
            timestamp=timestamp,
            incident_probability=random.random(),
            is_incident=random.choice([True, False]),
            location_anomaly=random.random(),
            hour_anomaly=random.random(),
            weekday_anomaly=random.random(),
            month_anomaly=random.random(),
        )

"""
# === Test: Preprocess data ===
def test_preprocess_data():
    user_id = create_user("Test", "test@example.com", "0600000000", "0700000000")
    device_id = register_device(user_id, "test-phone", "sim123", 80)
    generate_synthetic_data(user_id, device_id)

    data = preprocess_data(user_id)
    drop_user(user_id)

    assert isinstance(data, list)
    assert all(len(d) == 6 for d in data), "Each data point should have 6 values"

"""
# === Test: Profile Building ===
def test_build_user_profile():
    sample_data = [
        ("user1", "sos", "2025-01-01T12:00:00Z", 48.8566, 2.3522, True),
        ("user1", "periodic", "2025-01-01T13:00:00Z", 48.8570, 2.3530, False),
        ("user1", "periodic", "2025-01-01T14:00:00Z", 48.8580, 2.3540, False)
    ]
    profile = build_user_profile(sample_data)
    
    assert isinstance(profile, dict)
    assert "hour" in profile
    assert "weekday" in profile
    assert "month" in profile

# === Test: Anomaly Detection ===
def test_detect_user_anomalies():
    sample_data = [
        ("user1", "periodic", "2025-01-01T12:00:00Z", 48.8566, 2.3522, False),
        ("user1", "periodic", "2025-01-01T13:00:00Z", 48.8570, 2.3530, False),
        ("user1", "periodic", "2025-01-01T23:00:00Z", 48.8600, 2.3560, False)  # anomalous hour
    ]
    profile = build_user_profile(sample_data)
    anomalies = detect_user_anomalies(profile, sample_data)
    
    assert isinstance(anomalies, list)
    assert len(anomalies) >= 1

# === Test: SOS alert processing ===
def test_process_capture_sos():
    sample_data = [
        ("user1", "sos", "2025-01-01T12:00:00Z", 48.8566, 2.3522, True)
    ]
    user_profile = build_user_profile(sample_data)
    processed = process_capture("user1", "device1", sample_data, user_profile, "sos")
    
    assert isinstance(processed, list)
    assert len(processed) == 1
    assert processed[0]["is_incident"] is True


# === Test: Periodic check processing ===
def test_process_capture_periodic():
    user_id = create_user("PeriodicTest", "periodic@example.com", "0600000000", "0700000000")
    device_id = register_device(user_id, "test-phone", "sim202", 75)
    generate_synthetic_data(user_id, device_id)

    alert = {
        "type": "check",
        "battery_level": 70,
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc),
        "location": {"latitude": 48.85, "longitude": 2.35}
    }

    data = preprocess_data(user_id)
    profile = build_user_profile(user_id)
    anomalies = detect_user_anomalies(data, profile)
    prediction = predict_threshold(anomalies, profile)
    drop_user(user_id)

    assert "adjusted_threshold" in prediction


# === Test: Handle insufficient data gracefully ===
def test_insufficient_data():
    user_id = create_user("EmptyTest", "empty@example.com", "0600000000", "0700000000")
    device_id = register_device(user_id, "test-phone", "sim303", 50)

    data = preprocess_data(user_id)
    profile = build_user_profile(user_id)
    drop_user(user_id)

    assert data == [] or profile == {}, "Should handle lack of data gracefully"


# === Test: Handle missing model input ===
def test_missing_model():
    try:
        predict_incident([], {})
        predict_threshold([], {})
    except Exception as e:
        pytest.fail(f"Missing model call raised an error: {e}")

if __name__ == "__main__":
    pytest.main(["-v", "behavioral_alerts/core/test_file2.py"])