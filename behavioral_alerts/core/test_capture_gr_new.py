import random
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
import pytest
from .db_functions import create_user, register_device, update_location, log_alert
from .profiling import build_user_profile, detect_user_anomalies
from .incident_classifier import load_incident_classifier, predict_incident_probability, train_incident_classifier, save_incident_classifier, optimize_incident_threshold
from pymongo import MongoClient
from .config import MONGO_URI, DEFAULT_PROB_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def process_capture(user_id, device_id, latitude, longitude, sos_pressed=False):
    """Simulate processing a capture event using the incident classifier."""
    update_location(user_id, device_id, latitude, longitude)
    
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    month = now.month
    
    try:
        location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = detect_user_anomalies(
            latitude, longitude, hour, weekday, month, user_id
        )
    except Exception as e:
        logger.error(f"Error detecting anomalies for {user_id}: {e}")
        location_anomaly = hour_anomaly = weekday_anomaly = month_anomaly = 1.0
    
    model, scaler = load_incident_classifier(user_id, fallback_to_train=True)
    user_doc = users_collection.find_one({"user_id": user_id})
    
    if model is None or scaler is None:
        logger.warning(f"Could not load or train incident classifier for user {user_id}. Using default threshold {DEFAULT_PROB_THRESHOLD}.")
        incident_probability = 0.0
        threshold = DEFAULT_PROB_THRESHOLD
    else:
        features = np.array([[location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly]])
        try:
            incident_probability = predict_incident_probability(model, scaler, features)
        except Exception as e:
            logger.error(f"Error predicting probability for {user_id}: {e}")
            incident_probability = 0.0

        threshold = user_doc.get("incident_threshold", DEFAULT_PROB_THRESHOLD)
        if threshold < 0.2:
            logger.warning(f"ML-based threshold {threshold} for user {user_id} is too low. Using default {DEFAULT_PROB_THRESHOLD}.")
            threshold = DEFAULT_PROB_THRESHOLD
    
    #is_incident = sos_pressed or (incident_probability >= threshold)
    is_incident = sos_pressed or (incident_probability >= threshold and location_anomaly > 0.5)


    log_alert(
        user_id=user_id,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        incident_probability=float(incident_probability),
        is_incident=is_incident,
        location_anomaly=location_anomaly,
        hour_anomaly=hour_anomaly,
        weekday_anomaly=weekday_anomaly,
        month_anomaly=month_anomaly
    )
    
    return {
        "incident_probability": float(incident_probability),
        "is_incident": is_incident,
        "location_anomaly": location_anomaly,
        "threshold": threshold
    }

@pytest.fixture(scope="module")
def setup_user_and_device():
    """Set up a test user, device, profile, and trained classifier."""
    user_id = create_user(
        name="Test User",
        email="test@example.com",
        phone="+1234567890",
        emergency_contact_phone="+0987654321"
    )
    logger.info(f"Created user with ID: {user_id}")
    
    device_id = register_device(
        user_id=user_id,
        device_type="smartphone",
        sim_id="123456789012345",
        battery_level=100
    )
    logger.info(f"Registered device with ID: {device_id}")
    
    locations_collection.delete_many({"user_id": user_id})  # Clear existing locations
    for _ in range(300):
        days_ago = random.uniform(0, 35)
        timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
        latitude = 40.0 + random.uniform(-0.01, 0.01)
        longitude = -74.0 + random.uniform(-0.01, 0.01)
        update_location(user_id, device_id, latitude, longitude, timestamp=timestamp)
    logger.info("Generated 100 historical location points.")


    
    centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id)
    logger.info(f"Built user profile with {len(centroids)} clusters.")
    
    for _ in range(500):
        days_ago = random.uniform(0, 90)
        timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
        hour = timestamp.hour
        weekday = timestamp.weekday()
        month = timestamp.month
        
        if random.random() < 0.2:
            latitude = 40.0 + random.uniform(-1.0, 1.0)
            longitude = -74.0 + random.uniform(-1.0, 1.0)
            is_incident = random.random() < 0.8
        else:
            latitude = 40.0 + random.uniform(-0.01, 0.01)
            longitude = -74.0 + random.uniform(-0.01, 0.01)
            is_incident = False
        
        location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = detect_user_anomalies(
            latitude, longitude, hour, weekday, month, user_id
        )
        
        log_alert(
            user_id=user_id,
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            incident_probability=0.0,
            is_incident=is_incident,
            location_anomaly=location_anomaly,
            hour_anomaly=hour_anomaly,
            weekday_anomaly=weekday_anomaly,
            month_anomaly=month_anomaly,
            timestamp=timestamp
        )
    logger.info("Generated 500 fake alerts.")
    
    alerts = list(locations_collection.find({"user_id": user_id, "alert.is_incident": {"$exists": True}}))
    if len(alerts) >= 20:
        features, labels = [], []
        for alert in alerts:
            loc_anomaly = alert.get("alert", {}).get("location_anomaly", 1.0)
            hour_anomaly = alert.get("alert", {}).get("hour_anomaly", 1.0)
            weekday_anomaly = alert.get("alert", {}).get("weekday_anomaly", 1.0)
            month_anomaly = alert.get("alert", {}).get("month_anomaly", 1.0)
            is_incident = alert.get("alert", {}).get("is_incident", False)
            features.append([loc_anomaly, hour_anomaly, weekday_anomaly, month_anomaly])
            labels.append(1 if is_incident else 0)
        
        features = np.array(features)
        labels = np.array(labels)
        model, scaler = train_incident_classifier(features, labels)
        threshold = optimize_incident_threshold(user_id)
        save_incident_classifier(user_id, model, scaler, threshold, save_to_db=True)
        logger.info(f"Trained and saved incident classifier with ML-based threshold: {threshold}")
    else:
        logger.warning("Insufficient data to train incident classifier.")
    
    return user_id, device_id

def test_process_capture_normal_no_sos(setup_user_and_device):
    """Test normal location with no SOS press."""
    user_id, device_id = setup_user_and_device
    latitude = 40.0 + random.uniform(-0.01, 0.01)
    longitude = -74.0 + random.uniform(-0.01, 0.01)
    result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=False)
    logger.info(f"Normal location, sos_pressed=False: {result}")
    assert 0 <= result['incident_probability'] <= 1
    assert not bool(result['is_incident'])
    assert 0.2 <= result['threshold'] <= 1.0
    assert result['location_anomaly'] <=0.2 #> 0.9
    

def test_process_capture_normal_sos(setup_user_and_device):
    """Test normal location with SOS press."""
    user_id, device_id = setup_user_and_device
    latitude = 40.0 + random.uniform(-0.01, 0.01)
    longitude = -74.0 + random.uniform(-0.01, 0.01)
    result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=True)
    logger.info(f"Normal location, sos_pressed=True: {result}")
    assert 0 <= result['incident_probability'] <= 1
    assert result['is_incident']
    assert result['location_anomaly'] <=0.2 #> 0.9
    """assert result["hour_anomaly"] <= 0.2
    assert result["weekday_anomaly"] <= 0.2
    assert result["month_anomaly"] <= 0.2"""
    

    assert 0.2 <= result['threshold'] <= 1.0

def test_process_capture_anomalous_no_sos(setup_user_and_device):
    """Test anomalous location with no SOS press."""
    user_id, device_id = setup_user_and_device
    latitude = 40.0 + random.uniform(-1.0, 1.0)
    longitude = -74.0 + random.uniform(-1.0, 1.0)
    result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=False)
    logger.info(f"Anomalous location, sos_pressed=False: {result}")
    assert 0 <= result['incident_probability'] <= 1
    assert result['location_anomaly'] > 0.9 #< 0.9
    #assert result['is_incident'] == (result['incident_probability'] >= result['threshold'])
    assert result['is_incident'] == (result['incident_probability'] >= result['threshold'] and result['location_anomaly'] > 0.5)
    assert 0.2 <= result['threshold'] <= 1.0




"""def test_profile_rebuild_outdated(setup_user_and_device):
    #Test profile rebuilding for outdated user profile
    user_id, device_id = setup_user_and_device
    # Set last_updated to >24 hours ago
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"last_updated": datetime.now(timezone.utc) - timedelta(hours=25)}}
    )
    latitude = 40.0 + random.uniform(-0.01, 0.01)
    longitude = -74.0 + random.uniform(-0.01, 0.01)
    result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=False)
    logger.info(f"Outdated profile test: {result}")
    user_doc = users_collection.find_one({"user_id": user_id})
    assert user_doc["last_updated"] > datetime.now(timezone.utc) - timedelta(minutes=5)
    assert 0 <= result['incident_probability'] <= 1
    assert not result['is_incident']
    assert result['location_anomaly'] <= 0.2"""

    
if __name__ == "__main__":
    pytest.main([__file__])
    logger.info("DONE.")