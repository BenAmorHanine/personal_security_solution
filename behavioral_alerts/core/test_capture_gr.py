import random
from datetime import datetime, timedelta, timezone
from .db_functions import create_user, register_device, update_location, log_alert
from .profiling import build_user_profile, detect_user_anomalies
from .incident_prediction import prepare_incident_data, train_incident_model, save_incident_model, predict_incident
from .threshold_adjustment import adjust_threshold, train_threshold_model, load_threshold_model,prepare_threshold_data, predict_threshold,save_threshold_model
from .incident_classifier import load_incident_classifier, save_incident_classifier, optimize_incident_threshold
from pymongo import MongoClient
from .config import MONGO_URI, DEFAULT_PROB_THRESHOLD

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]
locations_collection = db["locations"]

def process_capture_original(user_id, device_id, latitude, longitude, sos_pressed=False):
    """Simulate processing a capture event."""
    # Update location
    update_location(user_id, device_id, latitude, longitude)
    
    # Get current time features
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    month = now.month
    
    # Detect anomalies
    location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = detect_user_anomalies(
        latitude, longitude, hour, weekday, month, user_id
    )
    
    # Predict incident probability
    incident_probability = predict_incident(
        user_id, location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly
    )
    
    # Load optimal threshold
    user_doc = users_collection.find_one({"user_id": user_id})
    optimal_threshold = user_doc.get("optimal_threshold", 0.5)
    
    # Determine if it's an incident
    is_incident = sos_pressed or (incident_probability > optimal_threshold)
    
    # Log the alert
    log_alert(
        user_id=user_id,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        incident_probability=incident_probability,
        is_incident=is_incident,
        location_anomaly=location_anomaly,
        hour_anomaly=hour_anomaly,
        weekday_anomaly=weekday_anomaly,
        month_anomaly=month_anomaly
    )
    
    return {"incident_probability": incident_probability, "is_incident": is_incident}


def process_capture_updated(user_id, device_id, latitude, longitude, sos_pressed=False):
    """Simulate processing a capture event."""
    # Update location
    update_location(user_id, device_id, latitude, longitude)
    
    # Get current time features
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    month = now.month
    
    # Detect anomalies
    location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = detect_user_anomalies(
        latitude, longitude, hour, weekday, month, user_id
    )
    
    # Predict incident probability
    incident_probability = predict_threshold(
        user_id, location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly
    ) or DEFAULT_PROB_THRESHOLD
    

    
    # Determine if it's an incident
    is_incident = sos_pressed or (incident_probability > optimal_threshold)
    
    # Log the alert
    log_alert(
        user_id=user_id,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        incident_probability=incident_probability,
        is_incident=is_incident,
        location_anomaly=location_anomaly,
        hour_anomaly=hour_anomaly,
        weekday_anomaly=weekday_anomaly,
        month_anomaly=month_anomaly
    )
    
    return {"incident_probability": incident_probability, "is_incident": is_incident}

def process_capture(user_id, device_id, latitude, longitude, sos_pressed=False, use_threshold_model=True):
    """Simulate processing a capture event."""
    # Update location
    update_location(user_id, device_id, latitude, longitude)
    
    # Get current time features
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    month = now.month
    
    # Detect anomalies
    location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = detect_user_anomalies(
        latitude, longitude, hour, weekday, month, user_id
    )
    
    # Load threshold or model
    user_doc = users_collection.find_one({"user_id": user_id})
    if use_threshold_model:
        model, scaler = load_threshold_model(user_id, fallback_to_train=True)
        if model is None or scaler is None:
            print(f"[✗] Could not load or train threshold model for user {user_id}. Using default threshold.")
            optimal_threshold = DEFAULT_PROB_THRESHOLD  # e.g., 0.5 from config
            incident_probability = 0.0
        else:
            features = [location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly]
            incident_probability = predict_threshold(model, scaler, features)
            optimal_threshold = user_doc.get("threshold_value", DEFAULT_PROB_THRESHOLD)
    else:
        incident_probability = predict_incident(
            user_id, location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly
        )
        optimal_threshold = user_doc.get("optimal_threshold", DEFAULT_PROB_THRESHOLD)
    
    # Determine if it's an incident
    is_incident = sos_pressed or (incident_probability > optimal_threshold)
    
    # Log the alert
    log_alert(
        user_id=user_id,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        incident_probability=incident_probability,
        is_incident=is_incident,
        location_anomaly=location_anomaly,
        hour_anomaly=hour_anomaly,
        weekday_anomaly=weekday_anomaly,
        month_anomaly=month_anomaly
    )
    
    return {"incident_probability": incident_probability, "is_incident": is_incident}
# Step 1: Create a fake user
user_id = create_user(
    name="Test User",
    email="test@example.com",
    phone="+1234567890",
    emergency_contact_phone="+0987654321"
)
print(f"Created user with ID: {user_id}")

# Step 2: Register a device
device_id = register_device(
    user_id=user_id,
    device_type="smartphone",
    sim_id="123456789012345",
    battery_level=100
)
print(f"Registered device with ID: {device_id}")

# Step 3: Generate historical location data (100 points over 35 days)
for _ in range(100):
    days_ago = random.uniform(0, 35)
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    latitude = 40.0 + random.uniform(-0.01, 0.01)  # Normal location
    longitude = -74.0 + random.uniform(-0.01, 0.01)
    update_location(user_id, device_id, latitude, longitude, timestamp=timestamp)
print("Generated 100 historical location points.")

# Step 4: Build the user's profile
centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile(user_id)
print("Built user profile.")

# Step 5: Generate 50 fake alerts (40 normal, 10 anomalous)
for _ in range(50):
    days_ago = random.uniform(0, 90)
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    hour = timestamp.hour
    weekday = timestamp.weekday()
    month = timestamp.month
    
    if random.random() < 0.2:  # 20% chance of anomalous location
        latitude = 40.0 + random.uniform(-0.5, 0.5)
        longitude = -74.0 + random.uniform(-0.5, 0.5)
        is_incident = random.random() < 0.8  # 80% chance of incident
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
        incident_probability=0.0,  # Dummy value for training data
        is_incident=is_incident,
        location_anomaly=location_anomaly,
        hour_anomaly=hour_anomaly,
        weekday_anomaly=weekday_anomaly,
        month_anomaly=month_anomaly,
        timestamp=timestamp
    )
print("Generated 50 fake alerts.")

# Step 6: Train the incident prediction model
anomaly_features, incident_labels = prepare_incident_data(user_id)
if anomaly_features is not None:
    model, scaler, optimal_threshold = train_incident_model(anomaly_features, incident_labels)
    save_incident_model(user_id, model, scaler, optimal_threshold, save_to_db=True)
    print(f"Trained and saved incident model with optimal threshold: {optimal_threshold}")
else:
    print("Insufficient data to train model.")

# Step 7: Test the process_capture function
# Normal location, sos_pressed=False
latitude = 40.0 + random.uniform(-0.01, 0.01)
longitude = -74.0 + random.uniform(-0.01, 0.01)
result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=False)
print(f"Normal location, sos_pressed=False: {result}")

# Normal location, sos_pressed=True
result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=True)
print(f"Normal location, sos_pressed=True: {result}")

# Anomalous location, sos_pressed=False
latitude = 40.0 + random.uniform(-0.5, 0.5)
longitude = -74.0 + random.uniform(-0.5, 0.5)
result = process_capture(user_id, device_id, latitude, longitude, sos_pressed=False)
print(f"Anomalous location, sos_pressed=False: {result}")