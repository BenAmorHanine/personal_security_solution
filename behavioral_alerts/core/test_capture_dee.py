# test_safety_system.py
import uuid
from datetime import datetime, timedelta, timezone
import random
import numpy as np

# Import your project modules
from db_functions import create_user, register_device, update_location, log_alert
from profiling import build_user_profile, detect_user_anomalies
from incident_prediction import train_incident_model, save_incident_model, predict_incident
from threshold_adjustment import adjust_threshold, save_threshold_model

# Configuration (replace with your actual values)
MONGO_URI = "your_mongodb_uri"
LOG_DIR = "./alert_logs"
MODEL_DIR = "./models"
USER_NAME = "Test User"
USER_EMAIL = f"test_{uuid.uuid4().hex[:8]}@example.com"
USER_PHONE = "+1234567890"
EMERGENCY_PHONE = "+0987654321"
DEVICE_TYPE = "phone"
SIM_ID = "sim123"
BATTERY_LEVEL = 100

def generate_test_data(user_id, device_id, days=30):
    """Generate test location data for the user"""
    now = datetime.now(timezone.utc)
    locations = []
    
    # Home location (common)
    home_lat, home_lon = 48.8566, 2.3522  # Paris coordinates
    
    # Work location
    work_lat, work_lon = 48.8576, 2.3512
    
    # Generate timestamps
    for i in range(days):
        for hour in [8, 12, 18]:  # Morning, noon, evening
            timestamp = now - timedelta(days=i, hours=hour)
            
            # During work hours (9-17) at work location
            if 9 <= hour <= 17:
                lat, lon = work_lat, work_lon
            else:  # Otherwise at home
                lat, lon = home_lat, home_lon
            
            # Add some random variation
            lat += random.uniform(-0.01, 0.01)
            lon += random.uniform(-0.01, 0.01)
            
            update_location(user_id, device_id, lat, lon, timestamp=timestamp)
            locations.append((lat, lon, timestamp))
    
    return locations

def process_capture(user_id, device_id, latitude, longitude, sos_pressed=False):
    """Process a location capture event"""
    timestamp = datetime.now(timezone.utc)
    hour = timestamp.hour
    weekday = timestamp.weekday()
    month = timestamp.month
    
    # Update location
    location_id = update_location(user_id, device_id, latitude, longitude, timestamp)
    
    # Detect anomalies
    location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly = detect_user_anomalies(
        latitude, longitude, hour, weekday, month, user_id
    )
    
    # Predict incident probability
    incident_probability = predict_incident(user_id, location_anomaly, hour_anomaly, weekday_anomaly, month_anomaly)
    
    # Determine if this should be considered an incident
    is_incident = sos_pressed or (incident_probability > 0.7)  # Threshold can be adjusted
    
    # Log alert
    alert_id = log_alert(
        user_id=user_id,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        incident_probability=incident_probability,
        is_incident=is_incident,
        location_anomaly=location_anomaly,
        hour_anomaly=hour_anomaly,
        weekday_anomaly=weekday_anomaly,
        month_anomaly=month_anomaly,
        timestamp=timestamp,
        save_locally=True
    )
    
    return alert_id, incident_probability, is_incident

def main():
    print("Starting safety system test...")
    
    # Step 1: Create user
    user_id = create_user(USER_NAME, USER_EMAIL, USER_PHONE, EMERGENCY_PHONE)
    print(f"Created user: {user_id}")
    
    # Step 2: Register device
    device_id = register_device(user_id, DEVICE_TYPE, SIM_ID, BATTERY_LEVEL)
    print(f"Registered device: {device_id}")
    
    # Step 3: Generate test location data (normal behavior)
    print("Generating test location data...")
    normal_locations = generate_test_data(user_id, device_id, days=30)
    
    # Step 4: Build user profile
    print("Building user profile...")
    build_user_profile(user_id)
    
    # Step 5: Train incident prediction model
    print("Training incident prediction model...")
    features, labels = prepare_incident_data(user_id)
    model, scaler, threshold = train_incident_model(features, labels)
    
    if model and scaler:
        save_incident_model(user_id, model, scaler, threshold, save_to_db=True)
        print("Incident model trained and saved")
    
    # Step 6: Test normal capture (non-SOS)
    print("\nTesting normal capture...")
    normal_lat, normal_lon, _ = normal_locations[0]
    alert_id, prob, is_incident = process_capture(
        user_id, device_id, normal_lat, normal_lon, sos_pressed=False
    )
    print(f"Normal capture result: Alert ID={alert_id}, Probability={prob:.2f}, Incident={is_incident}")
    
    # Step 7: Test SOS capture
    print("\nTesting SOS capture...")
    alert_id, prob, is_incident = process_capture(
        user_id, device_id, normal_lat, normal_lon, sos_pressed=True
    )
    print(f"SOS capture result: Alert ID={alert_id}, Probability={prob:.2f}, Incident={is_incident}")
    
    # Step 8: Test anomalous capture
    print("\nTesting anomalous capture...")
    anomalous_lat, anomalous_lon = 48.8600, 2.3600  # Different location
    alert_id, prob, is_incident = process_capture(
        user_id, device_id, anomalous_lat, anomalous_lon, sos_pressed=False
    )
    print(f"Anomalous capture result: Alert ID={alert_id}, Probability={prob:.2f}, Incident={is_incident}")
    
    # Step 9: Test threshold adjustment
    print("\nAdjusting threshold...")
    best_threshold = adjust_threshold(user_id)
    print(f"Optimal threshold: {best_threshold:.2f}")
    
    # Save threshold model
    features, labels = prepare_threshold_data(user_id)
    if features is not None:
        model, scaler = train_threshold_model(features, labels)
        save_threshold_model(user_id, model, scaler, best_threshold, save_to_db=True)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()