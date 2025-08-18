import random
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
import pytest
from db_functions import create_user, register_device, update_location, log_alert
from profiling import build_user_profile, detect_user_anomalies
from incident_classifier import load_incident_classifier, predict_incident_probability, train_incident_classifier, save_incident_classifier, optimize_incident_threshold
from pymongo import MongoClient
from config import MONGO_URI, DEFAULT_PROB_THRESHOLD
import logging
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


    """log_alert(
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
    )"""
    
    return {
        "incident_probability": float(incident_probability),
        "is_incident": is_incident,
        "location_anomaly": location_anomaly,
        "threshold": threshold
    }
