import pandas as pd
from fusion_core import *


import sys, os
from pathlib import Path

proj_root = Path(os.getcwd()).parent
sys.path.insert(0, str(proj_root))

from ..behavioral_alerts.core.processsing import  process_capture
from .vocal_analysis import analyze_vocal
from .risky_location_inference import predict_risk
import logging
import numpy as np
from datetime import datetime, timezone
import random
import torch
import librosa
import whisper




# ==============================================================================
# THE INTEGRATED Core Function
# ==============================================================================

def process_capture_all_inclusive(user_id, device_id, latitude, longitude, sos_pressed=False, audio_path=None):
    """
    Processes a capture event using a multi-signal approach: user anomaly, 
    geospatial risk, and real-time vocal analysis.
    
    Args:
        user_id (str): The ID of the user.
        device_id (str): The ID of the device.
        latitude (float): The current latitude.
        longitude (float): The current longitude.
        sos_pressed (bool): True if the user manually triggered an SOS.
        audio_path (str, optional): Path to an audio file for vocal analysis. Defaults to None.

    Returns:
        dict: A dictionary summarizing the analysis and final incident decision.
    """
    incident_probabiloty,is_incident,location_anomaly, threshold = process_capture(
        user_id, device_id, latitude, longitude, sos_pressed
    )

    # --- 2. Geospatial Risk Analysis (New) ---
    try:
        geo_risk_score, _, geo_event_type = predict_risk(latitude, longitude)
    except Exception as e:
        logger.error(f"Error predicting geospatial risk: {e}")
        geo_risk_score, geo_event_type = 0.0, 'ERROR'

    # --- 3. Vocal Analysis (New) ---
    vocal_analysis_results = analyze_vocal(audio_path)

    # --- 4. Combined Incident Decision Logic (Updated) ---
    # An incident is triggered if ANY of the following are true.
    user_anomaly_triggered = (incident_probability >= threshold and location_anomaly > 0.6)
    geo_risk_triggered = (geo_risk_score > 0.75) # Location is a known high-risk zone
    
    vocal_alert_triggered = False
    if vocal_analysis_results:
        text_class = vocal_analysis_results['classification']['label']
        stress_label = vocal_analysis_results['audio_features']['stress']['label']
        # Trigger if text is dangerous OR vocal stress indicates anger/fear.
        if text_class in ['danger', 'hate'] or stress_label in ['ang', 'fea']:
            vocal_alert_triggered = True

    # Final Decision: An incident is declared if any system flags it.
    is_incident = sos_pressed or user_anomaly_triggered or geo_risk_triggered or vocal_alert_triggered

    # --- 5. Enhanced Logging and Return Value ---
    alert_data = {
        "user_id": user_id,
        "is_incident": is_incident,
        "triggers": {
            "sos_pressed": sos_pressed,
            "user_anomaly": user_anomaly_triggered,
            "geo_risk": geo_risk_triggered,
            "vocal_alert": vocal_alert_triggered
        },
        "user_anomaly_details": {
            "probability": float(incident_probability),
            "threshold": threshold,
            "location_anomaly": location_anomaly
        },
        "geo_risk_details": {
            "score": geo_risk_score,
            "predicted_event": geo_event_type
        },
        "vocal_analysis_summary": {
            "classification": vocal_analysis_results['classification']['label'] if vocal_analysis_results else None,
            "stress": vocal_analysis_results['audio_features']['stress']['label'] if vocal_analysis_results else None
        }
    }
    log_alert(**alert_data)
    
    return alert_data


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running Scenario 1: No audio, low-risk situation ---")
    # Mock a normal situation
    result1 = process_capture(
        user_id="user_123", 
        device_id="device_abc", 
        latitude=35.82, 
        longitude=10.63
    )
    print(f"Final Decision: {'Incident' if result1['is_incident'] else 'Normal'}\n")

    print("\n--- Running Scenario 2: Audio indicates distress in a high-risk area ---")
    # Mock a dangerous situation with audio input
    result2 = process_capture(
        user_id="user_456", 
        device_id="device_xyz", 
        latitude=36.80, 
        longitude=10.18,
        audio_path="/path/to/distress_call.wav" # Path is for logic, no file is actually read by the mock
    )
    print(f"Final Decision: {'Incident' if result2['is_incident'] else 'Normal'}\n")

    print("\n--- Running Scenario 3: User presses the SOS button ---")
    result3 = process_capture(
        user_id="user_789", 
        device_id="device_qwe", 
        latitude=34.74, 
        longitude=10.76,
        sos_pressed=True
    )
    print(f"Final Decision: {'Incident' if result3['is_incident'] else 'Normal'}\n")