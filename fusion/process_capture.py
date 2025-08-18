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

    # --- 1. User Anomaly Detection ---
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
    vocal_analysis_results = None
    vocal_alert_triggered = False

    if audio_path:
        try:
            # Run the vocal analysis
            vocal_analysis_results = analyze_vocal(audio_path)

            if vocal_analysis_results:
                # --- Extract all relevant labels safely ---
                classification = vocal_analysis_results.get('classification', {}) or {}
                audio_features = vocal_analysis_results.get('audio_features', {}) or {}

                text_class   = classification.get('label')
                stress_label = (audio_features.get('stress') or {}).get('label')
                rhythm_label = (audio_features.get('rhythm') or {}).get('label')
                tone_label   = (audio_features.get('tone') or {}).get('label')

                # --- Define danger signal conditions ---
                danger_conditions = [
                    text_class in ['danger', 'hate'],
                    stress_label in ['ang', 'fea'],
                    rhythm_label == 'fast',
                    tone_label == 'fearful',
                ]

                # Trigger danger signal if any condition is True
                vocal_alert_triggered = any(danger_conditions)

        except Exception as e:
            logger.error(f"Vocal analysis failed: {e}")



    # --- 4. Combined Incident Decision Logic ---
    # An incident is triggered if ANY of the following are true.
    user_anomaly_triggered = (incident_probability >= threshold and location_anomaly > 0.6)
    geo_risk_triggered = (geo_risk_score > 0.7) # Location is a known high-risk zone
    vocal_alert_triggered = vocal_alert_triggered

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
            "is_alert_triggered": vocal_alert_triggered,
            "classification": vocal_analysis_results['classification']['label'] if vocal_analysis_results else None,
            "stress": vocal_analysis_results['audio_features']['stress']['label'] if vocal_analysis_results else None,
            "rhythm": vocal_analysis_results['audio_features']['rhythm']['label'] if vocal_analysis_results else None,
            "tone": vocal_analysis_results['audio_features']['tone']['label'] if vocal_analysis_results

        }
    }
    log_full_alert(**alert_data)
    
    return alert_data

