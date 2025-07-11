from behavioral_alerts.core.threshold_adjustment import (
    load_threshold_model,
    predict_threshold,
    prepare_threshold_data
)
from behavioral_alerts.core.incident_prediction import (
    load_incident_model,
    predict_incident
)
import numpy as np

def evaluate_user_behavior(
    user_id,
    location_score,
    time_score,
    ts_collection=None,
    incident_model=None,
    scaler=None,
    threshold_model=None
):
    # Load models only if not provided
    if not incident_model or not scaler:
        incident_model, scaler = load_incident_model(user_id)
    if not threshold_model:
        threshold_model = load_threshold_model(user_id)

    # Ensure models are loaded
    if not incident_model or not scaler or not threshold_model:
        return {
            "error": f"Missing model(s) for user {user_id}"
        }

    # Predict incident probability
    prob = predict_incident(incident_model, scaler, location_score, time_score)

    # Use actual data if ts_collection is provided
    if ts_collection is not None:
        threshold_features, _ = prepare_threshold_data(ts_collection, user_id)
        if threshold_features is None:
            return {
                "error": f"Not enough data for threshold prediction for {user_id}"
            }
        threshold_input = threshold_features[0]
    else:
        # Dummy/fallback values
        hour_std = np.std([8, 9, 11, 10])
        location_transitions = 0.4
        data_volume = 50
        threshold_input = [hour_std, location_transitions, data_volume]

    # Predict threshold
    threshold = predict_threshold(threshold_model, threshold_input)

    # Compare and return result
    return {
        "incident_probability": prob,
        "dynamic_threshold": threshold,
        "anomaly": prob >= threshold
    }
