from behavioral_alerts.core.profiling import build_user_profile, save_profile, should_retrain, load_profile
from behavioral_alerts.core.incident_prediction import prepare_incident_data, train_incident_model, save_incident_model
from behavioral_alerts.core.threshold_adjustment import prepare_threshold_data, train_threshold_model, save_threshold_model
from behavioral_alerts.core.utils import setup_timeseries_collection, setup_users_collection
from datetime import datetime

# Setup collections
ts_collection = setup_timeseries_collection()
users_collection = setup_users_collection()

# Example user ID (could be parameterized or iterated over multiple users)
user_id = "user1"

# Retrieve last training time (e.g., from DB; for now, assume None)
last_trained = None  # Replace with actual retrieval logic if available

# Check if retraining is needed
if should_retrain(ts_collection, user_id, last_trained):
    # Build user profile and train clustering model
    profile = build_user_profile(user_id, ts_collection, save_to_mongo=True)
    if profile[0] is not None:
        centroids, hour_freq, weekday_freq, month_freq, scaler = profile
        
        # Note: clustering_model is trained within build_user_profile but not returned.
        # For now, we load it from MongoDB/local storage after saving, or modify build_user_profile to return it.
        # Here, we assume it’s saved via save_profile inside build_user_profile.
        clustering_model, scaler = load_profile(user_id)  # Load the saved model if not returned
        
        # Save profile (already done in build_user_profile if save_to_mongo=True, but ensure consistency)
        save_profile(user_id, clustering_model, scaler, save_to_mongo=True, users_collection=users_collection)
        
        # Train and save incident prediction model
        features, labels = prepare_incident_data(users_collection, user_id)
        if features is not None:
            incident_model, incident_scaler = train_incident_model(features, labels)
            save_incident_model(user_id, incident_model, incident_scaler, save_to_mongo=True, users_collection=users_collection)
        
        # Train and save threshold model
        thresh_features, targets = prepare_threshold_data(ts_collection, user_id)
        if thresh_features is not None:
            threshold_model = train_threshold_model(thresh_features, targets)
            save_threshold_model(user_id, threshold_model, save_to_mongo=True, users_collection=users_collection)
        
        print(f"Models trained and saved successfully for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    else:
        print(f"Insufficient data to train models for {user_id}")
else:
    print(f"No retraining needed for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")