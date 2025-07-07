from src.profiling import build_user_profile, save_profile
from src.incident_prediction import prepare_incident_data, train_incident_model, save_incident_model
from src.threshold_adjustment import prepare_threshold_data, train_threshold_model, save_threshold_model
from src.data_utils import setup_timeseries_collection, setup_users_collection
from sklearn.cluster import OPTICS

ts_collection = setup_timeseries_collection()
users_collection = setup_users_collection()
centroids, hour_freq, weekday_freq, month_freq, scaler = build_user_profile("user1", ts_collection)
save_profile("user1", OPTICS(min_samples=5, xi=0.1), scaler)
features, labels = prepare_incident_data(users_collection, "user1")
if features is not None:
    model, scaler = train_incident_model(features, labels)
    save_incident_model("user1", model, scaler)
thresh_features, targets = prepare_threshold_data(ts_collection, "user1")
if thresh_features is not None:
    model = train_threshold_model(thresh_features, targets)
    save_threshold_model("user1", model)