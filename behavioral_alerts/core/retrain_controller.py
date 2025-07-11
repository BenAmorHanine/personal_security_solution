"""
from behavioral_alerts.core.profiling import should_retrain, build_user_profile
from behavioral_alerts.core.threshold_adjustment import prepare_threshold_data, train_threshold_model, save_threshold_model, train_threshold_model, save_threshold_model

def retrain_user_profile(user_id, ts_collection, users_collection, last_trained=None):
    #from behavioral_alerts.core.profiling import should_retrain, build_user_profile
    
    if should_retrain(ts_collection, user_id, last_trained):
        print(f"✅ Retraining behavioral profile for {user_id}")
        build_user_profile(user_id, ts_collection, save_to_mongo=True)
"""


from datetime import datetime, timedelta
from behavioral_alerts.core.profiling import should_retrain, build_user_profile
from behavioral_alerts.core.threshold_adjustment import (
    prepare_threshold_data,
    train_threshold_model,
    save_threshold_model
)

def retrain_user_profile(user_id, ts_collection, users_collection, last_trained=None):
    if should_retrain(ts_collection, user_id, last_trained):
        print(f"✅ Retraining behavioral profile for {user_id}")
        build_user_profile(user_id, ts_collection, save_to_mongo=True)

        # Now retrain threshold model as well
        features, targets = prepare_threshold_data(ts_collection, user_id)
        if features is not None and targets is not None:
            model = train_threshold_model(features, targets)
            save_threshold_model(user_id, model, save_to_mongo=True, users_collection=users_collection)
            print(f"✅ Threshold model updated for {user_id}")
        else:
            print(f"⚠️ Skipped threshold model retrain — insufficient data for {user_id}")
    else:
        print(f"ℹ️ No retrain needed for {user_id}")

