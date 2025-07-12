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


import logging
from datetime import datetime
from pymongo.collection import Collection

from .profiling import build_user_profile, should_retrain
from .threshold_adjustment import (
    prepare_threshold_data, train_threshold_model, save_threshold_model
)
from .incident_prediction import (
    prepare_incident_data, train_incident_model, save_incident_model
)

# Configure a simple logger
logger = logging.getLogger("user_retrainer")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def retrain_all_models_for_user(
    user_id: str,
    ts_collection: Collection,
    users_collection: Collection,
    last_trained: datetime = None
):
    """
    For a given user:
      1. Retrain behavior profile (OPTICS + histograms).
      2. Retrain dynamic threshold model.
      3. Retrain incident prediction model.
    """
    logger.info(f"􀋐 Starting retraining for user '{user_id}'")

    # --- 1) Behavior profile ---
    try:
        if should_retrain(ts_collection, user_id, last_trained):
            logger.info(f" Retraining behavior profile for '{user_id}'")
            build_user_profile(
                user_id, ts_collection,
                save_to_mongo=True
            )
        else:
            logger.info(f" Skipping profile: not due yet for '{user_id}'")
    except Exception as e:
        logger.error(f" Behavior profile retrain failed for {user_id}: {e}", exc_info=True)

    # --- 2) Threshold model ---
    try:
        feats, target = prepare_threshold_data(ts_collection, user_id)
        if feats is not None:
            tf_model = train_threshold_model(feats, target)
            save_threshold_model(
                user_id, tf_model,
                save_to_mongo=True,
                users_collection=users_collection,
                save_local=True
            )
            logger.info(f" Threshold model retrained for '{user_id}'")
        else:
            logger.warning(f" Insufficient data → skip threshold model for '{user_id}'")
    except Exception as e:
        logger.error(f" Threshold model retrain failed for {user_id}: {e}", exc_info=True)

    # --- 3) Incident model ---
    try:
        inc_feats, inc_target = prepare_incident_data(users_collection, user_id)
        if inc_feats is not None:
            inc_model, inc_scaler = train_incident_model(inc_feats, inc_target)
            save_incident_model(
                user_id, inc_model, inc_scaler,
                save_to_mongo=True,
                users_collection=users_collection,
                save_local=True
            )
            logger.info(f" Incident model retrained for '{user_id}'")
        else:
            logger.warning(f" Insufficient data → skip incident model for '{user_id}'")
    except Exception as e:
        logger.error(f" Incident model retrain failed for {user_id}: {e}", exc_info=True)

    logger.info(f"􀋐 Finished retraining for user '{user_id}'\n")


"""
from apscheduler.schedulers.background import BackgroundScheduler
from .train_models import retrain_all_models_for_user

scheduler = BackgroundScheduler()
scheduler.add_job(
    lambda: [
        retrain_all_models_for_user(uid, ts_collection, users_collection)
        for uid in users_collection.distinct("user_id")
    ],
    "interval",
    hours=1  # or whatever cadence you prefer
)
scheduler.start()

"""