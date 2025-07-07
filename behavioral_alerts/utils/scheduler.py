from apscheduler.schedulers.background import BackgroundScheduler
from db.mongo import get_collection
from behavioral_alerts.core.profiling import build_profile
from datetime import datetime, timedelta
import pandas as pd

def update_profiles():
    coll = get_collection()
    for user_id in coll.distinct("user_id"):
        build_profile(user_id)
        coll.delete_many({
            "timestamp": {"$lt": datetime.now() - timedelta(days=30)},
            "user_id": user_id
        })
    print(f"[Scheduler] Profiles updated at {datetime.now()}")

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_profiles, "interval", minutes=30)
    scheduler.start()



from apscheduler.schedulers.background import BackgroundScheduler
from core.profiling import build_user_profile
from db.mongo import get_db
from behavioral_alerts.core.config import MONGO_DB_NAME, MONGO_COLLECTION_NAME
from datetime import datetime, timedelta
import pandas as pd

def periodic_update():
    client = get_db()
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    for user_id in collection.distinct("user_id"):
        build_user_profile(user_id, collection)
        threshold = datetime.now() - timedelta(days=30)
        collection.delete_many({"user_id": user_id, "timestamp": {"$lt": threshold}})

    print(f"[INFO] Profiles updated at {datetime.now()}")

scheduler = BackgroundScheduler()
scheduler.add_job(periodic_update, "interval", minutes=30)
scheduler.start()
