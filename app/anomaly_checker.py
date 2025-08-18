from datetime import datetime, timezone
from pymongo import MongoClient
from ..behavioral_alerts.core.config import MONGO_URI
#from ..behavioral_alerts.core.processing import process_capture
from ..fusion.process_capture import process_capture_all_inclusive
from ..behavioral_alerts.core.db_functions import register_device
import random

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]

def periodic_process_all_users():
    for user in users_collection.find({}, {"user_id": 1, "device_id": 1}):
        user_id = user["user_id"]
        device_id = user["devices"][0]["device_id"] if user.get("devices") else register_device(user_id, device_type="periodic", sim_id="sim-device", battery_level=100) #simulated for now too

        # Simulated location data for now
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)

        print(f"[i] Running the process_capture for {user_id} at {datetime.now(timezone.utc)}")
        process_capture_all_inclusive(user_id, device_id, latitude, longitude, sos_pressed=False)

"""if __name__ == "__main__":
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(periodic_process_all_users, 'interval', minutes=15)
    print(f"[i] Scheduler started at {datetime.now(timezone.utc)}")
    scheduler.start()
"""