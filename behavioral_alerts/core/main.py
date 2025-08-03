
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import httpx
from pymongo import MongoClient
from datetime import datetime
from .config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
users_collection = db["users"]

async def run_periodic_checks():
    try:
        # Query user_id and device_id pairs from users_collection
        user_devices = list(users_collection.find({}, {"user_id": 1, "device_id": 1, "_id": 0}))
        if not user_devices:
            print(f"[✗] No users found in database at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
            return
        async with httpx.AsyncClient() as client:
            for user in user_devices:
                user_id = user["user_id"]
                device_id = user.get("device_id")
                if not device_id:
                    print(f"[✗] No device_id for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
                    continue
                response = await client.get(f"http://localhost:8000/periodic-check/{user_id}/{device_id}")
                print(f"[✓] Periodic check for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {response.json()}")
    except Exception as e:
        print(f"[✗] Error in periodic checks at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

scheduler = AsyncIOScheduler()
scheduler.add_job(run_periodic_checks, "interval", minutes=5)
scheduler.start()
asyncio.get_event_loop().run_forever()