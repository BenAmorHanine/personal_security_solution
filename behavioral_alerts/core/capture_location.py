
import random
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import time
import argparse
from behavioral_alerts.core.db_functions import update_location
from behavioral_alerts.core.config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]

def capture_location_periodically(user_id, device_id, duration_minutes=60, interval_seconds=600):
    """Simulate periodic location capture for a user over a specified duration."""
    try:
        base_lat, base_lon = 48.8566, 2.3522  # Paris coordinates
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now(timezone.utc) < end_time:
            lat = base_lat + random.uniform(-0.01, 0.01)
            lon = base_lon + random.uniform(-0.01, 0.01)
            timestamp = datetime.now(timezone.utc)
            location_id = update_location(user_id, device_id, lat, lon, timestamp=timestamp)
            if location_id is None:
                print(f"[✗] Failed to capture location for user {user_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S CET')}")
            else:
                print(f"[✓] Captured location {location_id} for user {user_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S CET')}")
            time.sleep(interval_seconds)
    except Exception as e:
        print(f"[✗] Error in periodic location capture for user {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate periodic location capture for a user.")
    parser.add_argument("--user_id", required=True, help="User ID for location capture")
    parser.add_argument("--device_id", required=True, help="Device ID for location capture")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--interval", type=int, default=600, help="Interval in seconds")
    args = parser.parse_args()
    
    capture_location_periodically(args.user_id, args.device_id, args.duration, args.interval)
