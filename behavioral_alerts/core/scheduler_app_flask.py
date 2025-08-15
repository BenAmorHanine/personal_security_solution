from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timezone, timedelta
import os
import sys
import json
from pymongo import MongoClient
from config import MONGO_URI, LOG_DIR
from anomaly_checker import periodic_process_all_users

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Flask & Scheduler setup
app = Flask(__name__)
scheduler = BackgroundScheduler()
scheduler.start()

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["safety_db_hydatis"]
locations_collection = db["locations"]

# -----------------------
# ALERT ROUTES
# -----------------------
@app.route('/alerts', methods=['GET'])
def all_alerts():
    alerts = []
    # Try reading from log files
    if os.path.exists(LOG_DIR):
        for filename in os.listdir(LOG_DIR):
            if filename.startswith("alert_") and filename.endswith(".json"):
                filepath = os.path.join(LOG_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        alert = json.load(f)
                        alerts.append(alert)
                except Exception as e:
                    print(f"[✗] Failed to read {filename}: {e}")
    # Fallback to DB if no logs found
    if not alerts:
        try:
            db_alerts = list(locations_collection.find({"alert": {"$exists": True}}))
            for a in db_alerts:
                alert_doc = {
                    "user_id": a["user_id"],
                    "device_id": a["device_id"],
                    "location": a["location"],
                    "timestamp": a["timestamp"].isoformat(),
                    "alert": a["alert"]
                }
                alerts.append(alert_doc)
            print(f"[i] Fetched {len(alerts)} alerts from DB as fallback.")
        except Exception as e:
            print(f"[✗] Failed fetching alerts from DB: {e}")
            return jsonify({"error": "Could not fetch alerts from logs or DB"}), 500
    alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(alerts)


@app.route('/alerts/recent', methods=['GET'])
def recent_alerts():
    """Return alerts from the last 5 minutes."""
    alerts = []
    now = datetime.now(timezone.utc)
    five_minutes_ago = now - timedelta(minutes=5)

    # Try reading from logs
    if os.path.exists(LOG_DIR):
        for filename in os.listdir(LOG_DIR):
            if filename.startswith("alert_") and filename.endswith(".json"):
                filepath = os.path.join(LOG_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        alert = json.load(f)
                        ts = datetime.fromisoformat(alert["timestamp"])
                        if ts >= five_minutes_ago:
                            alerts.append(alert)
                except Exception as e:
                    print(f"[✗] Failed to read {filename}: {e}")
                    continue

    # Fallback to DB
    if not alerts:
        try:
            db_alerts = list(locations_collection.find({
                "alert": {"$exists": True},
                "timestamp": {"$gte": five_minutes_ago}
            }))
            for a in db_alerts:
                alert_doc = {
                    "user_id": a["user_id"],
                    "device_id": a["device_id"],
                    "location": a["location"],
                    "timestamp": a["timestamp"].isoformat(),
                    "alert": a["alert"]
                }
                alerts.append(alert_doc)
            print(f"[i] Fetched {len(alerts)} recent alerts from DB as fallback.")
        except Exception as e:
            print(f"[✗] Failed fetching recent alerts from DB: {e}")
            return jsonify({"error": "Could not fetch recent alerts"}), 500

    alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(alerts)


# -----------------------
# HOME ROUTE
# -----------------------
@app.route('/')
def home():
    return "🌐 APScheduler Flask API is running."


# -----------------------
# MAIN STARTUP
# -----------------------
if __name__ == "__main__":
    # Schedule the periodic process automatically
    scheduler.add_job(periodic_process_all_users, 'interval', minutes=5, id='check_users')
    print(f"[i] Scheduler started at {datetime.now(timezone.utc)}")

    # Start Flask app
    app.run(port=5000)
