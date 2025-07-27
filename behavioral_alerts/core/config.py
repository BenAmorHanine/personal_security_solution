# config.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.path.join(BASE_DIR, "models")

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "safety_db_hydatis"

# Anomaly Detection Thresholds
DISTANCE_THRESHOLD =0.5 # approx ~5km
DEFAULT_PROB_THRESHOLD = 0.05
LATE_NIGHT_HOURS = list(range(22, 24)) + list(range(0, 4))  # 10PM–5AM

# Clustering Parameters
MIN_SAMPLES_CLUSTERING = 5
CLUSTERING_XI = 0.1
CLUSTERING_METHOD = "optics"  # Options: "dbscan", "optics" # a la base kona hatin dbscan ythhorli houwa ata ahsn perf
# Scheduler
PROFILE_UPDATE_INTERVAL_MINUTES = 30

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Logging (optional)
DEBUG_MODE = True


# Twilio Configuration
#to be updated later
TWILIO_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_PHONE = "+1234567890"