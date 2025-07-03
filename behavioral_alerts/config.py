# config.py



# Anomaly Detection Thresholds
DISTANCE_THRESHOLD = 0.05  # approx ~5km
PROB_THRESHOLD = 0.05
LATE_NIGHT_HOURS = list(range(22, 24)) + list(range(0, 5))  # 10PM–5AM

# Clustering Parameters
MIN_SAMPLES_CLUSTERING = 5
CLUSTERING_XI = 0.1

# Scheduler
PROFILE_UPDATE_INTERVAL_MINUTES = 30

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Logging (optional)
DEBUG_MODE = True
