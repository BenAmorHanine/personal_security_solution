# Personal Security Solution

A real-time, multi-signal personal safety system that detects potential security incidents by fusing behavioral anomaly detection, geospatial risk assessment, and vocal analysis.

## Overview

The system continuously monitors a user's location and context. When a capture event occurs (periodic check or manual SOS), three independent analysis pipelines evaluate the situation and their results are fused into a single incident decision.

```
                        ┌─────────────────────────────┐
  Location + Time  ───► │  Behavioral Anomaly Detection│
                        └──────────────┬──────────────┘
                                       │
  GPS Coordinates  ───► ┌──────────────▼──────────────┐   ┌────────────────┐
                        │   Geospatial Risk Assessment │──►│  Fusion Engine │──► Incident Alert
  Audio (optional) ───► ├──────────────┬──────────────┤   └────────────────┘
                        │  Vocal Analysis Pipeline     │
                        └─────────────────────────────┘
```

## Features

- **Behavioral Anomaly Detection** — Builds a per-user movement profile using OPTICS clustering. Flags unusual locations, hours, weekdays, and months relative to the user's historical patterns.
- **Incident Classification** — Trains a per-user Random Forest classifier on labeled incident history to predict incident probability and optimize the detection threshold via F1-based cross-validation.
- **Geospatial Risk Assessment** — Scores GPS coordinates against pre-trained DBSCAN and OPTICS models built from conflict/event datasets. Returns a risk score and predicted event type for any location.
- **Vocal Analysis** — Transcribes audio (Arabic) with OpenAI Whisper, classifies the transcription with a fine-tuned TunBERT model (`normal / danger / hate`), and extracts audio features (stress via Wav2Vec2, speech rhythm, and pitch/energy-based tone).
- **Fusion Engine** — Combines all signals. An incident is triggered if any of the following are true:
  - User pressed SOS manually
  - Behavioral anomaly probability exceeds the user's optimized threshold *and* location anomaly score > 0.6
  - Geospatial risk score > 0.7
  - Vocal analysis flags danger (dangerous text, angry/fearful tone, fast rhythm)
- **Flask API** — Exposes REST endpoints for alert retrieval and SOS triggering, with a background scheduler running periodic checks for all users.

## Architecture

```
personal_security_solution/
├── app/
│   ├── anomaly_checker.py        # Periodic processing job (runs all users)
│   └── scheduler_app_flask.py    # Flask app + APScheduler + REST API
│
├── behavioral_alerts/
│   ├── core/
│   │   ├── config.py             # DB URIs, thresholds, clustering parameters
│   │   ├── db_functions.py       # MongoDB helpers (users, devices, locations, alerts)
│   │   ├── incident_classifier.py# Per-user Random Forest classifier + threshold optimizer
│   │   ├── incident_prediction.py
│   │   ├── profiling.py          # OPTICS-based user movement profiling & anomaly scoring
│   │   └── processing.py        # Orchestrates anomaly detection for a single capture event
│   └── schemas/
│       ├── location_schema.py
│       └── users_schema.py       # MongoDB document schema reference
│
├── fusion/
│   ├── process_capture.py        # Main fusion function combining all three signals
│   ├── vocal_analysis.py         # Whisper + TunBERT + SpeechBrain + Pyannote pipeline
│   ├── risky_location_inference.py # Geospatial risk scoring (DBSCAN & OPTICS)
│   ├── call_models.py
│   └── api.py                    # FastAPI wrapper
│
├── models/
│   └── riskyzones/               # Pre-trained geospatial risk models (.pkl)
│
├── notebooks/                    # Exploratory notebooks for model development
├── Report/                       # LaTeX project report
└── papers___docs/                # Reference papers and documentation
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Flask / FastAPI |
| Database | MongoDB |
| Scheduling | APScheduler |
| ML / Clustering | scikit-learn (OPTICS, DBSCAN, RandomForest) |
| Audio Transcription | OpenAI Whisper |
| Text Classification | HuggingFace Transformers (TunBERT) |
| Emotion Recognition | SpeechBrain (Wav2Vec2 / IEMOCAP) |
| Speaker Diarization | Pyannote Audio |
| Geospatial | geopy |
| Data | pandas, numpy |

## Prerequisites

- Python 3.9+
- MongoDB running locally on `mongodb://localhost:27017`
- (Optional) CUDA-capable GPU for faster audio model inference

## Installation

```bash
# Clone the repository
git clone https://github.com/BenAmorHanine/personal_security_solution.git
cd personal_security_solution

# Install dependencies
pip install -r behavioral_alerts/requirements.txt
```

Additional packages needed for the full fusion pipeline (vocal + geospatial):

```bash
pip install torch torchaudio transformers whisper librosa speechbrain pyannote.audio geopy fastapi uvicorn
```

## Configuration

All tunable parameters live in `behavioral_alerts/core/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DB_NAME` | `safety_db_hydatis` | Database name |
| `DEFAULT_PROB_THRESHOLD` | `0.2` | Minimum incident probability threshold |
| `DISTANCE_THRESHOLD` | `0.5` | Location anomaly distance threshold (~5 km) |
| `CLUSTERING_METHOD` | `optics` | Clustering algorithm (`optics` or `dbscan`) |
| `PROFILE_UPDATE_INTERVAL_MINUTES` | `30` | How often user profiles are refreshed |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | FastAPI server settings |

## Running the Application

**Flask API (scheduler + REST endpoints)**

```bash
cd app
python scheduler_app_flask.py
```

The server starts on port `5000` and schedules a periodic check every 5 minutes.

**FastAPI server**

```bash
cd fusion
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/alerts` | Retrieve all logged alerts |
| `GET` | `/alerts/recent` | Retrieve alerts from the last 5 minutes |
| `POST` | `/sos/<user_id>/<device_id>/<lat>/<long>` | Trigger a manual SOS event |
| `POST` | `/process_capture` | Run a full capture analysis (FastAPI) |

## Data Model

**users** collection — stores user identity, registered devices, movement profile, and per-user ML model artifacts.

**locations** collection — stores every location update and the associated alert payload (anomaly scores, incident probability, vocal analysis summary, geospatial risk score).

See `behavioral_alerts/schemas/users_schema.py` for the full schema reference.

## How the Anomaly Detection Works

1. Every 30 minutes the scheduler rebuilds the user's movement profile from the last 35 days of location data using OPTICS clustering.
2. At each capture event, the current location is compared geodesically against the cluster centroids. A `location_anomaly` score between 0 (familiar) and 1 (completely new) is computed.
3. `hour_anomaly`, `weekday_anomaly`, and `month_anomaly` are derived from the inverse of the observed frequency of the current time dimension.
4. These four features are fed into the per-user Random Forest classifier, which returns an incident probability.
5. The classifier's optimal decision threshold is found via 5-fold cross-validation maximizing F1 score (minimum 0.2).

## License

This project was developed as part of an academic research project. See the `Report/` directory for the full technical report.
