

#probably wont be uesed, will stick to optics ig


import numpy as np
import pandas as pd
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import joblib
import os

# ========= Config =========
AUTOENCODER_DIR = "models/autoencoders"
if not os.path.exists(AUTOENCODER_DIR):
    os.makedirs(AUTOENCODER_DIR)

# ========= Preprocessing =========
def preprocess_user_data_for_ae(user_id: str, collection: Collection):
    docs = list(collection.find({"user_id": user_id}))
    if not docs or len(docs) < 30:
        return None, None

    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    features = df[["latitude", "longitude", "hour", "weekday", "month"]].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler

# ========= Autoencoder Model =========
def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(10, activation='relu')(input_layer)
    encoded = Dense(5, activation='relu')(encoded)
    decoded = Dense(10, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

# ========= Training =========
def train_autoencoder(user_id: str, collection: Collection):
    data, scaler = preprocess_user_data_for_ae(user_id, collection)
    if data is None:
        print("Not enough data to train AE for user", user_id)
        return None

    autoencoder = create_autoencoder(data.shape[1])
    autoencoder.fit(data, data, epochs=50, batch_size=16, verbose=0)

    # Save model and scaler
    model_path = f"{AUTOENCODER_DIR}/{user_id}_ae.h5"
    scaler_path = f"{AUTOENCODER_DIR}/{user_id}_scaler.pkl"
    autoencoder.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    return model_path

# ========= Inference =========
def compute_anomaly_score_autoencoder(user_id: str, lat: float, lon: float, timestamp: datetime):
    model_path = f"{AUTOENCODER_DIR}/{user_id}_ae.h5"
    scaler_path = f"{AUTOENCODER_DIR}/{user_id}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Autoencoder or scaler not found for user", user_id)
        return 0.0

    from tensorflow.keras.models import load_model
    autoencoder = load_model(model_path)
    scaler = joblib.load(scaler_path)

    hour = timestamp.hour
    weekday = timestamp.weekday()
    month = timestamp.month
    sample = np.array([[lat, lon, hour, weekday, month]])
    sample_scaled = scaler.transform(sample)

    reconstructed = autoencoder.predict(sample_scaled, verbose=0)
    mse = np.mean((sample_scaled - reconstructed) ** 2)
    return float(mse)

# ========= Integration Example =========
# Usage inside anomalies.py (instead of rule-based):
# from .autoencoder_module import compute_anomaly_score_autoencoder
# score = compute_anomaly_score_autoencoder(user_id, lat, lon, datetime.now())
# if score > THRESHOLD:
#     trigger_alert()
# else:
#     continue
