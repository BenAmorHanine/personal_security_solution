
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, OPTICS

import joblib
from pathlib import Path


#DBSCAN artifacts
dbscan       = joblib.load( "../models/riskyzones/risky_location_model_dbscan.pkl")
cw_dbscan    = joblib.load("../models/riskyzones/cluster_weights_dbscan.pkl")
cf_dbscan    = joblib.load("../models/riskyzones/cluster_fatalities_dbscan.pkl")
cet_dbscan   = joblib.load("../models/riskyzones/cluster_event_types_dbscan.pkl")

#OPTICS artifacts
optics       = joblib.load("../models/riskyzones/risky_location_model_optics.pkl")
cw_optics    = joblib.load("../models/riskyzones/cluster_weights_optics.pkl")
cf_optics    = joblib.load("../models/riskyzones/cluster_fatalities_optics.pkl")
cet_optics   = joblib.load("../models/riskyzones/cluster_event_types_optics.pkl")
#the coordinate scaler and
scaler = joblib.load("../models/riskyzones/scaler__risky_loc.pkl")
# Load X_scaled for max_distance calculation
X_scaled = joblib.load("../models/riskyzones/X_scaled.pkl")

def predict_risk_score_dbscan(lat, lon, model, scaler, cluster_weights, cluster_fatalities, cluster_event_types, X_scaled, beta=5.0):
    X_new = np.array([[lat, lon]], dtype=np.float32)
    X_new_scaled = scaler.transform(X_new)
    if len(model.components_) == 0:
        return (0.0, 0.0, 'None')
    distances = np.sqrt(((X_new_scaled - model.components_)**2).sum(axis=1))
    nearest_idx = np.argmin(distances)
    max_distance = np.sqrt(((X_scaled - X_scaled.mean(axis=0))**2).sum(axis=1)).max()
    normalized_distance = distances[nearest_idx] / max_distance
    nearest_cluster = model.labels_[model.core_sample_indices_[nearest_idx]]
    event_type = cluster_event_types.get(nearest_cluster, 'Unknown')
    if distances[nearest_idx] <= model.eps:
        score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
    else:
        base_score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        base_score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score = max(base_score * np.exp(-beta * (distances[nearest_idx] - model.eps)), 0.1 * base_score)
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
    return (score, normalized_distance, event_type)

def predict_risk_score_optics(lat, lon, model, scaler, cluster_weights, cluster_fatalities, cluster_event_types, X_scaled, beta=5.0):
    X_new = np.array([[lat, lon]], dtype=np.float32)
    X_new_scaled = scaler.transform(X_new)
    core_indices = np.where(model.core_distances_ != np.inf)[0]
    if len(core_indices) == 0:
        return (0.0, 0.0, 'None')
    core_points = X_scaled[core_indices]
    labels = model.labels_[core_indices]
    distances = np.sqrt(((X_new_scaled - core_points)**2).sum(axis=1))
    nearest_idx = np.argmin(distances)
    max_distance = np.sqrt(((X_scaled - X_scaled.mean(axis=0))**2).sum(axis=1)).max()
    normalized_distance = distances[nearest_idx] / max_distance
    nearest_cluster = labels[nearest_idx]
    event_type = cluster_event_types.get(nearest_cluster, 'Unknown')
    eps = np.percentile(model.core_distances_[core_indices], 95) if len(core_indices) > 0 else 0.05
    if distances[nearest_idx] <= eps:
        score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
    else:
        base_score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        base_score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score = max(base_score * np.exp(-beta * (distances[nearest_idx] - eps)), 0.1 * base_score)
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
    return (score, normalized_distance, event_type)

def predict_risk(lat, lon, method='dbscan'):
    if method == 'dbscan':
        return predict_risk_score_dbscan(
            lat, lon, dbscan, scaler,
            cw_dbscan, cf_dbscan,
            cet_dbscan, X_scaled
        )
    elif method == 'optics':
        return predict_risk_score_optics(
            lat, lon, optics, scaler,
            cw_optics, cf_optics,
            cet_optics, X_scaled
        )
    else:
        raise ValueError("Invalid method. Use 'dbscan' or 'optics'.")