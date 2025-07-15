# Step 5: Predict risk score
def predict_risk_score_dbscan(lat, lon, model, scaler, cluster_weights, cluster_fatalities, cluster_event_types, df, beta=beta):
    X_new = np.array([[lat, lon]], dtype=np.float32)
    X_new_scaled = scaler.transform(X_new)
    if len(model.components_) == 0:
        print("No core points in model. Increase eps/min_samples or add more data.")
        return (0.0, 0.0, 'None')
    distances = np.sqrt(((X_new_scaled - model.components_)**2).sum(axis=1))
    nearest_idx = np.argmin(distances)
    max_distance = np.sqrt(((X_scaled - X_scaled.mean(axis=0))**2).sum(axis=1)).max()
    #max_distance = np.max(np.sqrt(((X_scaled - model.components_)**2).sum(axis=1)))
    normalized_distance = distances[nearest_idx] / max_distance
    nearest_cluster = model.labels_[model.core_sample_indices_[nearest_idx]]
    event_type = cluster_event_types.get(nearest_cluster, 'Unknown')
    if distances[nearest_idx] <= model.eps:
        score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
        print(f"Assigned to cluster {nearest_cluster} (type: {event_type}) with score {score:.2f}, normalized distance {normalized_distance:.2f}")
    else:
        base_score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        base_score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        #score = base_score * np.exp(-beta * (distances[nearest_idx] - model.eps))
        score = max(base_score * np.exp(-beta * (distances[nearest_idx] - model.eps)), 0.1 * base_score)
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
        print(f"Noise point near cluster {nearest_cluster} (type: {event_type}, distance {distances[nearest_idx]:.2f} > eps {model.eps}), score {score:.2f}, normalized distance {normalized_distance:.2f}")
    return (score, normalized_distance, event_type)



def predict_risk_score_optics(lat, lon, model, scaler, cluster_weights, cluster_fatalities, cluster_event_types, X_scaled, beta=5.0):
    X_new = np.array([[lat, lon]], dtype=np.float32)
    X_new_scaled = scaler.transform(X_new)

    # Use core samples from OPTICS
    core_indices = np.where(model.core_distances_ != np.inf)[0]
    if len(core_indices) == 0:
        print("No core points in OPTICS model.")
        return (0.0, 0.0, 'None')  # Fixed syntax error here

    core_points = X_scaled[core_indices]
    labels = model.labels_[core_indices]

    # Compute distances to core points
    distances = np.sqrt(((X_new_scaled - core_points) ** 2).sum(axis=1))
    nearest_idx = np.argmin(distances)
    max_distance = np.sqrt(((X_scaled - X_scaled.mean(axis=0)) ** 2).sum(axis=1)).max()
    normalized_distance = distances[nearest_idx] / max_distance
    nearest_cluster = labels[nearest_idx]

    event_type = cluster_event_types.get(nearest_cluster, 'Unknown')

    # Estimate eps from reachability distances
    eps = np.percentile(model.core_distances_[core_indices], 95) if len(core_indices) > 0 else 0.05

    if distances[nearest_idx] <= eps:
        score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
        print(f"Assigned to OPTICS cluster {nearest_cluster} (type: {event_type}) with score {score:.2f}, normalized distance {normalized_distance:.2f}")
    else:
        base_score = cluster_weights.get(nearest_cluster, 0) / (cluster_weights.max() if not cluster_weights.empty else 1)
        base_score *= (1 + 0.1 * cluster_fatalities.get(nearest_cluster, 0))
        score = max(base_score * np.exp(-beta * (distances[nearest_idx] - eps)), 0.1 * base_score)
        score /= (cluster_weights / cluster_weights.max() * (1 + 0.1 * cluster_fatalities)).max()
        print(f"Noise point near OPTICS cluster {nearest_cluster} (type: {event_type}, distance {distances[nearest_idx]:.2f} > eps {eps}), score {score:.2f}, normalized distance {normalized_distance:.2f}")
    
    return (score, normalized_distance, event_type)

    
def predict_risk(lat, lon, method='dbscan'):
    if method == 'dbscan':
        return predict_risk_score_dbscan(
            lat, lon, dbscan, scaler,
            cluster_weights_dbscan, cluster_fatalities_dbscan,
            cluster_event_types, X_scaled
        )
    elif method == 'optics':
        return predict_risk_score_optics(
            lat, lon, model_optics, scaler,
            cluster_weights_optics, cluster_fatalities_optics,
            cluster_event_types_optics, X_scaled
        )
    else:
        raise ValueError("Invalid method. Use 'dbscan' or 'optics'.")