def process_capture(user_id: str, lat: float, lon: float,
                    ts_collection, geo_collection, users_collection):
    now = capture_and_store(user_id, lat, lon, ts_collection, geo_collection, users_collection)

    # === 1. Predict Unusual Time ===
    threshold_model = load_threshold_model(user_id)
    df = pd.DataFrame(list(ts_collection.find({"user_id": user_id})))
    features = [df["hour"].std() if not df.empty else 0, 0, len(df)]
    prob_threshold = predict_threshold(threshold_model, features) if threshold_model else 0.05

    loc_anomaly, time_anomaly = detect_user_anomalies(
        lat, lon, now.hour, now.weekday(), now.month, user_id, ts_collection, prob_threshold
    )

    # === 2. Predict Incident / Behavior Pattern (XGBoost) ===
    incident_model, scaler = load_incident_model(user_id)
    incident_prob = predict_incident(incident_model, scaler, loc_anomaly, time_anomaly) if incident_model else 0.0

    # === 3. Risky Location Score ===
    risk_location_score = predict_crime_risk(lat, lon)  # You defined this model

    # === 4. Movement Anomaly ===
    movement_score = detect_movement_anomaly(user_id, ts_collection, lat, lon, now)  # your GPS-based model

    # === 5. Audio Stress Model ===
    audio_stress_score = predict_audio_stress(user_id)  # emotion-from-voice score

    # === 6. Emergency Keyword from Speech ===
    keyword_score = detect_emergency_keyword(user_id)  # e.g. 1.0 if 'help', else 0.0

    # === Fuse All Model Outputs ===
    from behavioral_alerts.core.fusion import fuse_alerts

    signals = {
        "risk_location":     risk_location_score,
        "unusual_time":      time_anomaly,
        "abnormal_movement": movement_score,
        "audio_stress":      audio_stress_score,
        "keyword_alert":     keyword_score,
        "behavior_pattern":  incident_prob
    }

    fusion_result = fuse_alerts(signals)

    # === Optional: Store in alerts collection
    insert_user_alert(
        users_collection, user_id,
        loc_anomaly, time_anomaly,
        incident_prob, fusion_result["trigger_alert"]
    )

    # === Optional: Trigger physical alert or push notif
    if fusion_result["trigger_alert"]:
        trigger_alert_api(user_id)

    return {
        "signals": signals,
        "fusion": fusion_result
    }
