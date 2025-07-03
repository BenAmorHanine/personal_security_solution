def trigger_alert(user_id, lat, lon, loc_anomaly, time_anomaly):
    print(f"[ALERT] {user_id} at ({lat}, {lon})")
    print(f"Location anomaly: {loc_anomaly:.2f}, Time anomaly: {time_anomaly:.2f}")


def trigger_alert(user_id, lat, lon, loc_anomaly, time_anomaly):
    print(f"[ALERT] {user_id} at ({lat}, {lon}) — Loc: {loc_anomaly:.2f}, Time: {time_anomaly:.2f}")

