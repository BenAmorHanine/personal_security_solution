import os
import random
from datetime import datetime, timedelta, timezone

# Adjust your PYTHONPATH or package imports as needed
try:
    from .db_functions import create_user, register_device, update_location, log_alert
    from .capture import process_capture
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from db_functions import create_user, register_device, update_location, log_alert
    from capture import process_capture


def generate_test_locations(center_lat, center_lon, num_points=30, spread=0.01):
    """
    Generate a list of (lat, lon, timestamp) around a center point.
    """
    now = datetime.now(timezone.utc)
    points = []
    for i in range(num_points):
        lat = center_lat + random.uniform(-spread, spread)
        lon = center_lon + random.uniform(-spread, spread)
        ts = now - timedelta(minutes=(num_points - i) * 5)
        points.append((lat, lon, ts))
    return points


def generate_test_alerts(user_id, device_id, center_lat, center_lon, num_alerts=20, spread=0.02):
    """
    Create fake alerts with random anomaly scores and incident flags.
    """
    for i in range(num_alerts):
        lat = center_lat + random.uniform(-spread, spread)
        lon = center_lon + random.uniform(-spread, spread)
        ts = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 60))
        # random anomaly scores
        loc_anom = random.uniform(0, 1)
        hour_anom = random.uniform(0, 1)
        weekday_anom = random.uniform(0, 1)
        month_anom = random.uniform(0, 1)
        is_incident = random.choice([True, False])
        prob = random.uniform(0, 1)
        log_alert(
            user_id=user_id,
            device_id=device_id,
            latitude=lat,
            longitude=lon,
            timestamp=ts,
            incident_probability=prob,
            is_incident=is_incident,
            location_anomaly=loc_anom,
            hour_anomaly=hour_anom,
            weekday_anomaly=weekday_anom,
            month_anomaly=month_anom,
            save_locally=False
        )


def main():
    # 1) Create a fake user
    user_id = create_user(
        name="Test User",
        email="testuser@example.com",
        phone="+10000000000",
        emergency_contact_phone="+19999999999"
    )
    print(f"Created test user: {user_id}")

    # 2) Register a device
    device_id = register_device(
        user_id=user_id,
        device_type="test-device",
        sim_id="SIM123456",
        battery_level=100
    )
    print(f"Registered test device: {device_id}\n")

    # 3) Generate synthetic location history for profiling
    center_lat, center_lon = 37.7749, -122.4194  # Example: San Francisco
    locations = generate_test_locations(center_lat, center_lon, num_points=50)
    for lat, lon, ts in locations:
        update_location(user_id, device_id, lat, lon, timestamp=ts)

    # 4) Generate synthetic alerts for threshold/model testing
    generate_test_alerts(user_id, device_id, center_lat, center_lon, num_alerts=30)

    # 5) Test process_capture with sos_pressed=True and False
    test_points = [
        # A familiar point, near center
        (center_lat + 0.001, center_lon - 0.002),
        # Anomalous point, far away
        (center_lat + 0.1, center_lon + 0.1)
    ]

    print("\n--- Testing process_capture ---")
    for sos in [True, False]:
        for lat, lon in test_points:
            result = process_capture(
                user_id=user_id,
                device_id=device_id,
                latitude=lat,
                longitude=lon,
                sos_pressed=sos
            )
            print(f"sos_pressed={sos}, lat={lat:.4f}, lon={lon:.4f} -> result: {result}")


if __name__ == "__main__":
    main()
