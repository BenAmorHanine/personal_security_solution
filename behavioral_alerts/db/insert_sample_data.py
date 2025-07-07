from behavioral_alerts.core.utils import setup_timeseries_collection, setup_geospatial_collection, setup_users_collection, insert_location, insert_geo_data, insert_user_alert
from datetime import datetime

ts_collection = setup_timeseries_collection()
geo_collection = setup_geospatial_collection()
users_collection = setup_users_collection()
for i in range(20):
    insert_location(ts_collection, "user1", 40.7128 + i*0.0001, -74.0060, datetime.now())
    insert_geo_data(geo_collection, "user1", 40.7128 + i*0.0001, -74.0060)
    insert_user_alert(users_collection, "user1", 1.0 if i % 2 == 0 else 0.0, 0.8, i % 2 == 0)
print("Sample data inserted successfully at", datetime.now().strftime("%Y-%m-%d %H:%M:%S CET"))