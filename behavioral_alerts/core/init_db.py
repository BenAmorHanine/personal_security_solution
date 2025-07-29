"""from .utils import setup_timeseries_collection, setup_geospatial_collection, setup_users_collection
from datetime import datetime
from pymongo import MongoClient




ts_collection = setup_timeseries_collection()
geo_collection = setup_geospatial_collection()
users_collection = setup_users_collection()
print("Collections initialized successfully at", datetime.now().strftime("%Y-%m-%d %H:%M:%S CET"))"""

from .utils import setup_timeseries_collection, setup_geospatial_collection, setup_users_collection, setup_devices_collection
from datetime import datetime, timezone
from pymongo import MongoClient
from .config import MONGO_URI
import uuid
from .utils import setup_timeseries_collection, setup_geospatial_collection, setup_users_collection
from datetime import datetime, timezone
from pymongo import MongoClient
from .config import MONGO_URI
import uuid

def initialize_database():
    """Initialize the safety_db_hydatis database with required collections and sample data."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]
        
        # Initialize collections
        ts_collection = setup_timeseries_collection()
        geo_collection = setup_geospatial_collection()
        users_collection = setup_users_collection()
        
        # Insert sample user
        sample_user_id = str(uuid.uuid4())
        users_collection.insert_one({
            "user_id": sample_user_id,
            "name": "Sample User",
            "email": "sample@example.com",
            "phone": "+33612345678",
            "emergency_contact_phone": "+33687654321",
            "emergency_contacts": [
                {"name": "Contact 1", "phone": "+33612345679"},
                {"name": "Contact 2", "phone": "+33612345680"}
            ],
            "behavior_profile": {
                "centroids": [],
                "hour_freq": {},
                "weekday_freq": {},
                "month_freq": {},
                "last_updated": datetime.now(timezone.utc)
            },
            "created_at": datetime.now(timezone.utc)
        })
        print(f"[✓] Inserted sample user {sample_user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        # Insert sample police station
        geo_collection.insert_one({
            "location_id": "police_station_1",
            "user_id": None,
            "device_id": None,
            "location": {"type": "Point", "coordinates": [2.3522, 48.8566]},
            "timestamp": datetime.now(timezone.utc),
            "location_type": "police_station"
        })
        print(f"[✓] Inserted sample police station at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
        
        client.close()
        print(f"[✓] Collections initialized successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error initializing database at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

if __name__ == "__main__":
    initialize_database()