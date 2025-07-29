
from pymongo import MongoClient, GEOSPHERE
from .config import MONGO_URI
from datetime import datetime, timezone
import uuid

def initialize_database():
    """Initialize the MongoDB database and collections."""
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]
        
        # Create collections if they don't exist
        db.create_collection("users", validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["user_id", "name", "email", "phone", "emergency_contact_phone", "created_at"],
                "properties": {
                    "user_id": {"bsonType": "string"},
                    "name": {"bsonType": "string"},
                    "email": {"bsonType": "string"},
                    "phone": {"bsonType": "string"},
                    "emergency_contact_phone": {"bsonType": "string"},
                    "created_at": {"bsonType": "date"},
                    "behavior_profile": {"bsonType": ["object", "null"]},
                    "incident_model": {"bsonType": ["binData", "null"]},
                    "incident_scaler": {"bsonType": ["binData", "null"]},
                    "model_last_updated": {"bsonType": ["date", "null"]}
                }
            }
        })
        db.create_collection("locations", validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["location_id", "user_id", "device_id", "location", "timestamp"],
                "properties": {
                    "location_id": {"bsonType": "string"},
                    "user_id": {"bsonType": "string"},
                    "device_id": {"bsonType": "string"},
                    "location": {
                        "bsonType": "object",
                        "required": ["type", "coordinates"],
                        "properties": {
                            "type": {"enum": ["Point"]},
                            "coordinates": {
                                "bsonType": "array",
                                "items": {"bsonType": "double"}
                            }
                        }
                    },
                    "location_type": {"bsonType": "string"},
                    "timestamp": {"bsonType": "date"},
                    "alert": {"bsonType": ["object", "null"]}
                }
            }
        })
        db.create_collection("devices", validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["device_id", "user_id", "device_type", "sim_id", "battery_level", "registered_at"],
                "properties": {
                    "device_id": {"bsonType": "string"},
                    "user_id": {"bsonType": "string"},
                    "device_type": {"bsonType": "string"},
                    "sim_id": {"bsonType": "string"},
                    "battery_level": {"bsonType": "int"},
                    "registered_at": {"bsonType": "date"}
                }
            }
        })
        
        # Create indexes
        db.users.create_index([("user_id", 1)], unique=True)
        db.users.create_index([("email", 1)], unique=True)
        db.locations.create_index([("location_id", 1)], unique=True)
        db.locations.create_index([("user_id", 1), ("timestamp", -1)])
        db.locations.create_index([("location", GEOSPHERE)])
        db.devices.create_index([("device_id", 1)], unique=True)
        db.devices.create_index([("user_id", 1)])
        
        print(f"[✓] Initialized database at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")
    except Exception as e:
        print(f"[✗] Error initializing database at {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}: {e}")

if __name__ == "__main__":
    initialize_database()
