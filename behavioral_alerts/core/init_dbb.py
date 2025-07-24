from pymongo import MongoClient
from config import MONGO_URI

def initialize_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client["safety_db_hydatis"]

        # Create users collection
        users = db["users"]
        users.create_index([("user_id", 1)], unique=True)
        users.create_index([("email", 1)], unique=True)
        print("[✓] Created users collection with indexes")
        #validation
        db.command({
            "collMod": "users",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "email", "emergency_contact_phone", "created_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "email": {"bsonType": "string"},
                        "emergency_contact_phone": {"bsonType": "string"},
                        "subscription_status": {"enum": ["free", "premium"]}
                    }
                }
            }
        })


        # Create locations collection (time-series)
        db.create_collection(
            "locations",
            timeseries={
                "timeField": "timestamp",
                "metaField": "user_id",
                "granularity": "minutes"
            }
        )
        db["locations"].create_index([("location", "2dsphere")])
        print("[✓] Created locations time-series collection with 2dsphere index")

        client.close()
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    initialize_db()



"""
Testing:
After running init_db.py, insert a test user:
python

from db_functions import create_user
user_id = create_user(
    name="John Doe",
    email="john@example.com",
    phone="+1234567890",
    emergency_contact_phone="+0987654321"
)
print(f"Created test user: {user_id}")




Verify collections in MongoDB shell:

use safety_db
show collections
db.users.findOne()
db.locations.getCollectionInfos()
"""