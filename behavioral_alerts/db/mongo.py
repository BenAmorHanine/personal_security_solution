from pymongo import MongoClient
import os

from config import MONGO_URI

def get_db():
    client = MongoClient(MONGO_URI)
    return client

client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017/"))
db = client["hydatis"]
collection = db["user_locations"]
collection.create_index([("latitude", "2dsphere"), ("longitude", "2dsphere")])

def get_collection():
    return collection
