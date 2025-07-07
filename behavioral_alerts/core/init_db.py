from .data_utils import setup_timeseries_collection, setup_geospatial_collection, setup_users_collection
from datetime import datetime
from pymongo import MongoClient




ts_collection = setup_timeseries_collection()
geo_collection = setup_geospatial_collection()
users_collection = setup_users_collection()
print("Collections initialized successfully at", datetime.now().strftime("%Y-%m-%d %H:%M:%S CET"))