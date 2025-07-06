from pydantic import BaseModel
from datetime import datetime

#OPTIONALLL

class CaptureRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    emergency: bool = False




class UserProfile(BaseModel):
    user_id: str
    centroids: list[dict]  # List of dictionaries with 'latitude' and 'longitude'
    hour_freq: dict  # Hourly frequency distribution
    weekday_freq: dict  # Weekday frequency distribution
    month_freq: dict  # Monthly frequency distribution
    scaler: dict  # Scaler parameters for location data (if needed)
    clustering_method: str  # 'dbscan' or 'optics'


{
  "user_id": "string",
  "location_data": [
    {
      "latitude": float,
      "longitude": float,
      "timestamp": datetime,
      "hour": int,
      "weekday": int,
      "month": int
    }
  ]
}


#idk what all of these are