import requests

url = "http://localhost:8000/anomaly_score"
payload = {
       "user_id": "user1",
       "lat": 40.7200,
       "lon": -74.0100,
       "hour": 2,
       "weekday": 5,
       "month": 7
       }
response = requests.post(url, json=payload)
print(response.json())