"""import requests

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
print(response.json())"""

from behavioral_alerts.core.threshold_adjustment import *
features, target = prepare_threshold_data(data_collection, "user1")
model = train_threshold_model(features, target)
prediction = predict_threshold(model, features[0])
print("Threshold:", prediction)


from behavioral_alerts.core.incident_prediction import *
features, labels = prepare_incident_data(data_collection, "user1")
model, scaler = train_incident_model(features, labels)
probability = predict_incident(model, scaler, 0.8, 0.7)
print("Incident Probability:", probability)
