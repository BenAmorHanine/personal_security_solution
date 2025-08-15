locations collection

(now serves both raw pings and alert events)

 Raw GPS ping:
{
  _id: ObjectId,
  user_id: String,
  device_id: String,
  location: {
    type: "Point",
    coordinates: [ longitude, latitude ]
  },
  timestamp: Date
}

Alert event:
{
  _id: ObjectId,
  user_id: String,
  device_id: String,
  location: {
    type: "Point",
    coordinates: [ longitude, latitude ]
  },
  timestamp: Date,
  alert: {
    alert_id: String,              // UUID primary key for this event
    incident_probability: Number,  // model score
    is_incident: Boolean,
    location_anomaly: Number,      // [0–1]
    hour_anomaly: Number,          // [0–1]
    weekday_anomaly: Number,       // [0–1]
    month_anomaly: Number          // [0–1]
  }
}