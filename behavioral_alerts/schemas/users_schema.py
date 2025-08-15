users collection
{
  _id: ObjectId,
  user_id: String,                         // UUID primary key
  name: String,
  email: String,
  phone: String,
  emergency_contact_phone: String,
  created_at: Date,

  devices: [
    {
      device_id: String,                   // UUID
      device_type: String,
      sim_id: String,
      battery_level: Number,
      registered_at: Date
    },
    …
  ],

  profile: {                                // from profiling.py
    last_updated?: Date,
    centroids: [
      {
        cluster_id: Number,
        center: [Number, Number],           // scaled [lat, lon]
        size: Number,
        hour_mean: Number,
        weekday_mean: Number,
        month_mean: Number
      },
      …
    ],
    hour_freq:   { "<hour>": Number, … },      // e.g. "0":0.12 … "23":0.01
    weekday_freq:{ "<weekday>": Number, … },   // "0":0.14 … "6":0.10
    month_freq:  { "<month>": Number, … }      // "1":0.20 … "12":0.05
  },

  // classifier model artifacts (all Base64 strings):
  classifier_model?:   String,              // base64-encoded pickle
  classifier_scaler?:  String,
  classifier_value?:   Number,
  classifier_updated_at?: Date,

  // Incident model artifacts (all Base64 strings):
  incident_model?:    String,              // base64-encoded pickle
  incident_scaler?:   String,
  optimal_threshold?: Number
}
