import pandas as pd

def preprocess_user_data(user_id, collection):
    from .profiling import build_user_profile, should_retrain

    df = pd.DataFrame(list(collection.find({"user_id": user_id})))
    if df.empty or len(df) < 10:
        return None
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0) / 3600
    
    if 'cluster' not in df.columns or should_retrain(user_id):
        df, _, _ = build_user_profile(df, save_model=True, user_id=user_id)
    return df