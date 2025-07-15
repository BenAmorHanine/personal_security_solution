from typing import Dict, Any

DEFAULT_WEIGHTS = {
    "risk_location": 0.1,
    "unusual_time": 0.05,
    "abnormal_movement": 0.2,
    "audio_stress": 0.3,
    "keyword_alert": 0.3,
    "behavior_pattern": 0.05
}
DEFAULT_THRESHOLD = 0.4

def fuse_alerts(signals: Dict[str, float], weights: Dict[str, float] = DEFAULT_WEIGHTS, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Fuse model signals by computing a weighted sum, normalizing it, and comparing to a threshold.

    Args:
        signals: Dictionary of model names to scores (0 to 1).
        weights: Dictionary of model weights. Defaults to DEFAULT_WEIGHTS.
        threshold: Threshold for triggering an alert. Defaults to DEFAULT_THRESHOLD.

    Returns:
        Dictionary with weighted scores, total score, normalized score, and trigger decision.
    """
    weighted_scores = {}
    total = 0.0
    max_total = 0.0

    for name in weights:
        score = signals.get(name, 0.0)
        if not (0 <= score <= 1):
            raise ValueError(f"Signal {name} score {score} must be in [0,1]")
        weight = weights[name]
        weighted_score = score * weight
        weighted_scores[name] = weighted_score
        total += weighted_score
        max_total += weight

    normalized_score = total / max_total if max_total > 0 else 0.0
    trigger_alert = normalized_score >= threshold

    return {
        "weighted_scores": weighted_scores,
        "total_score": total,
        "normalized_score": normalized_score,
        "max_total": max_total,
        "trigger_alert": trigger_alert
    }




import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, filename="alerts.log")

def display_outputs(signals: Dict[str, float], weights: Dict[str, float], fusion_result: Dict) -> None:
    """
    Display model signals, weights, and fusion result.

    Args:
        signals: Dictionary of model names to scores.
        weights: Dictionary of model weights.
        fusion_result: Result from fuse_alerts (includes total_score, normalized_score, etc.).
    """
    for name in weights:
        score = signals.get(name, 0.0)
        weight = weights[name]
        weighted_score = fusion_result["weighted_scores"].get(name, 0.0)
        logging.info(f"Signal: {name}, Score: {score:.2f}, Weight: {weight:.2f}, Weighted: {weighted_score:.2f}")
        print(f"Signal: {name}, Score: {score:.2f}, Weight: {weight:.2f}, Weighted: {weighted_score:.2f}")
    logging.info(f"Total Score: {fusion_result['total_score']:.2f}")
    logging.info(f"Normalized Score: {fusion_result['normalized_score']:.2f}")
    logging.info(f"Threshold: {fusion_result.get('threshold', 0.4):.2f}")
    logging.info(f"Alert Triggered: {'Yes' if fusion_result['trigger_alert'] else 'No'}")
    print(f"Total Score: {fusion_result['total_score']:.2f}")
    print(f"Normalized Score: {fusion_result['normalized_score']:.2f}")
    print(f"Threshold: {fusion_result.get('threshold', 0.4):.2f}")
    print(f"Alert Triggered: {'Yes' if fusion_result['trigger_alert'] else 'No'}")





def trigger_alert(user_id: str) -> None:
    """
    Trigger an alert for the specified user.

    Args:
        user_id: User identifier for the alert.
    """
    print(f"ALERT TRIGGERED: Immediate action required for user {user_id}")
    # Add API call or notification logic here
    # e.g., requests.post("https://api.example.com/alert", data={"user_id": user_id})