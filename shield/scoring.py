"""Shield fraud scoring — feature extraction and model inference."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from shield.registry import ModelEntry, get_active_model, get_model

FEATURE_NAMES = [
    "amount_normalized", "amount_percentile_90d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "velocity_1h", "velocity_24h",
    "is_new_country", "is_new_device",
    "tx_type_encoded", "geo_distance_km",
    "time_since_last_tx_hours", "amount_velocity_ratio",
]


def _parse_ts(ts_str: str, fallback: datetime) -> datetime:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return fallback


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_features(payload: dict) -> np.ndarray:
    tx = payload.get("transaction", {})
    profile = payload.get("customerProfile", {})
    recent = payload.get("recentTransactions", [])
    geo_history = payload.get("geoHistory", [])

    amount = float(tx.get("amount", 0))
    avg_amount = float(profile.get("avg_transaction_amount", amount or 1))
    amount_normalized = min(amount / max(avg_amount, 1), 10.0)

    recent_amounts = [float(t.get("amount", 0)) for t in recent]
    amount_percentile_90d = (sum(1 for a in recent_amounts if a <= amount) / len(recent_amounts)) if recent_amounts else 0.5

    timestamp_str = tx.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        hour, dow = ts.hour, ts.weekday()
    except Exception:
        hour, dow = 12, 2

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)

    try:
        now = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except Exception:
        now = datetime.now(timezone.utc)

    one_h_ago = now - timedelta(hours=1)
    velocity_1h = sum(1 for t in recent if _parse_ts(t.get("timestamp", ""), now) >= one_h_ago)
    velocity_24h = len(recent)

    typical_countries = set(profile.get("typical_countries", ["CR"]))
    geo = tx.get("geolocation") or {}
    current_country = geo.get("country", "CR")
    is_new_country = 0.0 if current_country in typical_countries else 1.0

    known_devices = set(profile.get("known_devices", []))
    device_id = tx.get("device_id") or ""
    is_new_device = 0.0 if (not device_id or device_id in known_devices) else 1.0

    type_map = {"sinpe": 0, "card": 1, "atm": 2, "ecommerce": 3, "wallet": 4, "transfer": 5}
    tx_type_encoded = float(type_map.get(tx.get("transaction_type", "card"), 1))

    geo_distance_km = 0.0
    if geo_history and geo.get("lat") is not None:
        last_geo = geo_history[-1]
        geo_distance_km = min(
            _haversine(float(last_geo.get("lat", 0)), float(last_geo.get("lng", 0)), float(geo["lat"]), float(geo.get("lng", 0))),
            20000.0,
        )

    time_since_last_tx_hours = 168.0
    if recent:
        last_ts = _parse_ts(recent[0].get("timestamp", ""), now)
        time_since_last_tx_hours = min(max((now - last_ts).total_seconds() / 3600, 0), 168.0)

    total_24h = sum(float(t.get("amount", 0)) for t in recent) + amount
    amount_velocity_ratio = min(amount / max(total_24h, 1), 1.0)

    return np.array([
        amount_normalized, amount_percentile_90d,
        hour_sin, hour_cos, dow_sin, dow_cos,
        float(velocity_1h), float(velocity_24h),
        is_new_country, is_new_device,
        tx_type_encoded, geo_distance_km,
        time_since_last_tx_hours, amount_velocity_ratio,
    ], dtype=np.float64)


def _behavioral_and_importances(features: np.ndarray, ml_score: float) -> dict[str, Any]:
    behavioral_inputs = {
        "amount_normalized": min(features[0] / 5.0, 1.0) * 0.25,
        "is_new_country": features[8] * 0.20,
        "is_new_device": features[9] * 0.20,
        "velocity_24h": min(features[7] / 15.0, 1.0) * 0.15,
        "geo_distance_km": min(features[11] / 3000.0, 1.0) * 0.10,
        "amount_velocity_ratio": features[13] * 0.10,
    }
    behavioral_score = float(sum(behavioral_inputs.values()))
    combined_score = float(0.6 * ml_score + 0.4 * behavioral_score)

    mean = np.array([2.0, 0.5, 0, 1, 0, 1, 1.5, 5.0, 0.1, 0.1, 1.5, 500.0, 12.0, 0.2])
    std = np.array([1.5, 0.3, 1, 1, 1, 1, 2.0, 5.0, 0.3, 0.3, 1.5, 1000.0, 24.0, 0.3])
    z_scores = np.abs((features - mean) / np.maximum(std, 0.01))
    total_z = float(z_scores.sum()) or 1.0
    importances = {FEATURE_NAMES[i]: round(float(z_scores[i] / total_z), 4) for i in range(len(FEATURE_NAMES))}

    return {
        "isolation_score": round(ml_score * 100, 2),
        "behavioral_score": round(behavioral_score * 100, 2),
        "combined_score": round(combined_score * 100, 2),
        "feature_importances": importances,
    }


def score_with_entry(features: np.ndarray, entry: ModelEntry) -> dict[str, Any]:
    t0 = time.time()
    features_2d = features.reshape(1, -1)
    model = entry.model

    if entry.name == "iforest" or (hasattr(model, "score_samples") and not hasattr(model, "predict_proba")):
        raw_score = float(model.score_samples(features_2d)[0])
        normalized = (raw_score - (-0.3)) / (-0.7 - (-0.3))
        ml_score = float(max(0.0, min(1.0, normalized)))
    elif hasattr(model, "predict_proba"):
        ml_score = float(model.predict_proba(features_2d)[:, 1][0])
    else:
        ml_score = float(model.predict(features_2d)[0])

    result = _behavioral_and_importances(features, ml_score)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    result["model_version"] = f"{entry.name}_{entry.version}"
    return result


def score_transaction(payload: dict) -> dict[str, Any]:
    """Score a ScoringContext payload. Returns explicit model_available + model_source."""
    t0 = time.time()
    tx = payload.get("transaction", {})
    transaction_id = tx.get("transaction_id", "unknown")
    features = extract_features(payload)

    use_ensemble = os.environ.get("ML_INTERNAL_ENSEMBLE", "").lower() in ("1", "true", "yes")
    entry = get_active_model()

    if entry is not None and not use_ensemble:
        result = score_with_entry(features, entry)
        return {
            "transaction_id": transaction_id,
            "model_available": True,
            "model_source": "active",
            "model_name": result["model_version"],
            **{k: v for k, v in result.items() if k != "model_version"},
            "model_version": result["model_version"],
            "total_latency_ms": round((time.time() - t0) * 1000, 1),
        }

    if use_ensemble:
        members = []
        for name in ("logistic_regression", "xgboost", "lightgbm"):
            m = get_model(name)
            if m is not None:
                members.append(m)
        if members:
            features_2d = features.reshape(1, -1)
            proba_sum, n = 0.0, 0
            for m in members:
                try:
                    proba_sum += float(m.model.predict_proba(features_2d)[:, 1][0])
                    n += 1
                except Exception:
                    pass
            if n > 0:
                result = _behavioral_and_importances(features, proba_sum / n)
                return {
                    "transaction_id": transaction_id,
                    "model_available": True,
                    "model_source": "active",
                    "model_name": "ensemble_v2",
                    **result,
                    "model_version": "ensemble_v2",
                    "total_latency_ms": round((time.time() - t0) * 1000, 1),
                }

    return {
        "transaction_id": transaction_id,
        "model_available": False,
        "model_source": "unavailable",
        "model_name": "none",
        "model_version": "none",
        "isolation_score": 0.0,
        "behavioral_score": 0.0,
        "combined_score": 0.0,
        "feature_importances": {},
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "total_latency_ms": round((time.time() - t0) * 1000, 1),
    }
