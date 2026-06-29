"""Load labeled Shield training data from Postgres (ml_feedback + transactions)."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from shield.scoring import extract_features


def _get_connection():
    import psycopg2

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(url)


def count_labeled_feedback(since_iso: str | None = None) -> int:
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if since_iso:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM ml_feedback
                    WHERE label IN ('fraud', 'legitimate') AND created_at > %s
                    """,
                    (since_iso,),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM ml_feedback WHERE label IN ('fraud', 'legitimate')"
                )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def load_pg_feedback(min_rows: int = 50, tenant_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """JOIN ml_feedback with transactions; label fraud=1, legitimate=0."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            params: list[Any] = []
            tenant_filter = ""
            if tenant_id:
                tenant_filter = " AND mf.tenant_id = %s"
                params.append(tenant_id)
            cur.execute(
                f"""
                SELECT mf.label, t.transaction_id, t.transaction_type, t.amount, t.currency,
                       t.customer_id, t.device_id, t.ip_address, t.geolocation, t.timestamp,
                       t.tenant_id
                FROM ml_feedback mf
                JOIN transactions t
                  ON t.transaction_id = mf.transaction_id AND t.tenant_id = mf.tenant_id
                WHERE mf.label IN ('fraud', 'legitimate'){tenant_filter}
                ORDER BY mf.created_at DESC
                LIMIT 50000
                """,
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) < min_rows:
        raise RuntimeError(f"Only {len(rows)} labeled rows; need >= {min_rows}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        label, tx_id, tx_type, amount, _currency, customer_id, device_id, _ip, geo_raw, ts, _tenant = row
        geo = geo_raw if isinstance(geo_raw, dict) else (json.loads(geo_raw) if geo_raw else None)
        payload = {
            "transaction": {
                "transaction_id": tx_id,
                "transaction_type": tx_type,
                "amount": float(amount),
                "customer_id": customer_id,
                "device_id": device_id,
                "geolocation": geo,
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            },
            "customerProfile": {
                "avg_transaction_amount": float(amount),
                "typical_countries": ["CR"],
                "known_devices": [device_id] if device_id else [],
            },
            "recentTransactions": [],
            "geoHistory": [],
        }
        try:
            f = extract_features(payload)
            features.append(f)
            labels.append(1 if label == "fraud" else 0)
        except Exception:
            continue

    if len(features) < min_rows:
        raise RuntimeError(f"Only {len(features)} valid feature rows after extraction")

    return np.vstack(features), np.asarray(labels, dtype=np.int8)
