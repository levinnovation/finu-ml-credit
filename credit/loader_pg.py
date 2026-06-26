"""Load labeled credit decisions from PostgreSQL for retrain."""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.features import compute_features
from pipeline.schemas import PERSONAL_CREDIT_V1


def _get_conn():
    import psycopg2

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL required for credit retrain")
    return psycopg2.connect(url)


def _label_from_row(final_action: str) -> int:
    """Proxy label until post-origination default data exists."""
    return 1 if final_action == "decline" else 0


def count_labeled_decisions(since: Optional[str] = None, tenant_id: Optional[str] = None) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            clauses = ["customer_type = 'personal'"]
            params: list = []
            if tenant_id:
                clauses.append("tenant_id = %s::uuid")
                params.append(tenant_id)
            if since:
                clauses.append("decided_at > %s::timestamptz")
                params.append(since)
            sql = f"SELECT count(*) FROM credit_decisions WHERE {' AND '.join(clauses)}"
            cur.execute(sql, params)
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def load_pg_labels(
    min_rows: int = 200,
    tenant_id: Optional[str] = None,
    limit: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            clauses = ["customer_type = 'personal'"]
            params: list = []
            if tenant_id:
                clauses.append("tenant_id = %s::uuid")
                params.append(tenant_id)
            params.append(limit)
            sql = f"""
                SELECT application_context_snapshot, external_checks_snapshot,
                       final_action, scoring_result_snapshot
                FROM credit_decisions
                WHERE {' AND '.join(clauses)}
                ORDER BY decided_at DESC
                LIMIT %s
            """
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) < min_rows:
        raise RuntimeError(f"Need >= {min_rows} credit_decisions rows, got {len(rows)}")

    features_list = []
    labels = []
    for app_snap, ext_snap, final_action, _scoring in rows:
        app = app_snap if isinstance(app_snap, dict) else json.loads(app_snap or "{}")
        ext = ext_snap if isinstance(ext_snap, dict) else json.loads(ext_snap or "{}")
        personal = app.get("personal") or {}
        application = {
            "monthly_income": personal.get("ingresoMensualDeclarado") or personal.get("monthly_income") or 0,
            "amount": personal.get("montoSolicitado") or personal.get("amount") or 0,
            "term_months": personal.get("plazoMeses") or personal.get("term_months") or 12,
            "age": personal.get("age") or 35,
            "employment_months": personal.get("employment_months") or 24,
            "employment_type": personal.get("employment_type") or "asalariado",
        }
        credit_data = {}
        equifax = ext.get("equifax") or {}
        if equifax:
            credit_data = {
                "score": equifax.get("score") or equifax.get("puntaje") or 650,
                "active_debts": equifax.get("active_debts") or 0,
                "worst_delay_days": equifax.get("worst_delay_days") or 0,
            }
        feat = compute_features(application, credit_data=credit_data)
        features_list.append([feat[k] for k in PERSONAL_CREDIT_V1.features])
        labels.append(_label_from_row(final_action))

    X = np.asarray(features_list, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y


def export_labels_csv(output_path: str, tenant_id: Optional[str] = None, limit: int = 50_000) -> int:
    """Export training CSV with personal_v1 features + defaulted label."""
    X, y = load_pg_labels(min_rows=1, tenant_id=tenant_id, limit=limit)
    df = pd.DataFrame(X, columns=PERSONAL_CREDIT_V1.features)
    df["defaulted"] = y
    df.to_csv(output_path, index=False)
    return len(df)
