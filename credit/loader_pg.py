"""Load labeled credit decisions from PostgreSQL for retrain."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.features import compute_corporate_features, compute_features
from pipeline.schemas import CORPORATE_CREDIT_V1, PERSONAL_CREDIT_V1


def _get_conn():
    import psycopg2

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL required for credit retrain")
    return psycopg2.connect(url)


def _label_from_row(final_action: str) -> int:
    """Proxy label: 'the pipeline declined this applicant', NOT a verified
    default outcome. Used only when no row exists in credit_outcomes for
    the decision (see migrations/036_credit_outcomes.sql). Conflating
    'declined' with 'defaulted' is a known limitation -- see
    docs/data-schema-cr.md -- kept only until real outcome data exists."""
    return 1 if final_action == "decline" else 0


# Outcome statuses treated as "defaulted" (label=1) once a real credit_outcomes
# row exists for the decision. delinquent_30 is intentionally excluded: it's
# informational only, too early in the maturation window to treat as default.
_DEFAULT_OUTCOME_STATUSES = {"default", "delinquent_90"}


def _label_from_outcome(status: str) -> int:
    return 1 if status in _DEFAULT_OUTCOME_STATUSES else 0


def count_labeled_decisions(
    since: Optional[str] = None, tenant_id: Optional[str] = None, customer_type: str = "personal"
) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            clauses = ["customer_type = %s"]
            params: list = [customer_type]
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
) -> Tuple[np.ndarray, np.ndarray, "LabelProvenance"]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            clauses = ["d.customer_type = 'personal'"]
            params: list = []
            if tenant_id:
                clauses.append("d.tenant_id = %s::uuid")
                params.append(tenant_id)
            params.append(limit)
            # Prefer the most recent real outcome per decision (credit_outcomes,
            # populated by ops/collections -- see migrations/036_credit_outcomes.sql)
            # over the decline/approve decision-proxy label.
            sql = f"""
                SELECT
                    d.application_context_snapshot,
                    d.external_checks_snapshot,
                    d.final_action,
                    d.scoring_result_snapshot,
                    o.status AS outcome_status
                FROM credit_decisions d
                LEFT JOIN LATERAL (
                    SELECT status
                    FROM credit_outcomes
                    WHERE credit_decision_id = d.id
                    ORDER BY observed_at DESC
                    LIMIT 1
                ) o ON true
                WHERE {' AND '.join(clauses)}
                ORDER BY d.decided_at DESC
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
    label_types = []
    for app_snap, ext_snap, final_action, _scoring, outcome_status in rows:
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
        if outcome_status:
            labels.append(_label_from_outcome(outcome_status))
            label_types.append("verified_outcome")
        else:
            labels.append(_label_from_row(final_action))
            label_types.append("decision_proxy")

    X = np.asarray(features_list, dtype=float)
    y = np.asarray(labels, dtype=int)
    provenance = LabelProvenance.from_list(label_types)
    return X, y, provenance


def load_pg_corporate_labels(
    min_rows: int = 200,
    tenant_id: Optional[str] = None,
    limit: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, "LabelProvenance"]:
    """Same as load_pg_labels but for customer_type='corporate' rows, using
    the corporate_v1 feature schema (CORPORATE_CREDIT_V1)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            clauses = ["d.customer_type = 'corporate'"]
            params: list = []
            if tenant_id:
                clauses.append("d.tenant_id = %s::uuid")
                params.append(tenant_id)
            params.append(limit)
            sql = f"""
                SELECT
                    d.application_context_snapshot,
                    d.external_checks_snapshot,
                    d.final_action,
                    d.scoring_result_snapshot,
                    o.status AS outcome_status
                FROM credit_decisions d
                LEFT JOIN LATERAL (
                    SELECT status
                    FROM credit_outcomes
                    WHERE credit_decision_id = d.id
                    ORDER BY observed_at DESC
                    LIMIT 1
                ) o ON true
                WHERE {' AND '.join(clauses)}
                ORDER BY d.decided_at DESC
                LIMIT %s
            """
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) < min_rows:
        raise RuntimeError(f"Need >= {min_rows} corporate credit_decisions rows, got {len(rows)}")

    features_list = []
    labels = []
    label_types = []
    for app_snap, ext_snap, final_action, _scoring, outcome_status in rows:
        app = app_snap if isinstance(app_snap, dict) else json.loads(app_snap or "{}")
        ext = ext_snap if isinstance(ext_snap, dict) else json.loads(ext_snap or "{}")
        corporate = app.get("corporate") or {}
        application = {
            "monthly_sales": corporate.get("ventasMensuales") or corporate.get("monthly_sales") or 0,
            "ebitda_annual": corporate.get("ebitdaAnual") or corporate.get("ebitda_annual") or 0,
            "current_assets": corporate.get("activoCorriente") or corporate.get("current_assets") or 0,
            "current_liabilities": corporate.get("pasivoCorriente") or corporate.get("current_liabilities") or 0,
            "total_assets": corporate.get("activoTotal") or corporate.get("total_assets") or 0,
            "total_liabilities": corporate.get("pasivoTotal") or corporate.get("total_liabilities") or 0,
            "years_in_business": corporate.get("anosOperacion") or corporate.get("years_in_business") or 0,
            "amount": corporate.get("montoSolicitado") or corporate.get("amount") or 0,
        }
        equifax = ext.get("equifax") or {}
        if equifax:
            application["active_debts"] = equifax.get("active_debts") or 0
            application["worst_delay_days"] = equifax.get("worst_delay_days") or 0
        feat = compute_corporate_features(application)
        features_list.append([feat[k] for k in CORPORATE_CREDIT_V1.features])
        if outcome_status:
            labels.append(_label_from_outcome(outcome_status))
            label_types.append("verified_outcome")
        else:
            labels.append(_label_from_row(final_action))
            label_types.append("decision_proxy")

    X = np.asarray(features_list, dtype=float)
    y = np.asarray(labels, dtype=int)
    provenance = LabelProvenance.from_list(label_types)
    return X, y, provenance


@dataclass
class LabelProvenance:
    """Breakdown of how many training labels came from real post-origination
    outcomes (credit_outcomes) vs. the decision-proxy fallback. Logged to
    MLflow so nobody mistakes a proxy-label-only model for one trained on
    verified performance data."""

    verified_outcome: int
    decision_proxy: int

    @classmethod
    def from_list(cls, label_types: list[str]) -> "LabelProvenance":
        return cls(
            verified_outcome=sum(1 for t in label_types if t == "verified_outcome"),
            decision_proxy=sum(1 for t in label_types if t == "decision_proxy"),
        )

    @property
    def total(self) -> int:
        return self.verified_outcome + self.decision_proxy

    @property
    def verified_pct(self) -> float:
        return round(self.verified_outcome / self.total, 4) if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "label_verified_outcome_count": self.verified_outcome,
            "label_decision_proxy_count": self.decision_proxy,
            "label_verified_pct": self.verified_pct,
        }


def load_pg_eligibility_labels(
    min_rows: int = 200,
    tenant_id: Optional[str] = None,
    limit: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Real feature vectors from credit_decisions, labeled by applying the
    deterministic hard_rule_check() (models/eligibility.py) to each one.

    Unlike load_pg_labels' default-risk proxy label, this is NOT an
    approximation: hard_rule_check is a pure function of the features, so the
    label is exactly correct by construction. The point of training a model
    on it is to let a LightGBM classifier generalize the hard-rule boundary
    smoothly over the REAL feature distribution seen in production (instead
    of only the synthetic distribution used by training/train_eligibility_model.py),
    which should behave better on borderline real applicants than either the
    hard rules alone or a synthetic-only model.
    """
    from models.eligibility import hard_rule_check

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
                SELECT application_context_snapshot, external_checks_snapshot
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
    for app_snap, ext_snap in rows:
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
        passes, _reasons = hard_rule_check(feat)
        features_list.append([feat[k] for k in PERSONAL_CREDIT_V1.features])
        labels.append(1 if passes else 0)

    X = np.asarray(features_list, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y


def export_labels_csv(output_path: str, tenant_id: Optional[str] = None, limit: int = 50_000) -> int:
    """Export training CSV with personal_v1 features + defaulted label."""
    X, y, _provenance = load_pg_labels(min_rows=1, tenant_id=tenant_id, limit=limit)
    df = pd.DataFrame(X, columns=PERSONAL_CREDIT_V1.features)
    df["defaulted"] = y
    df.to_csv(output_path, index=False)
    return len(df)


def export_corporate_labels_csv(output_path: str, tenant_id: Optional[str] = None, limit: int = 50_000) -> int:
    """Export training CSV with corporate_v1 features + defaulted label."""
    X, y, _provenance = load_pg_corporate_labels(min_rows=1, tenant_id=tenant_id, limit=limit)
    df = pd.DataFrame(X, columns=CORPORATE_CREDIT_V1.features)
    df["defaulted"] = y
    df.to_csv(output_path, index=False)
    return len(df)
