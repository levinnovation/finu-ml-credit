"""Offline credit-default training pipeline.

CSV contract:
  - columns must include the personal_v1 feature schema
  - label column defaults to `defaulted`

This is intentionally offline. Production scoring only reads the champion
manifest produced by this script.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from config import settings
from ml.metrics import compute_quality_metrics, passes_promotion_gate
from ml.training_helpers import build_candidates, maybe_log_mlflow
from pipeline.schemas import PERSONAL_CREDIT_V1, get_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", default="defaulted")
    parser.add_argument(
        "--schema",
        default=PERSONAL_CREDIT_V1.version,
        help="Feature schema version to train against (personal_v1 | corporate_v1). See pipeline/schemas.py.",
    )
    parser.add_argument("--registry-path", default=str(Path(settings.model_cache_dir) / "model_registry.json"))
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="isotonic")
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow Model Registry promotion")
    parser.add_argument("--bootstrap", action="store_true", help="Skip promotion gate (synthetic bootstrap only)")
    parser.add_argument(
        "--data-source",
        default="production_decisions",
        help=(
            "Provenance label written to the manifest so /score and downstream "
            "consumers can tell a production-grade model apart from a synthetic "
            "placeholder. Ignored (forced to 'synthetic_bootstrap') when "
            "--bootstrap is set, since that path always skips or overrides the "
            "promotion gate."
        ),
    )
    args = parser.parse_args()
    data_source = "synthetic_bootstrap" if args.bootstrap else args.data_source
    schema = get_schema(args.schema)

    dataset_path = Path(args.dataset)
    df = pd.read_csv(dataset_path)
    missing = [f for f in schema.features + [args.label] if f not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing columns: {missing}")

    X = df[schema.features].astype(float)
    y = df[args.label].astype(int)
    stratify = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    out_dir = Path(settings.model_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for name, estimator in build_candidates(schema).items():
        calibrated = CalibratedClassifierCV(estimator, method=args.calibration, cv=3)
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_test)[:, 1]
        metrics = compute_quality_metrics(y_test, proba).as_dict()
        artifact = out_dir / f"{schema.name}_{name}_{int(time.time())}.pkl"
        with artifact.open("wb") as f:
            pickle.dump(calibrated, f)
        run_id = maybe_log_mlflow(
            name,
            calibrated,
            metrics,
            str(dataset_path),
            artifact,
            register=not args.no_register,
            schema=schema,
        )
        candidates.append({
            "name": schema.name,
            "stage": "candidate",
            "version": f"{name}-{int(time.time())}",
            "model_type": name,
            "feature_schema_version": schema.version,
            "mlflow_run_id": run_id,
            "artifact_path": str(artifact),
            "metrics": metrics,
            "thresholds": {"low_pd": 0.30, "medium_pd": 0.60},
            "data_source": data_source,
        })

    candidates.sort(key=lambda c: (c["metrics"].get("roc_auc") or 0), reverse=True)
    champion = candidates[0]
    ok, reason = passes_promotion_gate(champion["metrics"])
    champion["promotion"] = {"accepted": ok, "reason": reason}
    if not ok and not args.bootstrap:
        raise SystemExit(f"Best candidate rejected by promotion gate: {reason}")
    if not ok and args.bootstrap:
        champion["promotion"] = {"accepted": True, "reason": "bootstrap_skip_gate", "gate_reason": reason}
    champion["stage"] = "champion"
    challenger = candidates[1] if len(candidates) > 1 else None
    if challenger:
        challenger["stage"] = "challenger"

    # Merge into any existing manifest so training one schema (e.g.
    # corporate_v1) doesn't clobber another schema's already-registered
    # champion (e.g. personal_v1). "champions"/"challengers" are keyed by
    # model name; "champion"/"challenger" (singular, legacy) are kept in
    # sync for personal_v1 only, for older readers (api/credit_deploy.py,
    # any external tooling) that predate multi-schema support.
    registry_path = Path(args.registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if registry_path.exists():
        try:
            manifest = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
    champions = manifest.get("champions") or {}
    challengers = manifest.get("challengers") or {}
    champions[schema.name] = champion
    if challenger:
        challengers[schema.name] = challenger
    else:
        challengers.pop(schema.name, None)
    manifest["champions"] = champions
    manifest["challengers"] = challengers
    manifest["models"] = [m for m in manifest.get("models", []) if m.get("feature_schema_version") != schema.version] + candidates
    if schema.version == PERSONAL_CREDIT_V1.version:
        manifest["champion"] = champion
        manifest["challenger"] = challenger

    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(registry_path)
    print(json.dumps({"registry_path": str(registry_path), "champion": champion}, indent=2))


if __name__ == "__main__":
    main()
