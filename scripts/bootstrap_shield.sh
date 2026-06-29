#!/usr/bin/env bash
# Bootstrap Shield fraud model (synthetic labels + LightGBM) for finu-ml-credit volume
#
# Usage:
#   bash services/finu-ml-credit/scripts/bootstrap_shield.sh
#   SHIELD_MODEL_DIR=/tmp/finu-models/shield/v2 bash services/finu-ml-credit/scripts/bootstrap_shield.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${SHIELD_MODEL_DIR:-/tmp/finu-models/shield/v2}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"

python3 - "$OUT_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit(f"lightgbm required: {e}") from e

out_dir = Path(sys.argv[1])
X, y = make_classification(
    n_samples=800, n_features=14, n_informative=10, n_redundant=2,
    weights=[0.85, 0.15], random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = lgb.LGBMClassifier(n_estimators=80, random_state=42, verbose=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
metrics = {
    "f1": float(f1_score(y_test, preds, zero_division=0)),
    "precision": float(precision_score(y_test, preds, zero_division=0)),
    "recall": float(recall_score(y_test, preds, zero_division=0)),
}

name, version = "lightgbm", "v2"
joblib_path = out_dir / f"{name}_{version}.joblib"
meta_path = out_dir / f"{name}_{version}.metadata.json"
joblib.dump(model, joblib_path)
meta_path.write_text(json.dumps({
    "name": name, "version": version, "metrics": metrics,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "n_features": 14, "source": "bootstrap_synthetic",
}, indent=2))
(out_dir / "active_model.json").write_text(json.dumps({
    "name": name, "version": version, "joblib_path": joblib_path.name,
    "metrics": metrics, "promoted_at": datetime.now(timezone.utc).isoformat(),
}, indent=2))
print(f"✓ Shield bootstrap → {out_dir} (F1={metrics['f1']:.3f})")
PY
