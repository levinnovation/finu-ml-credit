#!/usr/bin/env bash
# Bootstrap credit champion locally (synthetic dataset + train + optional push)
#
# Usage:
#   bash services/finu-ml-credit/scripts/bootstrap_champion.sh
#   bash services/finu-ml-credit/scripts/bootstrap_champion.sh --push

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA="/tmp/finu-models/bootstrap_personal_v1.csv"
REGISTRY="/tmp/finu-models/model_registry.json"
PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1; shift ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /tmp/finu-models

echo "→ Generating bootstrap dataset..."
python3 scripts/generate_bootstrap_dataset.py --output "$DATA" --rows 2000

echo "→ Training champion (no MLflow register unless MLFLOW_TRACKING_URI set)..."
python3 scripts/train_credit_default.py --dataset "$DATA" --registry-path "$REGISTRY" --no-register --bootstrap

echo "→ Registry: $REGISTRY"
python3 -c "import json; print(json.load(open('$REGISTRY'))['champion']['version'])"

if [[ "$PUSH" -eq 1 ]]; then
  FINU_CREDIT_REGISTRY="$REGISTRY" bash "$ROOT/../../apps/fintech-saas/scripts/push-credit-model-to-railway.sh"
fi

echo "✓ Bootstrap complete"
