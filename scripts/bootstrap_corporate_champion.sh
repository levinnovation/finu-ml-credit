#!/usr/bin/env bash
# Bootstrap corporate credit champion locally (synthetic dataset + train + optional push)
#
# Usage:
#   bash services/finu-ml-credit/scripts/bootstrap_corporate_champion.sh
#   bash services/finu-ml-credit/scripts/bootstrap_corporate_champion.sh --push

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA="/tmp/finu-models/bootstrap_corporate_v1.csv"
REGISTRY="${FINU_CREDIT_REGISTRY:-/tmp/finu-models/model_registry.json}"
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

echo "→ Generating synthetic corporate dataset..."
python3 -c "
from synthetic.generator_corporate import generate, CorporateGeneratorConfig
df = generate(CorporateGeneratorConfig(n_rows=5000))
df.to_csv('$DATA', index=False)
print(f'Wrote {len(df)} rows, defaulted rate {df[\"defaulted\"].mean():.3f}')
"

echo "→ Training corporate champion (merges into existing registry, no MLflow register unless MLFLOW_TRACKING_URI set)..."
python3 scripts/train_credit_default.py --dataset "$DATA" --schema corporate_v1 --registry-path "$REGISTRY" --no-register --bootstrap

echo "→ Registry: $REGISTRY"
python3 -c "import json; print(json.load(open('$REGISTRY'))['champions']['credit_default_corporate']['version'])"

if [[ "$PUSH" -eq 1 ]]; then
  FINU_CREDIT_REGISTRY="$REGISTRY" bash "$ROOT/../../apps/fintech-saas/scripts/push-credit-model-to-railway.sh"
fi

echo "✓ Corporate bootstrap complete"
