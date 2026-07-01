# Experimental / not wired to production

Code in this directory is NOT imported by anything in `api/`, `models/`,
`credit/`, or `scripts/` -- confirmed via repo-wide grep before moving these
files here (2026-06-30, Credit Intelligence "mock a real" cleanup, Fase 5).

Moved from `models/` because their design contradicts the actual serving
architecture in `models/registry.py` / `api/score.py`: a single champion
model selected from `ml/training_helpers.build_candidates()`
(logistic_regression, random_forest, lightgbm, xgboost), NOT a blended
multi-model ensemble.

- `tabpfn_classifier.py` (was `models/classifier.py`): TabPFN zero-shot
  wrapper. TabPFN requires a one-time HuggingFace license acceptance
  (`TABPFN_TOKEN`) to download its checkpoint -- see
  `requirements-train.txt` and `docs/data-schema-cr.md`.
- `ensemble.py` (was `models/ensemble.py`): hardcoded blend of
  TabPFN (35%) + LightGBM (25%) + XGBoost (20%) + CatBoost (20%). CatBoost
  isn't even a dependency of this service.

If TabPFN/ensembling gets revisited, it needs to be reconciled with
`ml/training_helpers.build_candidates()` and `models/registry.py`'s
single-champion pattern first -- either as an additional candidate model
type, or by redesigning `api/score.py` to actually blend predictions
(today it calls `champion.estimator.predict_proba(X)` once, no blending).
