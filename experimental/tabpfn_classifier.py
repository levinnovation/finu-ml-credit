"""
TabPFN classifier wrapper for credit scoring.

Purpose:
  Zero-shot prior-data fitted network for tabular credit data.
  No training needed — inference works out of the box on CPU.

When to use:
  Primary ML model in the credit scoring ensemble.
  Provides probability of default prediction.

Inputs:
  - X: numpy array or pandas DataFrame (n_samples, n_features)
  - device: "cpu" (default) or "cuda"

Returns:
  - predict_proba(X) → numpy array [n_samples, 2] with [P(no_default), P(default)]
  - predict(X) → numpy array [n_samples] with class labels

Configuration:
  TABPFN_TOKEN env var for HuggingFace authentication
  model_path for loading pre-fitted models
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CreditClassifier:
    """TabPFN-based credit default probability classifier."""

    def __init__(self, device: str = "cpu", model_path: Optional[str] = None):
        self.device = device
        self.model_path = model_path
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from tabpfn import TabPFNClassifier

        if self.model_path and Path(self.model_path).exists():
            from tabpfn.model_loading import load_fitted_tabpfn_model

            self._model = load_fitted_tabpfn_model(self.model_path, device=self.device)
            logger.info(f"Loaded fitted model from {self.model_path}")
        else:
            self._model = TabPFNClassifier(device=self.device)
            logger.info("Loaded TabPFN classifier (zero-shot)")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._load()
        self._model.fit(X, y)
        logger.info(f"Fitted on {len(X)} samples, {X.shape[1]} features")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._load()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of default (class 1)."""
        self._load()
        proba = self._model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba

    def save(self, path: str):
        from tabpfn.model_loading import save_fitted_tabpfn_model

        save_fitted_tabpfn_model(self._model, path)
        logger.info(f"Model saved to {path}")
