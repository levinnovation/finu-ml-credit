"""Model quality metrics and promotion gates for credit ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class QualityMetrics:
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    brier_score: Optional[float]
    ks_statistic: Optional[float]
    precision_at_threshold: Optional[float]
    recall_at_threshold: Optional[float]

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "ks_statistic": self.ks_statistic,
            "precision_at_threshold": self.precision_at_threshold,
            "recall_at_threshold": self.recall_at_threshold,
        }


def _binary_counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, fp, fn


def roc_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos_ranks = ranks[y == 1].sum()
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    return round(float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 6)


def pr_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(y == 1)), 1)
    # Average precision: sum precision at each positive label's recall step.
    positives = y_sorted == 1
    if not np.any(positives):
        return None
    return round(float(np.sum(precision[positives]) / np.sum(positives)), 6)


def ks_statistic(y_true: Iterable[int], y_score: Iterable[float]) -> Optional[float]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    if len(np.unique(y)) < 2:
        return None
    pos = np.sort(s[y == 1])
    neg = np.sort(s[y == 0])
    thresholds = np.unique(s)
    diffs = []
    for t in thresholds:
        pos_cdf = float(np.mean(pos <= t)) if len(pos) else 0.0
        neg_cdf = float(np.mean(neg <= t)) if len(neg) else 0.0
        diffs.append(abs(pos_cdf - neg_cdf))
    return round(float(max(diffs)), 6) if diffs else None


def compute_quality_metrics(
    y_true: Iterable[int],
    probability_default: Iterable[float],
    threshold: float = 0.5,
) -> QualityMetrics:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability_default), dtype=float)
    pred = (p >= threshold).astype(int)
    tp, fp, fn = _binary_counts(y, pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    brier = float(np.mean((p - y) ** 2)) if len(y) else None
    return QualityMetrics(
        roc_auc=roc_auc(y, p),
        pr_auc=pr_auc(y, p),
        brier_score=round(brier, 6) if brier is not None else None,
        ks_statistic=ks_statistic(y, p),
        precision_at_threshold=round(float(precision), 6),
        recall_at_threshold=round(float(recall), 6),
    )


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    buckets: int = 10,
) -> Optional[float]:
    exp = np.asarray(list(expected), dtype=float)
    act = np.asarray(list(actual), dtype=float)
    if len(exp) == 0 or len(act) == 0:
        return None
    edges = np.percentile(exp, np.linspace(0, 100, buckets + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)
    exp_pct = np.maximum(exp_counts / max(exp_counts.sum(), 1), 1e-6)
    act_pct = np.maximum(act_counts / max(act_counts.sum(), 1), 1e-6)
    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 6)


def passes_promotion_gate(
    candidate: Dict[str, Optional[float]],
    champion: Optional[Dict[str, Optional[float]]] = None,
    min_auc: float = 0.62,
    min_pr_auc: float = 0.20,
    max_brier: float = 0.25,
    min_auc_delta: float = 0.005,
) -> tuple[bool, str]:
    auc = candidate.get("roc_auc")
    pr_auc = candidate.get("pr_auc")
    brier = candidate.get("brier_score")
    if auc is None or auc < min_auc:
        return False, "roc_auc_below_minimum"
    if pr_auc is None or pr_auc < min_pr_auc:
        return False, "pr_auc_below_minimum"
    if brier is None or brier > max_brier:
        return False, "brier_score_above_maximum"
    if champion and champion.get("roc_auc") is not None:
        if auc < float(champion["roc_auc"]) + min_auc_delta:
            return False, "does_not_beat_champion_margin"
    return True, "passed"
