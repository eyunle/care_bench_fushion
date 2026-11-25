from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from .constants import LABEL_COLUMNS


@dataclass
class Calibration:
    coefficient: float
    intercept: float
    threshold: float
    fitted: bool = True

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return expit(self.coefficient * logits + self.intercept)


def fit_calibration(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> Calibration:
    valid = mask.astype(bool)
    x = logits[valid].reshape(-1, 1)
    y = targets[valid].astype(int)
    if len(np.unique(y)) < 2:
        return Calibration(1.0, 0.0, 0.5, fitted=False)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(x, y)
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    probabilities = expit(coefficient * x[:, 0] + intercept)
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    finite = np.isfinite(thresholds)
    if finite.any():
        index = int(np.argmax((tpr - fpr)[finite]))
        threshold = float(thresholds[finite][index])
    else:
        threshold = 0.5
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return Calibration(coefficient, intercept, threshold)


def save_calibrations(calibrations: Sequence[Calibration], path: Path) -> None:
    payload = {
        label: asdict(calibration)
        for label, calibration in zip(LABEL_COLUMNS, calibrations)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibrations(path: Path) -> List[Calibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Calibration(**payload[label]) for label in LABEL_COLUMNS]


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(
    targets: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y = targets.astype(int)
    prediction = (calibrated_probabilities >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, raw_probabilities))
        auprc = float(average_precision_score(y, raw_probabilities))
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "ppv": _safe_divide(tp, tp + fp),
        "npv": _safe_divide(tn, tn + fn),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier_raw": float(brier_score_loss(y, raw_probabilities)),
        "brier_calibrated": float(brier_score_loss(y, calibrated_probabilities)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
    }


def evaluate_predictions(
    logits: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    calibrations: Sequence[Calibration],
) -> Tuple[Dict[str, object], np.ndarray]:
    raw_probabilities = expit(logits)
    calibrated = np.zeros_like(raw_probabilities)
    results: Dict[str, object] = {"labels": {}}
    for index, (label, calibration) in enumerate(
        zip(LABEL_COLUMNS, calibrations)
    ):
        valid = target_mask[:, index].astype(bool)
        calibrated[:, index] = calibration.apply(logits[:, index])
        results["labels"][label] = binary_metrics(
            targets[valid, index],
            raw_probabilities[valid, index],
            calibrated[valid, index],
            calibration.threshold,
        )
    metric_names = ("auroc", "auprc", "f1", "balanced_accuracy", "brier_calibrated")
    results["macro"] = {
        metric: float(
            np.nanmean(
                [results["labels"][label][metric] for label in LABEL_COLUMNS]
            )
        )
        for metric in metric_names
    }
    return results, calibrated


def fit_all_calibrations(
    logits: np.ndarray, targets: np.ndarray, target_mask: np.ndarray
) -> List[Calibration]:
    return [
        fit_calibration(logits[:, i], targets[:, i], target_mask[:, i])
        for i in range(len(LABEL_COLUMNS))
    ]


def cluster_bootstrap_ci(
    prediction_frame: pd.DataFrame,
    calibrations: Sequence[Calibration],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Patient-cluster bootstrap. Repeated visits are sampled together.

    Returns percentile confidence intervals for every requested metric and label.
    """
    if n_bootstrap <= 0:
        return {}
    rng = np.random.RandomState(seed)
    patients = prediction_frame["id_register"].astype(str).unique()
    patient_rows = {
        patient: prediction_frame.index[
            prediction_frame["id_register"].astype(str) == patient
        ].to_numpy()
        for patient in patients
    }
    storage: Dict[str, Dict[str, List[float]]] = {
        label: {
            metric: []
            for metric in (
                "auroc",
                "auprc",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "f1",
                "balanced_accuracy",
                "brier_calibrated",
            )
        }
        for label in LABEL_COLUMNS
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_rows[patient] for patient in sampled])
        sample = prediction_frame.loc[indices]
        logits = sample[[f"logit_{label}" for label in LABEL_COLUMNS]].to_numpy()
        targets = sample[[f"target_{label}" for label in LABEL_COLUMNS]].to_numpy()
        masks = sample[[f"mask_{label}" for label in LABEL_COLUMNS]].to_numpy()
        metrics, _ = evaluate_predictions(logits, targets, masks, calibrations)
        for label in LABEL_COLUMNS:
            for metric in storage[label]:
                value = float(metrics["labels"][label][metric])
                if np.isfinite(value):
                    storage[label][metric].append(value)
    intervals: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, metric_values in storage.items():
        intervals[label] = {}
        for metric, values in metric_values.items():
            if values:
                low, high = np.percentile(values, [2.5, 97.5])
                intervals[label][metric] = {
                    "low": float(low),
                    "high": float(high),
                }
    return intervals


def paired_patient_permutation_test(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label: str,
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """Paired patient-level randomization test for AUROC differences."""
    keys = ["id_register", "sample_id"]
    columns = keys + [f"target_{label}", f"mask_{label}", f"prob_{label}"]
    merged = left[columns].merge(
        right[columns],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    valid = (
        merged[f"mask_{label}_left"].astype(bool)
        & merged[f"mask_{label}_right"].astype(bool)
    )
    merged = merged[valid].copy()
    target = merged[f"target_{label}_left"].to_numpy(dtype=int)
    if len(np.unique(target)) < 2:
        return float("nan")
    left_score = merged[f"prob_{label}_left"].to_numpy(dtype=float)
    right_score = merged[f"prob_{label}_right"].to_numpy(dtype=float)
    observed = roc_auc_score(target, left_score) - roc_auc_score(target, right_score)
    rng = np.random.RandomState(seed)
    patients = merged["id_register"].astype(str).unique()
    extreme = 0
    for _ in range(n_permutations):
        swap_patients = set(patients[rng.rand(len(patients)) < 0.5])
        swap = merged["id_register"].astype(str).isin(swap_patients).to_numpy()
        perm_left = np.where(swap, right_score, left_score)
        perm_right = np.where(swap, left_score, right_score)
        difference = roc_auc_score(target, perm_left) - roc_auc_score(
            target, perm_right
        )
        extreme += int(abs(difference) >= abs(observed))
    return float((extreme + 1) / (n_permutations + 1))


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    valid = [(name, value) for name, value in p_values.items() if np.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    adjusted: Dict[str, float] = {name: float("nan") for name in p_values}
    running = 0.0
    count = len(valid)
    for rank, (name, value) in enumerate(valid):
        corrected = min(1.0, (count - rank) * value)
        running = max(running, corrected)
        adjusted[name] = float(running)
    return adjusted

