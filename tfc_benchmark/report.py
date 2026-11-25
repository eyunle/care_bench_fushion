from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import LABEL_COLUMNS
from .metrics import holm_adjust, paired_patient_permutation_test

DISPLAY_LABELS = {
    "影像学脂肪肝": "Imaging fatty liver",
    "糖尿病前期": "Prediabetes",
    "MAFLD": "MAFLD",
}
FILE_LABELS = {
    "影像学脂肪肝": "imaging_fatty_liver",
    "糖尿病前期": "prediabetes",
    "MAFLD": "mafld",
}


def collect_results(output_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metrics_path in output_root.glob("**/metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config_path = metrics_path.parent / "config.yaml"
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for condition, condition_metrics in metrics.items():
            path_text = str(metrics_path).replace("\\", "/")
            if "/baselines/" in path_text:
                family = "baseline"
            elif "/fusion_methods/" in path_text:
                family = "fusion"
            else:
                family = "other"
            for label in LABEL_COLUMNS:
                values = condition_metrics["labels"][label]
                rows.append(
                    {
                        "model": config["model"],
                        "family": family,
                        "seed": config["seed"],
                        "enabled_modalities": "+".join(config["enabled_modalities"]),
                        "condition": condition,
                        "label": label,
                        **{
                            key: value
                            for key, value in values.items()
                            if isinstance(value, (int, float))
                        },
                        "run_dir": str(metrics_path.parent),
                    }
                )
    return pd.DataFrame(rows)


def collect_predictions(output_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for prediction_path in output_root.glob("**/predictions_*.csv"):
        config_path = prediction_path.parent / "config.yaml"
        if not config_path.exists():
            continue
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(
                prediction_path,
                dtype={"id_register": str, "sample_id": str},
            )
        except Exception:
            continue
        path_text = str(prediction_path).replace("\\", "/")
        if "/baselines/" in path_text:
            family = "baseline"
        elif "/fusion_methods/" in path_text:
            family = "fusion"
        else:
            family = "other"
        frame["model"] = config["model"]
        frame["family"] = family
        frame["seed"] = config["seed"]
        frame["enabled_modalities"] = "+".join(config["enabled_modalities"])
        frame["condition"] = prediction_path.stem.replace("predictions_", "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    keys = [
        "family",
        "model",
        "enabled_modalities",
        "condition",
        "id_register",
        "sample_id",
        "visit_year",
        "device_id",
    ]
    aggregations = {}
    for label in LABEL_COLUMNS:
        aggregations[f"prob_{label}"] = "mean"
        aggregations[f"logit_{label}"] = "mean"
        aggregations[f"target_{label}"] = "first"
        aggregations[f"mask_{label}"] = "first"
    return predictions.groupby(keys, as_index=False).agg(aggregations)


def create_plots(ensemble: pd.DataFrame, aggregate: pd.DataFrame, report_dir: Path):
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        return []
    created = []
    complete = ensemble[ensemble["condition"] == "complete"]
    for label in LABEL_COLUMNS:
        valid_groups = []
        for keys, group in complete.groupby(
            ["family", "model", "enabled_modalities"], sort=False
        ):
            valid = group[f"mask_{label}"].astype(bool)
            data = group[valid]
            if data[f"target_{label}"].nunique() < 2:
                continue
            display = f"{keys[1]} ({keys[2]})"
            valid_groups.append((display, data))
        for plot_name in ("roc", "pr", "calibration"):
            figure, axis = plt.subplots(figsize=(8, 6))
            for display, data in valid_groups:
                y = data[f"target_{label}"].to_numpy(dtype=int)
                p = data[f"prob_{label}"].to_numpy(dtype=float)
                if plot_name == "roc":
                    x, y_value, _ = roc_curve(y, p)
                    axis.plot(x, y_value, linewidth=1, alpha=0.8, label=display)
                    axis.set_xlabel("False positive rate")
                    axis.set_ylabel("True positive rate")
                elif plot_name == "pr":
                    y_value, x, _ = precision_recall_curve(y, p)
                    axis.plot(x, y_value, linewidth=1, alpha=0.8, label=display)
                    axis.set_xlabel("Recall")
                    axis.set_ylabel("Precision")
                else:
                    y_value, x = calibration_curve(y, p, n_bins=10, strategy="quantile")
                    axis.plot(x, y_value, marker="o", linewidth=1, alpha=0.8, label=display)
                    axis.set_xlabel("Mean predicted probability")
                    axis.set_ylabel("Observed frequency")
            axis.set_title(f"{DISPLAY_LABELS[label]} - {plot_name}")
            if plot_name in {"roc", "calibration"}:
                axis.plot([0, 1], [0, 1], "k--", linewidth=1)
            axis.legend(fontsize=6, ncol=2, loc="best")
            figure.tight_layout()
            path = report_dir / f"{plot_name}_{FILE_LABELS[label]}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            created.append(path)

    robustness = aggregate.pivot_table(
        index=["family", "model", "enabled_modalities", "label"],
        columns="condition",
        values="auroc_mean",
    ).reset_index()
    if "complete" in robustness.columns:
        conditions = [
            condition
            for condition in (
                "missing_tongue",
                "missing_face",
                "missing_clinical",
                "missing_both_images",
            )
            if condition in robustness.columns
        ]
        for label in LABEL_COLUMNS:
            data = robustness[robustness["label"] == label].copy()
            if data.empty:
                continue
            labels = [
                f"{row.model} ({row.enabled_modalities})"
                for row in data.itertuples()
            ]
            figure, axis = plt.subplots(figsize=(12, max(5, len(data) * 0.25)))
            y_positions = np.arange(len(data))
            width = 0.8 / max(len(conditions), 1)
            for index, condition in enumerate(conditions):
                delta = data[condition] - data["complete"]
                axis.barh(
                    y_positions + index * width,
                    delta,
                    height=width,
                    label=condition,
                )
            axis.set_yticks(y_positions + width * (len(conditions) - 1) / 2)
            axis.set_yticklabels(labels, fontsize=7)
            axis.axvline(0, color="black", linewidth=1)
            axis.set_xlabel("AUROC change versus complete modalities")
            axis.set_title(
                f"{DISPLAY_LABELS[label]} - missing modality robustness"
            )
            axis.legend(fontsize=8)
            figure.tight_layout()
            path = report_dir / f"missing_robustness_{FILE_LABELS[label]}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            created.append(path)
    return created


def significance_table(
    ensemble: pd.DataFrame, n_permutations: int, seed: int = 42
) -> pd.DataFrame:
    if ensemble.empty or n_permutations <= 0:
        return pd.DataFrame()
    complete = ensemble[ensemble["condition"] == "complete"].copy()
    full = complete[
        complete["enabled_modalities"] == "tongue+face+clinical"
    ]
    rows = []
    for label in LABEL_COLUMNS:
        unimodal_candidates = complete[
            complete["enabled_modalities"].isin(["tongue", "face", "clinical"])
        ]
        best_reference = None
        best_auroc = -float("inf")
        from sklearn.metrics import roc_auc_score

        for keys, group in unimodal_candidates.groupby(
            ["family", "model", "enabled_modalities"]
        ):
            valid = group[f"mask_{label}"].astype(bool)
            data = group[valid]
            if data[f"target_{label}"].nunique() < 2:
                continue
            score = roc_auc_score(
                data[f"target_{label}"], data[f"prob_{label}"]
            )
            if score > best_auroc:
                best_auroc = score
                best_reference = (keys, group)
        late_reference = full[
            (full["model"] == "latefusion") & (full["family"] == "fusion")
        ]
        if late_reference.empty:
            candidates = full[full["model"] == "latefusion"]
            if not candidates.empty:
                first_family = candidates["family"].iloc[0]
                late_reference = candidates[candidates["family"] == first_family]
        references = []
        if best_reference is not None:
            references.append(
                (f"best_unimodal:{best_reference[0][2]}", best_reference[1])
            )
        if not late_reference.empty:
            references.append(("latefusion", late_reference))
        for keys, candidate in full.groupby(
            ["family", "model", "enabled_modalities"]
        ):
            for reference_name, reference in references:
                if (
                    reference_name == "latefusion"
                    and keys[1] == "latefusion"
                    and keys[0] == reference["family"].iloc[0]
                ):
                    continue
                comparison = f"{keys[0]}:{keys[1]}__vs__{reference_name}"
                p_value = paired_patient_permutation_test(
                    candidate,
                    reference,
                    label,
                    n_permutations=n_permutations,
                    seed=seed,
                )
                rows.append(
                    {
                        "label": label,
                        "comparison": comparison,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        adjusted = {}
        for label, group in result.groupby("label"):
            mapping = dict(zip(group["comparison"], group["p_value"]))
            for comparison, value in holm_adjust(mapping).items():
                adjusted[(label, comparison)] = value
        result["p_holm"] = [
            adjusted[(row.label, row.comparison)] for row in result.itertuples()
        ]
    return result


def aggregate_results(
    output_root: Path,
    report_dir: Path,
    n_permutations: int = 5000,
) -> Dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = collect_results(output_root)
    if raw.empty:
        raise ValueError(f"no metrics.json files found under {output_root}")
    raw_path = report_dir / "all_seed_metrics.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    metric_columns = [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "brier_raw",
        "brier_calibrated",
    ]
    available = [column for column in metric_columns if column in raw.columns]
    group_columns = ["family", "model", "enabled_modalities", "condition", "label"]
    mean = raw.groupby(group_columns)[available].mean().add_suffix("_mean")
    std = raw.groupby(group_columns)[available].std().add_suffix("_std")
    aggregate = mean.join(std).reset_index()
    aggregate_path = report_dir / "aggregate_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    complete = aggregate[aggregate["condition"] == "complete"].copy()
    ranking = complete.sort_values(["label", "auroc_mean"], ascending=[True, False])
    ranking_path = report_dir / "auroc_ranking.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    robustness = aggregate.pivot_table(
        index=["family", "model", "enabled_modalities", "label"],
        columns="condition",
        values="auroc_mean",
    ).reset_index()
    if "complete" in robustness.columns:
        for condition in (
            "missing_tongue",
            "missing_face",
            "missing_clinical",
            "missing_both_images",
        ):
            if condition in robustness.columns:
                robustness[f"delta_{condition}"] = (
                    robustness[condition] - robustness["complete"]
                )
    robustness_path = report_dir / "missing_modality_robustness.csv"
    robustness.to_csv(robustness_path, index=False, encoding="utf-8-sig")
    predictions = collect_predictions(output_root)
    ensemble = ensemble_predictions(predictions)
    ensemble_path = report_dir / "seed_ensemble_predictions.csv"
    if not ensemble.empty:
        ensemble.to_csv(ensemble_path, index=False, encoding="utf-8-sig")
    significance = significance_table(ensemble, n_permutations=n_permutations)
    significance_path = report_dir / "paired_significance.csv"
    if not significance.empty:
        significance.to_csv(significance_path, index=False, encoding="utf-8-sig")
    plots = create_plots(ensemble, aggregate, report_dir)
    outputs = {
        "all_seed_metrics": raw_path,
        "aggregate_metrics": aggregate_path,
        "auroc_ranking": ranking_path,
        "robustness": robustness_path,
    }
    if not ensemble.empty:
        outputs["ensemble_predictions"] = ensemble_path
    if not significance.empty:
        outputs["paired_significance"] = significance_path
    for index, path in enumerate(plots):
        outputs[f"plot_{index:02d}"] = path
    return outputs

