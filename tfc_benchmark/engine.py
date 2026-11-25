from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from .constants import LABEL_COLUMNS
from .data import (
    ClinicalStats,
    TongueFaceClinicalDataset,
    compute_pos_weight,
    create_loaders,
    fit_clinical_stats,
    load_manifest,
)
from .metrics import (
    cluster_bootstrap_ci,
    evaluate_predictions,
    fit_all_calibrations,
    save_calibrations,
)
from .models import ModelOutput, build_model
from .serialization import json_safe


@dataclass
class ExperimentConfig:
    manifest: str
    output_root: str = "experiments_tfc"
    model: str = "latefusion"
    seed: int = 42
    image_size: int = 224
    image_backbone: str = "resnet50"
    pretrained: bool = True
    hidden_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 50
    patience: int = 10
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    aux_loss_weight: float = 0.1
    precision: str = "16-mixed"
    gradient_accumulation: int = 1
    bootstrap_samples: int = 2000
    enabled_modalities: Sequence[str] = field(
        default_factory=lambda: ["tongue", "face", "clinical"]
    )
    model_params: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, overrides: Optional[Mapping[str, object]] = None):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if overrides:
            payload.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def experiment_dir(config: ExperimentConfig) -> Path:
    return Path(config.output_root) / config.model / f"seed_{config.seed}"


def model_kwargs(config: ExperimentConfig) -> Dict[str, object]:
    output: Dict[str, object] = {
        "hidden_dim": config.hidden_dim,
        "num_labels": len(LABEL_COLUMNS),
        "clinical_dim": 10,
        "dropout": config.dropout,
        "pretrained": config.pretrained,
        "image_backbone": config.image_backbone,
    }
    output.update(config.model_params)
    return output


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return output


def _apply_enabled_modalities(
    batch: MutableMapping[str, object], enabled_modalities: Sequence[str]
) -> None:
    mapping = {"tongue": 0, "face": 1, "clinical": 2}
    enabled = {mapping[name] for name in enabled_modalities}
    mask = batch["modality_mask"]
    for index in range(3):
        if index not in enabled:
            mask[:, index] = 0


def masked_weighted_bce(
    logits: Tensor, targets: Tensor, target_mask: Tensor, pos_weight: Tensor
) -> Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    return (loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)


def macro_auroc(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    values = []
    for index in range(targets.shape[1]):
        valid = mask[:, index].astype(bool)
        y = targets[valid, index]
        if len(np.unique(y)) < 2:
            continue
        values.append(roc_auc_score(y, scores[valid, index]))
    return float(np.mean(values)) if values else float("nan")


def predict(
    model: nn.Module,
    loader,
    device: torch.device,
    enabled_modalities: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    logits: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            _apply_enabled_modalities(batch, enabled_modalities)
            output: ModelOutput = model(batch)
            logits.append(output.logits.detach().float().cpu().numpy())
            targets.append(batch["targets"].detach().float().cpu().numpy())
            masks.append(batch["target_mask"].detach().float().cpu().numpy())
            size = len(batch["id_register"])
            for index in range(size):
                metadata.append(
                    {
                        "id_register": str(batch["id_register"][index]),
                        "sample_id": str(batch["sample_id"][index]),
                        "visit_year": str(batch["visit_year"][index]),
                        "device_id": str(batch["device_id"][index]),
                    }
                )
    return (
        np.concatenate(logits),
        np.concatenate(targets),
        np.concatenate(masks),
        pd.DataFrame(metadata),
    )


def _prediction_frame(
    metadata: pd.DataFrame,
    logits: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    calibrated: np.ndarray,
) -> pd.DataFrame:
    output = metadata.copy()
    for index, label in enumerate(LABEL_COLUMNS):
        output[f"logit_{label}"] = logits[:, index]
        output[f"prob_{label}"] = calibrated[:, index]
        output[f"target_{label}"] = targets[:, index]
        output[f"mask_{label}"] = masks[:, index].astype(int)
    return output


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Optional[Path] = None,
    bootstrap_samples: Optional[int] = None,
) -> Dict[str, object]:
    run_dir = experiment_dir(config)
    checkpoint_path = checkpoint_path or run_dir / "best.pt"
    device = _device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_kwargs = checkpoint.get("model_kwargs", model_kwargs(config))
    model = build_model(config.model, **checkpoint_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    stats_path = run_dir / "clinical_stats.json"
    train_loader, val_loader, _, stats, frame = create_loaders(
        Path(config.manifest),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        image_size=config.image_size,
        stats_path=stats_path,
    )
    val_logits, val_targets, val_masks, _ = predict(
        model, val_loader, device, config.enabled_modalities
    )
    calibrations = fit_all_calibrations(val_logits, val_targets, val_masks)
    save_calibrations(calibrations, run_dir / "calibration.json")

    conditions = (
        "complete",
        "missing_tongue",
        "missing_face",
        "missing_clinical",
        "missing_both_images",
    )
    all_results: Dict[str, object] = {}
    for condition in conditions:
        test_frame = frame[frame["split"] == "test"].copy()
        dataset = TongueFaceClinicalDataset(
            test_frame,
            stats,
            train=False,
            image_size=config.image_size,
            missing_condition=condition,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        logits, targets, masks, metadata = predict(
            model, loader, device, config.enabled_modalities
        )
        results, calibrated = evaluate_predictions(
            logits, targets, masks, calibrations
        )
        prediction_frame = _prediction_frame(
            metadata, logits, targets, masks, calibrated
        )
        prediction_path = run_dir / f"predictions_{condition}.csv"
        prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        if condition == "complete":
            n_bootstrap = (
                config.bootstrap_samples
                if bootstrap_samples is None
                else bootstrap_samples
            )
            results["bootstrap_95ci"] = cluster_bootstrap_ci(
                prediction_frame,
                calibrations,
                n_bootstrap=n_bootstrap,
                seed=config.seed,
            )
        all_results[condition] = results
    (run_dir / "metrics.json").write_text(
        json.dumps(json_safe(all_results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_results


def train_experiment(
    config: ExperimentConfig,
    evaluate_after_training: bool = True,
    bootstrap_samples: Optional[int] = None,
) -> Dict[str, object]:
    set_seed(config.seed)
    run_dir = experiment_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")
    stats_path = run_dir / "clinical_stats.json"
    train_loader, val_loader, _, _, frame = create_loaders(
        Path(config.manifest),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        image_size=config.image_size,
        stats_path=stats_path,
    )
    train_frame = frame[frame["split"] == "train"]
    pos_weight = compute_pos_weight(train_frame)
    device = _device()
    kwargs = model_kwargs(config)
    model = build_model(config.model, **kwargs).to(device)
    pos_weight = pos_weight.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, config.patience // 3)
    )
    use_amp = (
        config.precision == "16-mixed"
        and device.type == "cuda"
        and hasattr(torch.cuda, "amp")
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_batches = 0
        for batch_index, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            _apply_enabled_modalities(batch, config.enabled_modalities)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output: ModelOutput = model(batch)
                prediction_loss = masked_weighted_bce(
                    output.logits,
                    batch["targets"],
                    batch["target_mask"],
                    pos_weight,
                )
                loss = prediction_loss + config.aux_loss_weight * output.aux_loss
                scaled_loss = loss / max(config.gradient_accumulation, 1)
            scaler.scale(scaled_loss).backward()
            if (
                (batch_index + 1) % max(config.gradient_accumulation, 1) == 0
                or batch_index + 1 == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        val_logits, val_targets, val_masks, _ = predict(
            model, val_loader, device, config.enabled_modalities
        )
        score = macro_auroc(val_logits, val_targets, val_masks)
        scheduler.step(score)
        record = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total_batches, 1),
            "val_macro_auroc": float(score),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"[{config.model} seed={config.seed}] epoch={epoch + 1} "
            f"loss={record['train_loss']:.5f} val_macro_auroc={score:.5f}"
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": kwargs,
                    "config": asdict(config),
                    "best_epoch": best_epoch,
                    "best_val_macro_auroc": best_score,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    training_summary: Dict[str, object] = {
        "model": config.model,
        "seed": config.seed,
        "best_epoch": int(best_epoch),
        "best_val_macro_auroc": float(best_score),
        "epochs_completed": len(history),
        "device": str(device),
    }
    if evaluate_after_training:
        training_summary["evaluation"] = evaluate_checkpoint(
            config, bootstrap_samples=bootstrap_samples
        )
    (run_dir / "run_summary.json").write_text(
        json.dumps(json_safe(training_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return training_summary

