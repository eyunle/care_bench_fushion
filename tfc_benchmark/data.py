from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import (
    CLINICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    IMAGENET_MEAN,
    IMAGENET_MEAN_RGB,
    IMAGENET_STD,
    LABEL_COLUMNS,
)


@dataclass
class ClinicalStats:
    medians: Dict[str, float]
    centers: Dict[str, float]
    scales: Dict[str, float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "ClinicalStats":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def fit_clinical_stats(train_frame: pd.DataFrame) -> ClinicalStats:
    medians: Dict[str, float] = {}
    centers: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for column in CONTINUOUS_COLUMNS:
        values = pd.to_numeric(train_frame[column], errors="coerce")
        median = float(values.median())
        if not np.isfinite(median):
            median = 0.0
        filled = values.fillna(median)
        q1, q3 = np.nanpercentile(filled.to_numpy(dtype=float), [25, 75])
        scale = float(q3 - q1)
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        medians[column] = median
        centers[column] = median
        scales[column] = scale
    gender = pd.to_numeric(train_frame["gender"], errors="coerce")
    gender_median = float(gender.median())
    if not np.isfinite(gender_median):
        gender_median = 0.0
    medians["gender"] = float(int(gender_median >= 0.5))
    centers["gender"] = 0.0
    scales["gender"] = 1.0
    return ClinicalStats(medians=medians, centers=centers, scales=scales)


def transform_clinical(
    row: Mapping[str, object], stats: ClinicalStats
) -> np.ndarray:
    values: List[float] = []
    missing: List[float] = []
    for column in CLINICAL_COLUMNS:
        raw = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        is_missing = bool(pd.isna(raw))
        value = stats.medians[column] if is_missing else float(raw)
        if column in CONTINUOUS_COLUMNS:
            value = (value - stats.centers[column]) / stats.scales[column]
        else:
            value = float(int(value >= 0.5))
        values.append(float(value))
        missing.append(float(is_missing))
    return np.asarray(values + missing, dtype=np.float32)


def pad_face_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    horizontal = side - width
    vertical = side - height
    border = (
        horizontal // 2,
        vertical // 2,
        horizontal - horizontal // 2,
        vertical - vertical // 2,
    )
    return ImageOps.expand(image, border=border, fill=IMAGENET_MEAN_RGB)


def build_image_transform(train: bool = False, image_size: int = 224):
    # Geometry is deliberately conservative so the full clinical image remains visible.
    operations: List[object] = [transforms.Resize((image_size, image_size))]
    if train:
        operations.append(
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.03, hue=0.01
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class TongueFaceClinicalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        clinical_stats: ClinicalStats,
        train: bool = False,
        image_size: int = 224,
        missing_condition: str = "complete",
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.clinical_stats = clinical_stats
        self.transform = build_image_transform(train=train, image_size=image_size)
        self.missing_condition = missing_condition
        valid_conditions = {
            "complete",
            "missing_tongue",
            "missing_face",
            "missing_clinical",
            "missing_both_images",
        }
        if missing_condition not in valid_conditions:
            raise ValueError(
                f"unknown missing condition {missing_condition}; "
                f"expected one of {sorted(valid_conditions)}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def _read_image(self, path: str, is_face: bool) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if is_face:
                image = pad_face_to_square(image)
            return self.transform(image)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        tongue = self._read_image(str(row["tongue_path"]), is_face=False)
        face = self._read_image(str(row["face_path"]), is_face=True)
        clinical = torch.from_numpy(transform_clinical(row, self.clinical_stats))
        labels = []
        label_mask = []
        for label in LABEL_COLUMNS:
            valid = int(row[f"{label}_mask"]) == 1
            labels.append(float(row[label]) if valid else 0.0)
            label_mask.append(float(valid))
        modality_mask = torch.ones(3, dtype=torch.float32)
        if self.missing_condition == "missing_tongue":
            modality_mask[0] = 0
        elif self.missing_condition == "missing_face":
            modality_mask[1] = 0
        elif self.missing_condition == "missing_clinical":
            modality_mask[2] = 0
        elif self.missing_condition == "missing_both_images":
            modality_mask[:2] = 0
        return {
            "tongue_image": tongue,
            "face_image": face,
            "clinical": clinical,
            "modality_mask": modality_mask,
            "targets": torch.tensor(labels, dtype=torch.float32),
            "target_mask": torch.tensor(label_mask, dtype=torch.float32),
            "id_register": str(row["id_register"]),
            "sample_id": str(row["sample_id"]),
            "visit_year": str(row["visit_year"]),
            "device_id": str(row["device_id"]),
        }


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "id_register": str,
            "sample_id": str,
            "visit_year": str,
            "device_id": str,
        },
    )
    required = {
        "tongue_path",
        "face_path",
        "split",
        *CLINICAL_COLUMNS,
        *LABEL_COLUMNS,
        *(f"{label}_mask" for label in LABEL_COLUMNS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    return frame


def compute_pos_weight(train_frame: pd.DataFrame) -> torch.Tensor:
    weights = []
    for label in LABEL_COLUMNS:
        valid = train_frame[train_frame[f"{label}_mask"] == 1]
        positives = float(valid[label].sum())
        negatives = float(len(valid) - positives)
        weights.append(negatives / max(positives, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_loaders(
    manifest_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int = 224,
    stats_path: Optional[Path] = None,
    missing_condition: str = "complete",
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, ClinicalStats, pd.DataFrame]:
    frame = load_manifest(manifest_path)
    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("train/val/test splits must all be non-empty")
    if stats_path is not None and stats_path.exists():
        stats = ClinicalStats.load(stats_path)
    else:
        stats = fit_clinical_stats(train_frame)
        if stats_path is not None:
            stats.save(stats_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        TongueFaceClinicalDataset(
            train_frame, stats, train=True, image_size=image_size
        ),
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        TongueFaceClinicalDataset(
            val_frame, stats, train=False, image_size=image_size
        ),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        TongueFaceClinicalDataset(
            test_frame,
            stats,
            train=False,
            image_size=image_size,
            missing_condition=missing_condition,
        ),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader, stats, frame


def make_smoke_manifest(
    frame: pd.DataFrame, max_patients: int = 256, seed: int = 42
) -> pd.DataFrame:
    """Select a deterministic patient subset while retaining every split and label."""
    rng = np.random.RandomState(seed)
    selected: List[pd.DataFrame] = []
    per_split = max(12, max_patients // 3)
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split]
        patients = part["id_register"].drop_duplicates().to_numpy()
        rng.shuffle(patients)
        chosen = set(patients[: min(per_split, len(patients))])
        selected.append(part[part["id_register"].isin(chosen)])
    return pd.concat(selected, ignore_index=True)

