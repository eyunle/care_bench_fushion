from __future__ import annotations

import os
from pathlib import Path

LABEL_COLUMNS = ("影像学脂肪肝", "糖尿病前期", "MAFLD")
CLINICAL_COLUMNS = ("age", "HEIGHT", "WEIGHT", "gender", "waist")
CONTINUOUS_COLUMNS = ("age", "HEIGHT", "WEIGHT", "waist")
YEARS = ("2022", "2023", "2024")
DEVICES = {
    ("2022", "device_A"): "2022_deviceA",
    ("2023", "device_A"): "2023_deviceA",
    ("2024", "device_B"): "2024_deviceB",
}
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRAINING_SEEDS = (42, 123, 1234)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_RGB = tuple(round(v * 255) for v in IMAGENET_MEAN)


def default_source_csv() -> Path:
    return Path(
        os.environ.get(
            "TFC_SOURCE_CSV", "data/tongue_face_merged_all_1.csv"
        )
    )


def default_tongue_root() -> Path:
    return Path(os.environ.get("TFC_TONGUE_ROOT", "data/tongue_images"))


def default_face_root() -> Path:
    return Path(os.environ.get("TFC_FACE_ROOT", "data/face_images_low"))

