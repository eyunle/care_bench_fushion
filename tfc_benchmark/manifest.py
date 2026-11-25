from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .constants import (
    CLINICAL_COLUMNS,
    DEVICES,
    LABEL_COLUMNS,
    SPLIT_FRACTIONS,
    default_face_root,
    default_source_csv,
    default_tongue_root,
)


def _valid_binary(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text in {"0", "1", "0.0", "1.0"}


def _as_binary(value: object) -> float:
    if not _valid_binary(value):
        return math.nan
    return float(int(float(str(value).strip())))


def resolve_paths(
    row: Mapping[str, object], tongue_root: Path, face_root: Path
) -> Tuple[Path, Path]:
    year = str(row["visit_year"]).strip()
    device = str(row["device_id"]).strip()
    key = (year, device)
    if key not in DEVICES:
        raise ValueError(f"unsupported year/device mapping: {year}/{device}")
    tongue = tongue_root / year / str(row["image_path"]).strip()
    face = face_root / DEVICES[key] / str(row["image_f_path"]).strip()
    return tongue, face


def _decode_image(path: Path) -> Tuple[bool, str, int, int]:
    if not path.is_file():
        return False, "missing", 0, 0
    try:
        with Image.open(path) as image:
            image.load()
            return True, "", int(image.width), int(image.height)
    except Exception as exc:  # audit must preserve the concrete reason
        return False, f"{type(exc).__name__}: {exc}", 0, 0


def _patient_vectors(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build group-level balancing vectors without using test-derived statistics."""
    vectors: Dict[str, np.ndarray] = {}
    for patient_id, group in frame.groupby("id_register", sort=False):
        values: List[float] = [float(len(group))]
        for year in ("2022", "2023", "2024"):
            values.append(float((group["visit_year"].astype(str) == year).sum()))
        for label in LABEL_COLUMNS:
            mask = group[f"{label}_mask"].to_numpy(dtype=bool)
            y = group[label].to_numpy(dtype=float)
            values.extend([float(mask.sum()), float(np.nansum(y[mask]))])
        vectors[str(patient_id)] = np.asarray(values, dtype=np.float64)
    return vectors


def grouped_multilabel_split(
    frame: pd.DataFrame,
    fractions: Mapping[str, float] = SPLIT_FRACTIONS,
    seed: int = 42,
    candidate_splits: int = 3000,
) -> Dict[str, str]:
    """
    Deterministic patient-level candidate search.

    Candidate assignments use exact patient-count fractions. The selected assignment
    minimizes deviations in visit fractions, year composition, label availability and
    label prevalence. This is more stable than row-wise stratification when patients have
    repeated visits and labels can change between years.
    """
    vectors = _patient_vectors(frame)
    split_names = tuple(fractions)
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("split fractions must sum to 1")
    patient_ids = np.asarray(list(vectors), dtype=object)
    matrix = np.stack([vectors[str(patient_id)] for patient_id in patient_ids])
    total = matrix.sum(axis=0)
    global_year_share = total[1:4] / max(total[0], 1.0)
    global_valid_share = np.asarray(
        [total[4] / total[0], total[6] / total[0], total[8] / total[0]]
    )
    global_prevalence = np.asarray(
        [
            total[5] / max(total[4], 1.0),
            total[7] / max(total[6], 1.0),
            total[9] / max(total[8], 1.0),
        ]
    )
    n_patients = len(patient_ids)
    boundaries = []
    used = 0
    for index, name in enumerate(split_names):
        if index == len(split_names) - 1:
            count = n_patients - used
        else:
            count = int(round(n_patients * fractions[name]))
        boundaries.append((used, used + count))
        used += count

    def assignment_score(permutation: np.ndarray) -> float:
        score = 0.0
        for name, (start, end) in zip(split_names, boundaries):
            split_total = matrix[permutation[start:end]].sum(axis=0)
            row_fraction = split_total[0] / max(total[0], 1.0)
            score += 8.0 * (row_fraction - fractions[name]) ** 2
            split_year_share = split_total[1:4] / max(split_total[0], 1.0)
            split_valid_share = np.asarray(
                [
                    split_total[4] / max(split_total[0], 1.0),
                    split_total[6] / max(split_total[0], 1.0),
                    split_total[8] / max(split_total[0], 1.0),
                ]
            )
            split_prevalence = np.asarray(
                [
                    split_total[5] / max(split_total[4], 1.0),
                    split_total[7] / max(split_total[6], 1.0),
                    split_total[9] / max(split_total[8], 1.0),
                ]
            )
            score += float(np.mean((split_year_share - global_year_share) ** 2))
            score += float(
                2.0 * np.mean((split_valid_share - global_valid_share) ** 2)
            )
            score += float(
                8.0 * np.mean((split_prevalence - global_prevalence) ** 2)
            )
        return score

    rng = np.random.RandomState(seed)
    best_permutation = np.arange(n_patients)
    best_score = float("inf")
    for _ in range(max(candidate_splits, 1)):
        permutation = rng.permutation(n_patients)
        score = assignment_score(permutation)
        if score < best_score:
            best_score = score
            best_permutation = permutation.copy()

    assignments: Dict[str, str] = {}
    for name, (start, end) in zip(split_names, boundaries):
        for index in best_permutation[start:end]:
            assignments[str(patient_ids[index])] = name
    return assignments


def _split_summary(frame: pd.DataFrame) -> Dict[str, object]:
    output: Dict[str, object] = {}
    patient_sets: Dict[str, set] = {}
    for split, part in frame.groupby("split", sort=False):
        patient_sets[str(split)] = set(part["id_register"].astype(str))
        item: MutableMapping[str, object] = {
            "rows": int(len(part)),
            "patients": int(part["id_register"].nunique()),
            "years": {
                str(k): int(v)
                for k, v in part["visit_year"].astype(str).value_counts().sort_index().items()
            },
            "labels": {},
        }
        for label in LABEL_COLUMNS:
            valid = part[part[f"{label}_mask"] == 1]
            item["labels"][label] = {
                "valid": int(len(valid)),
                "positive": int(valid[label].sum()),
                "negative": int(len(valid) - valid[label].sum()),
            }
        output[str(split)] = item
    intersections = {}
    names = sorted(patient_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            intersections[f"{left}__{right}"] = len(
                patient_sets[left] & patient_sets[right]
            )
    output["patient_intersections"] = intersections
    return output


def build_manifest(
    source_csv: Path,
    tongue_root: Path,
    face_root: Path,
    output_dir: Path,
    split_seed: int = 42,
    decode_images: bool = True,
) -> Dict[str, Path]:
    source_csv = source_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source_csv, dtype={"id_register": str, "visit_year": str})
    required = {
        "id_register",
        "sample_id",
        "visit_year",
        "device_id",
        "image_path",
        "image_f_path",
        *CLINICAL_COLUMNS,
        *LABEL_COLUMNS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"source CSV missing columns: {missing}")

    audit_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    for row_number, row in raw.iterrows():
        base = {
            "source_row": int(row_number + 2),
            "id_register": str(row["id_register"]),
            "sample_id": str(row["sample_id"]),
            "visit_year": str(row["visit_year"]),
            "device_id": str(row["device_id"]),
        }
        try:
            tongue_path, face_path = resolve_paths(row, tongue_root, face_root)
            if decode_images:
                tongue_ok, tongue_error, tongue_w, tongue_h = _decode_image(tongue_path)
                face_ok, face_error, face_w, face_h = _decode_image(face_path)
            else:
                tongue_ok, face_ok = tongue_path.is_file(), face_path.is_file()
                tongue_error = "" if tongue_ok else "missing"
                face_error = "" if face_ok else "missing"
                tongue_w = tongue_h = face_w = face_h = 0
        except Exception as exc:
            tongue_path = Path("")
            face_path = Path("")
            tongue_ok = face_ok = False
            tongue_error = face_error = f"{type(exc).__name__}: {exc}"
            tongue_w = tongue_h = face_w = face_h = 0

        labels = {label: _as_binary(row[label]) for label in LABEL_COLUMNS}
        masks = {f"{label}_mask": int(_valid_binary(row[label])) for label in LABEL_COLUMNS}
        any_label = any(masks.values())
        eligible = bool(tongue_ok and face_ok and any_label)
        reasons = []
        if not tongue_ok:
            reasons.append("tongue_invalid")
        if not face_ok:
            reasons.append("face_invalid")
        if not any_label:
            reasons.append("no_valid_label")
        audit_rows.append(
            {
                **base,
                "tongue_path": str(tongue_path),
                "face_path": str(face_path),
                "tongue_ok": int(tongue_ok),
                "face_ok": int(face_ok),
                "tongue_error": tongue_error,
                "face_error": face_error,
                "tongue_width": tongue_w,
                "tongue_height": tongue_h,
                "face_width": face_w,
                "face_height": face_h,
                "any_valid_label": int(any_label),
                "eligible": int(eligible),
                "audit_reason": ";".join(reasons),
            }
        )
        if not eligible:
            continue
        clinical = {
            column: pd.to_numeric(row[column], errors="coerce")
            for column in CLINICAL_COLUMNS
        }
        manifest_rows.append(
            {
                **base,
                "tongue_path": str(tongue_path.resolve()),
                "face_path": str(face_path.resolve()),
                **clinical,
                **labels,
                **masks,
            }
        )

    audit = pd.DataFrame(audit_rows)
    manifest = pd.DataFrame(manifest_rows)
    assignments = grouped_multilabel_split(manifest, seed=split_seed)
    manifest["split"] = manifest["id_register"].map(assignments)
    manifest = manifest.sort_values(
        ["split", "id_register", "visit_year", "sample_id"], kind="stable"
    ).reset_index(drop=True)

    manifest_path = output_dir / "tfc_manifest.csv"
    audit_path = output_dir / "tfc_image_audit.csv"
    summary_path = output_dir / "tfc_manifest_summary.json"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary = {
        "source_csv": str(source_csv),
        "source_rows": int(len(raw)),
        "source_patients": int(raw["id_register"].nunique()),
        "cross_year_patients": int(
            (raw.groupby("id_register")["visit_year"].nunique() > 1).sum()
        ),
        "tongue_decodable": int(audit["tongue_ok"].sum()),
        "face_decodable": int(audit["face_ok"].sum()),
        "both_decodable": int((audit["tongue_ok"] & audit["face_ok"]).sum()),
        "eligible_rows": int(len(manifest)),
        "split_seed": int(split_seed),
        "splits": _split_summary(manifest),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "manifest": manifest_path,
        "audit": audit_path,
        "summary": summary_path,
    }


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-csv", type=Path, default=default_source_csv())
    parser.add_argument("--tongue-root", type=Path, default=default_tongue_root())
    parser.add_argument("--face-root", type=Path, default=default_face_root())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tfc"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Only check file existence. Full runs should leave decoding enabled.",
    )


def command_prepare(args: argparse.Namespace) -> None:
    paths = build_manifest(
        source_csv=args.source_csv,
        tongue_root=args.tongue_root,
        face_root=args.face_root,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        decode_images=not args.skip_decode,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")

