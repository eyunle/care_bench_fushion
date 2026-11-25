# Tongue–Face–Clinical CareBench extension

This package adds a separate three-modality benchmark without changing the
original MIMIC-IV/MIMIC-CXR pipeline. The raw CSV and image directories are
only read.

## Reproduction environment

The reported experiments used Python 3.11.18, CUDA 12.8, PyTorch 2.11.0,
torchvision 0.26.0, and torchaudio 2.11.0.

```bash
conda create -n medtf python=3.11 -y
conda activate medtf
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_tfc.txt
```

The code also runs on CPU for tests and smoke checks, but the full benchmark
is intended for a CUDA GPU.

## Implemented methods

`latefusion`, `utde`, `daft`, `mmtm`, `aug`, `inforeg`, `healnet`,
`flexmoe`, `drfuse`, `umse`, `shaspec`, `m3care`, `medfuse`, and `smil`.

All methods share two independent ResNet-50 image encoders and a clinical MLP.
They jointly output three binary logits and use a per-label missing-target
mask. Missing-aware methods receive a three-element modality mask. Complete
modality methods receive zero embeddings when evaluated with a missing
modality, matching the CareBench controlled-missingness convention.

## 1. Build and audit the manifest

Expected private-data layout:

```text
data/
├── tongue_face_merged_all_1.csv
├── tongue_images/
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
└── face_images_low/
    ├── 2022_deviceA/
    ├── 2023_deviceA/
    └── 2024_deviceB/
```

The data itself is not included in the repository. The defaults above can be
overridden with `TFC_SOURCE_CSV`, `TFC_TONGUE_ROOT`, and `TFC_FACE_ROOT`, or
with command-line arguments:

```powershell
python -m tfc_benchmark.cli prepare `
  --source-csv "D:\private_data\tongue_face_merged_all_1.csv" `
  --tongue-root "D:\private_data\tongue_images" `
  --face-root "D:\private_data\face_images_low" `
  --output-dir artifacts/tfc
```

The command decodes every referenced image, writes an independent manifest,
an image audit CSV, and a JSON split report. Patients are assigned once to a
fixed 70/15/15 grouped multilabel split. The 2023 `device_B` directory is
intentionally unsupported because those images do not have matching clinical
rows.

The dataset used for the reported run had 6,912 visits from 4,254 patients.
The audited common cohort contained 6,377 visits with two decodable images and
at least one valid target. The split audit must report zero patient overlap.

## 2. Local smoke test

```powershell
python -m pytest tests/test_tfc_data.py tests/test_tfc_models.py tests/test_tfc_tuning.py
python -m tfc_benchmark.cli smoke --config configs/tfc_default.yaml
```

The smoke command uses a small patient subset, a tiny image encoder, one epoch,
and all 14 methods. It does not overwrite the full manifest.

## 3. Budgeted tuning and full training

```bash
python -m tfc_benchmark.cli tune \
  --config configs/tfc_default.yaml --model drfuse --trials 10 --seed 42

python -m tfc_benchmark.cli run-all --config configs/tfc_default.yaml
```

Run `tune` for the eight configured tunable methods before `run-all`. If
`scikit-optimize` is installed, tuning uses Gaussian-process GP-Hedge;
otherwise it records that it used the deterministic random fallback.

`run-all` trains six additional modality-subset LateFusion baselines and all
14 full three-modality methods using seeds 42, 123, and 1234. Together with
the full three-modality LateFusion run, this covers all seven non-empty
modality subsets without duplicate training. The complete run therefore has
80 tuning trials and 60 final training jobs.

For an end-to-end server run after exporting the three path variables:

```bash
bash scripts/run_tfc_server.sh configs/tfc_default.yaml
```

Tuning is restart-safe: a contiguous prefix of completed trials is detected
from its result files and skipped. Run `run-all` only after tuning finishes;
preserve completed output directories when resuming manually.

## 4. Evaluation outputs

Each run stores:

- the exact YAML configuration and clinical training statistics;
- the AUROC-selected checkpoint and training history;
- validation-fitted Platt calibration and Youden thresholds;
- complete and four missing-modality prediction CSVs;
- AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, balanced accuracy,
  raw/calibrated Brier scores, and patient-cluster bootstrap intervals.

Aggregate completed experiments with:

```bash
python -m tfc_benchmark.cli report \
  --output-root experiments_tfc --report-dir reports_tfc
```

## Server environment

If PyTorch is already installed, install only the pinned non-PyTorch packages:

```bash
python -m pip install -r requirements_tfc.txt
```

Set `TFC_SOURCE_CSV`, `TFC_TONGUE_ROOT`, and `TFC_FACE_ROOT` to server-side
read-only paths before invoking `scripts/run_tfc_server.sh`.

