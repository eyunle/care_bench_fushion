from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .constants import TRAINING_SEEDS
from .data import load_manifest, make_smoke_manifest
from .engine import (
    ExperimentConfig,
    evaluate_checkpoint,
    train_experiment,
)
from .manifest import add_prepare_arguments, command_prepare
from .models import MODEL_NAMES
from .report import aggregate_results
from .serialization import json_safe
from .tuning import TUNABLE_MODELS, tune_model


BASELINE_MODALITIES = {
    "tongue": ["tongue"],
    "face": ["face"],
    "clinical": ["clinical"],
    "tongue_face": ["tongue", "face"],
    "tongue_clinical": ["tongue", "clinical"],
    "face_clinical": ["face", "clinical"],
}


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {
        "model": getattr(args, "model", None),
        "seed": getattr(args, "seed", None),
        "manifest": str(args.manifest) if getattr(args, "manifest", None) else None,
        "output_root": (
            str(args.output_root) if getattr(args, "output_root", None) else None
        ),
    }
    return ExperimentConfig.load(args.config, overrides=overrides)


def command_train(args: argparse.Namespace) -> None:
    config = _load_config(args)
    result = train_experiment(
        config,
        evaluate_after_training=not args.skip_evaluation,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    config = _load_config(args)
    result = evaluate_checkpoint(
        config,
        checkpoint_path=args.checkpoint,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


def command_tune(args: argparse.Namespace) -> None:
    config = _load_config(args)
    result = tune_model(config, trials=args.trials, seed=args.seed)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


def command_smoke(args: argparse.Namespace) -> None:
    base = _load_config(args)
    frame = load_manifest(Path(base.manifest))
    smoke = make_smoke_manifest(
        frame, max_patients=args.max_patients, seed=base.seed
    )
    smoke_dir = Path(base.output_root) / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_manifest = smoke_dir / "smoke_manifest.csv"
    smoke.to_csv(smoke_manifest, index=False, encoding="utf-8-sig")
    failures = {}
    for model_name in MODEL_NAMES:
        config = copy.deepcopy(base)
        config.manifest = str(smoke_manifest)
        config.model = model_name
        config.output_root = str(smoke_dir / "runs")
        config.image_backbone = "tiny"
        config.pretrained = False
        config.hidden_dim = args.hidden_dim
        config.image_size = args.image_size
        config.batch_size = args.batch_size
        config.num_workers = args.num_workers
        config.epochs = 1
        config.patience = 1
        config.bootstrap_samples = 0
        try:
            train_experiment(config, bootstrap_samples=0)
        except Exception as exc:
            failures[model_name] = f"{type(exc).__name__}: {exc}"
            print(f"SMOKE FAILED {model_name}: {failures[model_name]}")
    result_path = smoke_dir / "smoke_results.json"
    result_path.write_text(
        json.dumps(
            {
                "models": list(MODEL_NAMES),
                "failures": failures,
                "success": not failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} smoke tests failed; see {result_path}")
    print(f"All {len(MODEL_NAMES)} models passed. Results: {result_path}")


def command_run_all(args: argparse.Namespace) -> None:
    base = _load_config(args)
    seeds = args.seeds or list(TRAINING_SEEDS)
    # Six additional subsets; full three-modality LateFusion is included in MODEL_NAMES,
    # yielding all seven non-empty modality subsets without duplicate training.
    if not args.skip_baselines:
        for name, modalities in BASELINE_MODALITIES.items():
            for seed in seeds:
                config = copy.deepcopy(base)
                config.model = "latefusion"
                config.seed = seed
                config.enabled_modalities = modalities
                config.output_root = str(Path(base.output_root) / "baselines" / name)
                train_experiment(config)
    for model_name in MODEL_NAMES:
        for seed in seeds:
            config = copy.deepcopy(base)
            config.model = model_name
            config.seed = seed
            config.enabled_modalities = ["tongue", "face", "clinical"]
            config.output_root = str(Path(base.output_root) / "fusion_methods")
            tuned_path = (
                Path(base.output_root)
                / "tuning"
                / model_name
                / "best_config.yaml"
            )
            if tuned_path.exists():
                tuned = ExperimentConfig.load(tuned_path)
                config.model_params = tuned.model_params
            train_experiment(config)


def command_report(args: argparse.Namespace) -> None:
    paths = aggregate_results(
        args.output_root,
        args.report_dir,
        n_permutations=args.permutations,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def _add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tfc_default.yaml")
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tongue-face-clinical CareBench extension"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="audit data and build manifest")
    add_prepare_arguments(prepare)
    prepare.set_defaults(function=command_prepare)

    train = subparsers.add_parser("train", help="train one model and seed")
    _add_common_config_arguments(train)
    train.add_argument("--skip-evaluation", action="store_true")
    train.add_argument("--bootstrap-samples", type=int)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint")
    _add_common_config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--bootstrap-samples", type=int)
    evaluate.set_defaults(function=command_evaluate)

    tune = subparsers.add_parser("tune", help="budgeted model-specific search")
    _add_common_config_arguments(tune)
    tune.add_argument("--trials", type=int, default=10)
    tune.set_defaults(function=command_tune)

    smoke = subparsers.add_parser("smoke", help="one-epoch smoke test for all models")
    _add_common_config_arguments(smoke)
    smoke.add_argument("--max-patients", type=int, default=256)
    smoke.add_argument("--hidden-dim", type=int, default=32)
    smoke.add_argument("--image-size", type=int, default=64)
    smoke.add_argument("--batch-size", type=int, default=4)
    smoke.add_argument("--num-workers", type=int, default=0)
    smoke.set_defaults(function=command_smoke)

    run_all = subparsers.add_parser("run-all", help="run baselines and all methods")
    _add_common_config_arguments(run_all)
    run_all.add_argument("--seeds", nargs="+", type=int)
    run_all.add_argument("--skip-baselines", action="store_true")
    run_all.set_defaults(function=command_run_all)

    report = subparsers.add_parser("report", help="aggregate completed runs")
    report.add_argument("--output-root", type=Path, default=Path("experiments_tfc"))
    report.add_argument("--report-dir", type=Path, default=Path("reports_tfc"))
    report.add_argument("--permutations", type=int, default=5000)
    report.set_defaults(function=command_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

