from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

from .engine import ExperimentConfig, train_experiment
from .serialization import json_safe


TUNABLE_MODELS = (
    "drfuse",
    "flexmoe",
    "healnet",
    "m3care",
    "shaspec",
    "smil",
    "aug",
    "inforeg",
)

# kind, low/choices, high. Log-uniform is represented by "log_real".
SEARCH_SPACES = {
    "drfuse": {"align_weight": ("log_real", 0.01, 2.0)},
    "flexmoe": {
        "num_experts": ("categorical", [4, 8, 16]),
        "top_k": ("categorical", [2, 4]),
    },
    "healnet": {
        "depth": ("categorical", [1, 2, 3]),
        "latent_tokens": ("categorical", [8, 16, 32]),
    },
    "m3care": {"stability_weight": ("log_real", 0.001, 2.0)},
    "shaspec": {
        "shared_weight": ("log_real", 0.01, 0.1),
        "domain_weight": ("log_real", 0.005, 0.2),
    },
    "smil": {"reconstruction_weight": ("real", 0.05, 0.2)},
    "aug": {"merge_alpha": ("real", 0.2, 0.8)},
    "inforeg": {"gamma": ("real", 0.05, 0.5)},
}


def _native_params(names: Sequence[str], values: Sequence[object]) -> Dict[str, object]:
    """Convert optimizer NumPy scalars into YAML-safe Python values."""
    return {
        name: json_safe(value)
        for name, value in zip(names, values)
    }


def _load_completed_trials(
    root: Path,
    model: str,
    seed: int,
    trials: int,
) -> List[Dict[str, object]]:
    """Load the contiguous, fully completed prefix of tuning trials."""
    records: List[Dict[str, object]] = []
    parameter_names = tuple(SEARCH_SPACES[model])
    for trial_index in range(trials):
        run_dir = root / f"trial_{trial_index:02d}" / model / f"seed_{seed}"
        summary_path = run_dir / "run_summary.json"
        config_path = run_dir / "config.yaml"
        history_path = run_dir / "history.csv"
        if not (
            summary_path.is_file()
            and config_path.is_file()
            and history_path.is_file()
        ):
            break
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            model_params = payload.get("model_params") or {}
            params = {name: model_params[name] for name in parameter_names}
            score = float(summary["best_val_macro_auroc"])
            epochs_completed = int(summary["epochs_completed"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            break
        if not np.isfinite(score) or epochs_completed <= 0:
            break
        records.append(
            {"trial": trial_index, "score": score, "params": params}
        )
    return records


def _sample_random(model: str, rng: np.random.RandomState) -> Dict[str, object]:
    sampled: Dict[str, object] = {}
    for name, specification in SEARCH_SPACES[model].items():
        kind = specification[0]
        if kind == "categorical":
            choices = specification[1]
            sampled[name] = choices[int(rng.randint(0, len(choices)))]
        elif kind == "real":
            sampled[name] = float(rng.uniform(specification[1], specification[2]))
        elif kind == "log_real":
            sampled[name] = float(
                np.exp(
                    rng.uniform(
                        np.log(specification[1]), np.log(specification[2])
                    )
                )
            )
        else:
            raise ValueError(f"unsupported search-space type: {kind}")
    return sampled


def tune_model(
    base_config: ExperimentConfig,
    trials: int = 10,
    seed: int = 42,
) -> Dict[str, object]:
    if base_config.model not in TUNABLE_MODELS:
        raise ValueError(
            f"{base_config.model} has no benchmark search space; "
            f"expected one of {TUNABLE_MODELS}"
        )
    rng = np.random.RandomState(seed)
    root = Path(base_config.output_root) / "tuning" / base_config.model
    root.mkdir(parents=True, exist_ok=True)
    records = _load_completed_trials(
        root, base_config.model, seed=seed, trials=trials
    )
    resumed_trials = len(records)
    best_record = max(records, key=lambda record: record["score"], default=None)
    best_score = (
        float(best_record["score"]) if best_record is not None else -float("inf")
    )
    best_params: Dict[str, object] = (
        dict(best_record["params"]) if best_record is not None else {}
    )
    if resumed_trials:
        print(
            f"[{base_config.model}] resuming after {resumed_trials} "
            "completed tuning trials"
        )

    # Use scikit-optimize GP-Hedge when available; retain a deterministic random
    # fallback so the benchmark remains runnable on minimal server environments.
    try:
        from skopt import Optimizer
        from skopt.space import Categorical, Real

        dimensions = []
        names = []
        for name, specification in SEARCH_SPACES[base_config.model].items():
            names.append(name)
            if specification[0] == "categorical":
                dimensions.append(Categorical(specification[1], name=name))
            else:
                prior = "log-uniform" if specification[0] == "log_real" else "uniform"
                dimensions.append(
                    Real(
                        specification[1],
                        specification[2],
                        prior=prior,
                        name=name,
                    )
                )

        optimizer = Optimizer(
            dimensions,
            n_initial_points=min(3, trials),
            acq_func="gp_hedge",
            random_state=seed,
        )
        for record in records:
            # Advance the optimizer's RNG in the same order as the original run,
            # then restore the exact observed point and objective value.
            optimizer.ask()
            values = [record["params"][name] for name in names]
            optimizer.tell(values, -float(record["score"]))

        for trial_index in range(resumed_trials, trials):
            values = optimizer.ask()
            params = _native_params(names, values)
            config = copy.deepcopy(base_config)
            config.output_root = str(root / f"trial_{trial_index:02d}")
            config.seed = seed
            config.model_params = {**config.model_params, **params}
            result = train_experiment(
                config, evaluate_after_training=False, bootstrap_samples=0
            )
            score = float(result["best_val_macro_auroc"])
            records.append({"trial": trial_index, "score": score, "params": params})
            optimizer.tell(values, -score)
            if score > best_score:
                best_score = score
                best_params = params
        backend = "skopt_gp_hedge"
    except ImportError:
        backend = "deterministic_random_fallback"
        for record in records:
            _sample_random(base_config.model, rng)
        for trial_index in range(resumed_trials, trials):
            params = _sample_random(base_config.model, rng)
            config = copy.deepcopy(base_config)
            config.output_root = str(root / f"trial_{trial_index:02d}")
            config.seed = seed
            config.model_params = {**config.model_params, **params}
            result = train_experiment(
                config, evaluate_after_training=False, bootstrap_samples=0
            )
            score = float(result["best_val_macro_auroc"])
            records.append({"trial": trial_index, "score": score, "params": params})
            if score > best_score:
                best_score = score
                best_params = params

    output = {
        "model": base_config.model,
        "seed": seed,
        "trials": trials,
        "resumed_trials": resumed_trials,
        "backend": backend,
        "best_score": best_score,
        "best_params": best_params,
        "records": records,
    }
    (root / "tuning_results.json").write_text(
        json.dumps(json_safe(output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tuned_config = copy.deepcopy(base_config)
    tuned_config.model_params = {**tuned_config.model_params, **best_params}
    tuned_config.save(root / "best_config.yaml")
    return output

