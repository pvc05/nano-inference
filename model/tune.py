"""Bayesian hyperparameter search (Optuna TPE) over the training recipe.

The architecture is fixed by the FPGA budget; the search covers optimiser
settings and the activation, and scores each trial by its validation loss
*after* int8 QAT, since that is the model that ships.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import optuna
import torch

from .data import Corpus
from .net import CharMLP, ModelConfig
from .trainer import TrainConfig, fit

# Keys of the best-trial JSON that map onto ModelConfig / TrainConfig fields.
MODEL_KEYS = {"activation"}
TRAIN_KEYS = {"lr", "batch_size", "weight_decay", "qat_lr", "warmup_steps", "min_lr_frac"}


def suggest(trial: optuna.Trial) -> dict:
    lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)
    return {
        "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
        "lr": lr,
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "qat_lr": lr * trial.suggest_float("qat_lr_frac", 0.01, 0.5, log=True),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [0, 200, 1000]),
        "min_lr_frac": trial.suggest_float("min_lr_frac", 0.005, 0.2, log=True),
    }


def apply_hparams(hp: dict, model_cfg: ModelConfig, train_cfg: TrainConfig) -> tuple[ModelConfig, TrainConfig]:
    model_cfg = replace(model_cfg, **{k: v for k, v in hp.items() if k in MODEL_KEYS})
    train_cfg = replace(train_cfg, **{k: v for k, v in hp.items() if k in TRAIN_KEYS})
    return model_cfg, train_cfg


def run_study(
    corpus: Corpus,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    n_trials: int,
    trial_epochs: float = 1.0,
    trial_qat_epochs: float = 0.25,
    storage: str | None = None,
    study_name: str = "charmlp",
    timeout: float | None = None,
    seed: int = 1337,
    log: Callable[[str], None] = print,
) -> optuna.Study:
    """Minimise post-QAT val loss on a shortened schedule. Resumes `study_name` if it exists in `storage`."""
    if storage and storage.startswith("sqlite:///"):
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=train_cfg.evals_per_phase // 2),
    )

    def objective(trial: optuna.Trial) -> float:
        hp = suggest(trial)
        mcfg, tcfg = apply_hparams(hp, model_cfg, train_cfg)
        tcfg = replace(tcfg, epochs=trial_epochs, qat_epochs=trial_qat_epochs, seed=seed + trial.number)
        torch.manual_seed(tcfg.seed)
        model = CharMLP(mcfg).to(corpus.train.device)

        def on_eval(idx: int, val_loss: float):
            trial.report(val_loss, idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = fit(model, corpus, tcfg, log=None, on_eval=on_eval)
        trial.set_user_attr("float_val_loss", result["float_val"]["loss"])
        return result["qat_val"]["loss"]

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        value = f"{trial.value:.4f}" if trial.value is not None else trial.state.name.lower()
        done = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best = f"{min(done):.4f}" if done else "-"
        log(f"[tune] trial {trial.number:>3}: {value}  best {best}  {trial.params}")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[report])
    return study


def best_hparams(study: optuna.Study) -> dict:
    """Best trial's settings in apply_hparams form (qat_lr resolved to an absolute value)."""
    p = dict(study.best_params)
    p["qat_lr"] = p["lr"] * p.pop("qat_lr_frac")
    return p


def save_hparams(hp: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hp, indent=2) + "\n")


def load_hparams(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
