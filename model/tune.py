"""Bayesian hyperparameter search (Optuna TPE) over the training recipe.

The architecture is fixed by the FPGA budget; the search covers optimiser
settings and the activation, and scores each trial by its validation loss
*after* int8 QAT, since that is the model that ships.
"""

import json
import multiprocessing as mp
import os
from dataclasses import replace
from functools import partial
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
SPLITS = ("train", "val", "test")


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


def _make_study(storage: str | None, study_name: str, evals_per_phase: int) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=evals_per_phase // 2),
    )


def _sampler(seed: int, parallel: bool) -> optuna.samplers.BaseSampler:
    # constant_liar stops parallel workers from proposing the same point while
    # each other's trials are still running.
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=8, constant_liar=parallel)


def _optimize(
    study: optuna.Study,
    corpus: Corpus,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    n_trials: int | None,
    trial_epochs: float,
    trial_qat_epochs: float,
    timeout: float | None,
    seed: int,
    log: Callable[[str], None],
    callbacks: list = (),
):
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

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[report, *callbacks])


def _worker(n_jobs: int, splits: dict, context: int, device: str, sampler_seed: int, target: int, kwargs: dict):
    """One parallel search process: shares the study through its RDB storage and
    stops once the study holds `target` trials (counting ones still running)."""
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // n_jobs))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    corpus = Corpus(*(torch.from_numpy(splits[s]).long() for s in SPLITS), context).to(device)
    study = _make_study(kwargs.pop("storage"), kwargs.pop("study_name"), kwargs["train_cfg"].evals_per_phase)
    study.sampler = _sampler(sampler_seed, parallel=True)
    stop = optuna.study.MaxTrialsCallback(target, states=None)
    _optimize(study, corpus, n_trials=None, log=partial(print, flush=True), callbacks=[stop], **kwargs)


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
    n_jobs: int = 1,
    log: Callable[[str], None] = print,
) -> optuna.Study:
    """Minimise post-QAT val loss on a shortened schedule. Resumes `study_name` if it exists in `storage`.

    With n_jobs > 1 the trials run in that many worker processes sharing the
    study through `storage`, with the CPU threads split evenly between them.
    """
    if storage and storage.startswith("sqlite:///"):
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_jobs = max(1, min(n_jobs, n_trials))
    study = _make_study(storage, study_name, train_cfg.evals_per_phase)
    # Offset sampler seeds by the trials already in the study so a resumed
    # search doesn't re-propose the random points it started with last time.
    existing = len(study.trials)
    study.sampler = _sampler(seed + existing, parallel=n_jobs > 1)
    kwargs = dict(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        trial_epochs=trial_epochs,
        trial_qat_epochs=trial_qat_epochs,
        timeout=timeout,
        seed=seed,
    )
    if n_jobs == 1:
        _optimize(study, corpus, n_trials=n_trials, log=log, **kwargs)
        return study
    if storage is None:
        raise ValueError("parallel tuning needs RDB storage shared by the workers, e.g. sqlite:///runs/optuna.db")

    # Token ids fit in a byte, so each worker gets a ~6 MB copy of the corpus.
    splits = {s: corpus.split(s).to(torch.uint8).cpu().numpy() for s in SPLITS}
    kwargs.update(storage=storage, study_name=study_name)
    device, target = str(corpus.train.device), existing + n_trials
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_worker,
            args=(n_jobs, splits, corpus.context, device, seed + existing + i, target, kwargs),
            daemon=True,
        )
        for i in range(n_jobs)
    ]
    log(f"[tune] {n_jobs} worker processes, {max(1, (os.cpu_count() or 1) // n_jobs)} CPU threads each")
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    finally:
        # On Ctrl-C the workers get SIGINT too; give them time to mark their
        # running trials FAIL in the study before forcing them down.
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    failed = sum(p.exitcode != 0 for p in procs)
    if failed:
        raise RuntimeError(f"{failed} of {n_jobs} tuning workers exited with an error (see output above)")
    return optuna.load_study(study_name=study_name, storage=storage)


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
