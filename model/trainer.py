"""Training loop: float pre-training followed by int8 quantisation-aware fine-tuning."""

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .data import Corpus, iter_batches, random_batch
from .net import CharMLP


@dataclass
class TrainConfig:
    # Optimiser defaults come from the best Optuna trial (post-QAT val loss 1.560
    # on the 1-epoch search budget, tanh activation); see model/tune.py.
    epochs: float = 8.0  # float phase, in passes over the training split
    qat_epochs: float = 2.0  # fake-quant fine-tuning phase
    batch_size: int = 256
    lr: float = 1.77e-2
    qat_lr: float = 1.3e-3
    min_lr_frac: float = 0.1  # cosine decay floor, as a fraction of the peak
    warmup_steps: int = 200
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    evals_per_phase: int = 10
    calib_examples: int = 65536
    seed: int = 1337


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    # The model is tiny, so kernel launch overhead dominates: on an M-series Mac
    # MPS is ~5x slower than CPU at batch 256 (2.2 vs 0.5 ms/step). Pass
    # --device mps explicitly to use it anyway.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: CharMLP, corpus: Corpus, split: str = "val", batch_size: int = 65536, limit: int | None = None):
    """Mean cross-entropy (nats), bits per character and top-1 accuracy."""
    was_training = model.training
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in iter_batches(corpus, split, batch_size, limit):
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    model.train(was_training)
    loss = loss_sum / n
    return {"loss": loss, "bpc": loss / math.log(2), "acc": correct / n}


def _lr_at(step: int, total: int, peak: float, warmup: int, floor_frac: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _optimizer(model: CharMLP, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def train_phase(
    model: CharMLP,
    corpus: Corpus,
    cfg: TrainConfig,
    epochs: float,
    peak_lr: float,
    warmup: int,
    phase: str,
    generator: torch.Generator,
    log: Callable[[str], None] | None = print,
    on_eval: Callable[[int, float], None] | None = None,
    eval_offset: int = 0,
) -> list[dict]:
    steps = max(1, round(epochs * corpus.num_examples("train") / cfg.batch_size))
    eval_at = {max(1, round(steps * k / cfg.evals_per_phase)) for k in range(1, cfg.evals_per_phase + 1)}
    opt = _optimizer(model, peak_lr, cfg.weight_decay)
    model.train()
    # Running loss stays on the device so GPU/MPS steps don't sync every iteration.
    history, running, n_running, t0 = [], torch.zeros((), device=corpus.train.device), 0, time.time()
    for step in range(steps):
        lr = _lr_at(step, steps, peak_lr, warmup, cfg.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = random_batch(corpus, "train", cfg.batch_size, generator)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        running += loss.detach()
        n_running += 1

        if step + 1 in eval_at:
            val = evaluate(model, corpus, "val")
            train_loss = running.item() / n_running
            rec = {"phase": phase, "step": step + 1, "train_loss": train_loss, "val_loss": val["loss"], "lr": lr}
            history.append(rec)
            running.zero_()
            n_running = 0
            if log:
                log(
                    f"[{phase}] step {step + 1:>6}/{steps}  train {rec['train_loss']:.4f}  "
                    f"val {val['loss']:.4f} ({val['bpc']:.3f} bpc, acc {val['acc']:.3f})  "
                    f"lr {lr:.2e}  {time.time() - t0:.0f}s"
                )
            if on_eval:
                on_eval(eval_offset + len(history) - 1, val["loss"])
    return history


def fit(
    model: CharMLP,
    corpus: Corpus,
    cfg: TrainConfig,
    log: Callable[[str], None] | None = print,
    on_eval: Callable[[int, float], None] | None = None,
) -> dict:
    """Float training then QAT. Leaves `model` in fake-quant mode and returns metrics
    plus the pre-QAT float weights (`float_state`)."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    history = train_phase(model, corpus, cfg, cfg.epochs, cfg.lr, cfg.warmup_steps, "float", gen, log, on_eval)
    float_val = evaluate(model, corpus, "val")
    float_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    calib_x, _ = random_batch(corpus, "train", cfg.calib_examples, gen)
    model.enable_quant(calib_x)
    if log:
        ptq = evaluate(model, corpus, "val")
        log(f"[qat] post-training-quant val {ptq['loss']:.4f} before fine-tuning (float {float_val['loss']:.4f})")
    if cfg.qat_epochs > 0:
        history += train_phase(
            model, corpus, cfg, cfg.qat_epochs, cfg.qat_lr, 0, "qat", gen, log, on_eval, eval_offset=len(history)
        )
    qat_val = evaluate(model, corpus, "val")
    return {"float_val": float_val, "qat_val": qat_val, "history": history, "float_state": float_state}
