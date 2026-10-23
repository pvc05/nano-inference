"""Interchangeable inference backends sharing one interface.

A backend maps token contexts (N, context) to logits (N, vocab). The float
PyTorch model and the int8 golden model live here; an FPGA backend for the
benchmarking stage only needs to implement the same `logits` method.
"""

import math
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from . import export, vocab
from .data import Corpus, iter_batches
from .int8_model import Int8Model
from .net import CharMLP, load_checkpoint


class Backend(Protocol):
    name: str
    context: int
    vocab_size: int

    def logits(self, contexts: np.ndarray) -> np.ndarray: ...


class TorchBackend:
    """Float (or fake-quant) PyTorch model on any device."""

    def __init__(self, model: CharMLP, device: str | torch.device = "cpu", name: str | None = None):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.context = model.cfg.context
        self.vocab_size = model.cfg.vocab_size
        self.name = name or ("torch-fakequant" if model.quantized else "torch-float")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device="cpu", quantized: bool | None = None) -> "TorchBackend":
        model, _ = load_checkpoint(path, map_location=device)
        if quantized is not None:
            model.quantized = quantized
        return cls(model, device)

    @torch.no_grad()
    def logits(self, contexts: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(contexts), dtype=torch.long, device=self.device)
        return self.model(x).float().cpu().numpy()


class Int8Backend:
    """Bit-exact integer reference of what the FPGA computes."""

    name = "int8-reference"

    def __init__(self, model: Int8Model):
        self.model = model
        self.context = model.config.context
        self.vocab_size = model.config.vocab_size

    @classmethod
    def from_dir(cls, weights_dir: str | Path) -> "Int8Backend":
        return cls(export.load(weights_dir))

    def int_logits(self, contexts: np.ndarray) -> np.ndarray:
        return self.model.int_logits(contexts)

    def logits(self, contexts: np.ndarray) -> np.ndarray:
        return self.model.logits(contexts)


def predict(backend: Backend, contexts: np.ndarray) -> np.ndarray:
    return backend.logits(contexts).argmax(1)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def evaluate(backend: Backend, corpus: Corpus, split: str = "val", batch_size: int = 65536) -> dict:
    """Cross-entropy (nats), bits per char and top-1 accuracy of any backend."""
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in iter_batches(corpus.to("cpu"), split, batch_size):
        y = y.numpy()
        logp = _log_softmax(backend.logits(x.numpy()).astype(np.float64))
        loss_sum -= logp[np.arange(len(y)), y].sum()
        correct += int((logp.argmax(1) == y).sum())
        n += len(y)
    loss = loss_sum / n
    return {"loss": loss, "bpc": loss / math.log(2), "acc": correct / n}


def generate(
    backend: Backend,
    prompt: str = "",
    length: int = 300,
    temperature: float = 1.0,
    seed: int | None = None,
) -> str:
    """Autoregressive sampling (temperature 0 = greedy argmax, as a simple FPGA would do)."""
    rng = np.random.default_rng(seed)
    prompt_ids = vocab.encode(prompt.lower())
    ctx = [vocab.NEWLINE] * backend.context + prompt_ids
    out = []
    for _ in range(length):
        z = backend.logits(np.array([ctx[-backend.context :]]))[0].astype(np.float64)
        if temperature <= 0:
            nxt = int(z.argmax())
        else:
            p = np.exp((z - z.max()) / temperature)
            nxt = int(rng.choice(len(p), p=p / p.sum()))
        ctx.append(nxt)
        out.append(nxt)
    return vocab.decode(prompt_ids + out)
