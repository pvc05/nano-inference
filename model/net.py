"""Bengio-style character MLP (makemore part 2) with int8 fake-quant QAT.

    4 context tokens -> embedding (64 x 16) -> concat (64) -> fc1 (64 -> 128)
    -> tanh | relu -> fc2 (128 -> 64) -> logits over the 64-symbol vocabulary
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vocab
from .quant import INT8_MAX, fake_quant, fake_quant_bias, fake_quant_weight

# Pre-activations beyond this map to +-127 after tanh and int8 rounding, so
# there is no point spending int8 codes on them.
TANH_SATURATION = math.atanh(1 - 0.5 / INT8_MAX)
ACTIVATIONS = ("tanh", "relu")


@dataclass
class ModelConfig:
    vocab_size: int = vocab.VOCAB_SIZE
    context: int = 4
    emb_dim: int = 16
    hidden: int = 128
    activation: str = "tanh"

    @property
    def in_dim(self) -> int:
        return self.context * self.emb_dim


class CharMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.vocab_size)
        # Running range of the fc1 output used to pick its int8 scale:
        # max |pre-activation| for tanh, max post-ReLU value for relu.
        self.register_buffer("act_range", torch.tensor(0.0))
        self.observer_momentum = 0.99
        self.quantized = False
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=1.0)
        nn.init.normal_(self.fc1.weight, std=1.0 / math.sqrt(self.cfg.in_dim))
        nn.init.zeros_(self.fc1.bias)
        # Small output layer so the initial prediction is close to uniform.
        nn.init.normal_(self.fc2.weight, std=0.1 / math.sqrt(self.cfg.hidden))
        nn.init.zeros_(self.fc2.bias)

    # --- quantisation state ----------------------------------------------

    def act_scale(self) -> torch.Tensor:
        """Scale of the int8 tensor produced by requantising the fc1 accumulator."""
        r = self.act_range
        if self.cfg.activation == "tanh":
            r = r.clamp(max=TANH_SATURATION)
        return (r / INT8_MAX).clamp_min(1e-8)

    def hidden_scale(self) -> torch.Tensor:
        if self.cfg.activation == "tanh":
            return self.act_range.new_tensor(1.0 / INT8_MAX)
        return self.act_scale()

    def _range_of(self, pre: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "tanh":
            return pre.detach().abs().max()
        return pre.detach().max().clamp_min(0)

    @torch.no_grad()
    def calibrate(self, contexts: torch.Tensor):
        """Initialise the activation range from float pre-activations."""
        self.act_range.copy_(self._range_of(self.fc1(self.emb(contexts).flatten(1))))

    def enable_quant(self, calib_contexts: torch.Tensor | None = None):
        if calib_contexts is not None:
            self.calibrate(calib_contexts)
        self.quantized = True

    # --- forward -----------------------------------------------------------

    def _act(self, x):
        return torch.tanh(x) if self.cfg.activation == "tanh" else F.relu(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return self._forward_fake_quant(idx)
        x = self.emb(idx).flatten(1)
        return self.fc2(self._act(self.fc1(x)))

    def _forward_fake_quant(self, idx: torch.Tensor) -> torch.Tensor:
        """Float simulation of the integer pipeline in int8_model.Int8Model."""
        emb_w, s_e = fake_quant_weight(self.emb.weight)
        x = F.embedding(idx, emb_w).flatten(1)
        w1, s_w1 = fake_quant_weight(self.fc1.weight)
        pre = F.linear(x, w1, fake_quant_bias(self.fc1.bias, s_e * s_w1))

        if self.training:
            m = self.observer_momentum
            self.act_range.mul_(m).add_((1 - m) * self._range_of(pre))
        s_a = self.act_scale()
        if self.cfg.activation == "tanh":
            pre = fake_quant(pre, s_a, -INT8_MAX - 1, INT8_MAX)  # LUT address
            h = fake_quant(torch.tanh(pre), 1.0 / INT8_MAX, -INT8_MAX, INT8_MAX)  # LUT data
        else:
            h = fake_quant(F.relu(pre), s_a, 0, INT8_MAX)
        s_h = self.hidden_scale()

        w2, s_w2 = fake_quant_weight(self.fc2.weight)
        return F.linear(h, w2, fake_quant_bias(self.fc2.bias, s_h * s_w2))

    def num_weights(self) -> dict[str, int]:
        return {n: p.numel() for n, p in self.named_parameters()}


def save_checkpoint(path: str | Path, model: CharMLP, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.cfg), "state_dict": model.state_dict(), "quantized": model.quantized, **extra},
        path,
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[CharMLP, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = CharMLP(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.quantized = ckpt.get("quantized", False)
    model.eval()
    return model, ckpt
