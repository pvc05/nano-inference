"""Integer-only reference implementation of the quantised CharMLP.

This is the golden model for the FPGA: every step is integer arithmetic that
maps directly onto hardware (table lookup, int8 x int8 MACs into a wide
accumulator, multiply + arithmetic shift, saturate, 256-entry tanh LUT).
"""

from dataclasses import dataclass, field

import numpy as np

from .net import CharMLP, ModelConfig
from .quant import (
    INT8_MAX,
    fixed_point_multiplier,
    quantize_bias,
    quantize_weight,
    requantize,
    signed_bits,
    weight_scale,
)


@dataclass
class Int8Model:
    config: ModelConfig
    emb: np.ndarray  # int8 (vocab, emb_dim)
    w1: np.ndarray  # int8 (hidden, context * emb_dim), input index = position * emb_dim + d
    b1: np.ndarray  # int32 (hidden,), scale emb * w1
    w2: np.ndarray  # int8 (vocab, hidden)
    b2: np.ndarray  # int32 (vocab,), scale hidden * w2
    req1_mult: int  # fc1 accumulator -> int8 activation: (acc * mult + round) >> shift
    req1_shift: int
    act_lut: np.ndarray | None  # int8 (256,) tanh table addressed by the pre-activation byte
    scales: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_float(cls, model: CharMLP, mult_bits: int = 16) -> "Int8Model":
        """Freeze a (QAT-trained) CharMLP into integer tensors and requant constants."""
        cfg = model.cfg
        s_e = weight_scale(model.emb.weight).item()
        s_w1 = weight_scale(model.fc1.weight).item()
        s_w2 = weight_scale(model.fc2.weight).item()
        s_act = model.act_scale().item()
        s_h = model.hidden_scale().item()

        m1, shift1 = fixed_point_multiplier(s_e * s_w1 / s_act, mult_bits)
        lut = None
        if cfg.activation == "tanh":
            addr = np.arange(256)
            pre = np.where(addr < 128, addr, addr - 256)  # two's complement byte
            lut = np.round(INT8_MAX * np.tanh(pre * s_act)).astype(np.int8)

        return cls(
            config=cfg,
            emb=quantize_weight(model.emb.weight, s_e),
            w1=quantize_weight(model.fc1.weight, s_w1),
            b1=quantize_bias(model.fc1.bias, s_e * s_w1),
            w2=quantize_weight(model.fc2.weight, s_w2),
            b2=quantize_bias(model.fc2.bias, s_h * s_w2),
            req1_mult=m1,
            req1_shift=shift1,
            act_lut=lut,
            scales={
                "emb": s_e,
                "fc1_weight": s_w1,
                "act": s_act,
                "hidden": s_h,
                "fc2_weight": s_w2,
                "logits": s_h * s_w2,
                "mult_bits": mult_bits,
            },
        )

    # --- inference ---------------------------------------------------------

    @staticmethod
    def _int_matmul(a: np.ndarray, b_t: np.ndarray, bias: np.ndarray) -> np.ndarray:
        # Exact: every partial sum is an integer far below 2**53, so float64
        # BLAS gives the same result as an int32 MAC array, just faster.
        return np.rint(a.astype(np.float64) @ b_t.T.astype(np.float64)).astype(np.int64) + bias

    def hidden(self, contexts: np.ndarray) -> np.ndarray:
        """int8 hidden activations (N, hidden) for token contexts (N, context)."""
        ctx = np.asarray(contexts, dtype=np.int64)
        x = self.emb[ctx].reshape(len(ctx), -1)
        acc1 = self._int_matmul(x, self.w1, self.b1)
        if self.config.activation == "tanh":
            pre = requantize(acc1, self.req1_mult, self.req1_shift, -INT8_MAX - 1, INT8_MAX)
            return self.act_lut[pre & 0xFF]
        return requantize(acc1, self.req1_mult, self.req1_shift, 0, INT8_MAX).astype(np.int8)

    def int_logits(self, contexts: np.ndarray) -> np.ndarray:
        """Raw int32 fc2 accumulators (N, vocab). argmax of these is the prediction."""
        acc2 = self._int_matmul(self.hidden(contexts), self.w2, self.b2)
        return acc2.astype(np.int32)

    def logits(self, contexts: np.ndarray) -> np.ndarray:
        return self.int_logits(contexts) * self.scales["logits"]

    # --- hardware sizing -------------------------------------------------

    def accumulator_bits(self) -> dict[str, int]:
        """Worst-case two's complement widths for any input token sequence."""
        x_max = int(np.abs(self.emb.astype(np.int64)).max())
        acc1 = int((np.abs(self.w1.astype(np.int64)).sum(1) * x_max + np.abs(self.b1.astype(np.int64))).max())
        acc2 = int((np.abs(self.w2.astype(np.int64)).sum(1) * INT8_MAX + np.abs(self.b2.astype(np.int64))).max())
        return {
            "acc1": signed_bits(acc1),
            "acc2": signed_bits(acc2),
            "req1_product": signed_bits(acc1 * self.req1_mult + (1 << (self.req1_shift - 1))),
        }
