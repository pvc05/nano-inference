"""Int8 fake-quantisation primitives (used during QAT) and the fixed-point
requantisation scheme shared with the integer reference and the FPGA.

Scheme (all symmetric, per-tensor, zero-point 0):
  real = scale * q,  q in [-127, 127] for weights, int32 for biases.
A layer's int32 accumulator has scale s_in * s_w. Rescaling it to an int8
activation of scale s_out uses M = s_in * s_w / s_out ~= m * 2**-shift, i.e.
  q_out = clamp((acc * m + (1 << (shift - 1))) >> shift, lo, hi)
with an arithmetic shift, so the hardware needs one multiply and one shift.
"""

import math

import numpy as np
import torch

INT8_MAX = 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest with a straight-through gradient."""
    return x + (torch.round(x) - x).detach()


def fake_quant(x: torch.Tensor, scale: torch.Tensor | float, qmin: int, qmax: int) -> torch.Tensor:
    """Quantise-dequantise: scale * clamp(round(x / scale), qmin, qmax)."""
    return torch.clamp(round_ste(x / scale), qmin, qmax) * scale


def weight_scale(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor scale mapping max|w| onto 127 (no gradient)."""
    return (w.detach().abs().max() / INT8_MAX).clamp_min(1e-12)


def fake_quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = weight_scale(w)
    return fake_quant(w, s, -INT8_MAX, INT8_MAX), s


def fake_quant_bias(b: torch.Tensor, acc_scale: torch.Tensor) -> torch.Tensor:
    return fake_quant(b, acc_scale, INT32_MIN, INT32_MAX)


def quantize_weight(w: torch.Tensor, scale: float) -> np.ndarray:
    q = torch.clamp(torch.round(w.detach().cpu().double() / scale), -INT8_MAX, INT8_MAX)
    return q.numpy().astype(np.int8)


def quantize_bias(b: torch.Tensor, acc_scale: float) -> np.ndarray:
    q = torch.clamp(torch.round(b.detach().cpu().double() / acc_scale), INT32_MIN, INT32_MAX)
    return q.numpy().astype(np.int32)


def fixed_point_multiplier(real: float, bits: int = 16) -> tuple[int, int]:
    """Approximate a positive real factor as m * 2**-shift with m in [2**(bits-1), 2**bits)."""
    if real <= 0:
        raise ValueError("requantisation factor must be positive")
    exp = math.floor(math.log2(real))
    shift = bits - 1 - exp
    m = round(real * 2.0**shift)
    if m == 2**bits:  # rounding overflowed into the next power of two
        m //= 2
        shift -= 1
    if shift < 1:
        raise ValueError(f"requantisation factor {real} too large for a right shift")
    return m, shift


def requantize(acc: np.ndarray, m: int, shift: int, qmin: int, qmax: int) -> np.ndarray:
    """Integer-only rescale: round-half-up of acc * m / 2**shift, then saturate."""
    acc = acc.astype(np.int64)
    q = (acc * m + (1 << (shift - 1))) >> shift
    return np.clip(q, qmin, qmax)


def signed_bits(max_abs: int) -> int:
    """Width of a two's complement register that holds [-max_abs, max_abs]."""
    return int(max_abs).bit_length() + 1
