"""Hardware-style check of an exported model-weights/ directory.

Recomputes every golden test vector the way the FPGA datapath will: only the
.hex files and params.vh are read, memories are flat arrays indexed with the
documented address arithmetic, and all math is plain Python integers (no numpy,
no PyTorch, no other project code). It also asserts every accumulator fits the
widths declared in params.vh.

    python -m model.hw_check [model-weights]
"""

import re
import sys
from pathlib import Path


def read_params(path: Path) -> dict[str, int]:
    pattern = r"localparam\s+(?:\[\d+:0\]\s+)?(\w+)\s*=\s*(?:\d+'d)?(\d+)\s*;"
    return {k: int(v) for k, v in re.findall(pattern, path.read_text())}


def read_mem(path: Path, bits: int, signed: bool = True) -> list[int]:
    """$readmemh semantics: one hex word per line, two's complement if signed."""
    words = [int(w, 16) for w in path.read_text().split()]
    if signed:
        words = [w - (1 << bits) if w >= 1 << (bits - 1) else w for w in words]
    return words


def check(weights_dir: str | Path, verbose: bool = True) -> int:
    """Return the number of test vectors checked; raise AssertionError on any mismatch."""
    d = Path(weights_dir)
    P = read_params(d / "params.vh")
    C, E, IN, H, V = P["CONTEXT"], P["EMB_DIM"], P["IN_DIM"], P["HIDDEN"], P["VOCAB_SIZE"]
    mult, shift = P["REQ1_MULT"], P["REQ1_SHIFT"]
    acc1_lim, acc2_lim = 1 << (P["ACC1_BITS"] - 1), 1 << (P["ACC2_BITS"] - 1)
    prod_lim = 1 << (P["REQ1_PRODUCT_BITS"] - 1)

    embedding = read_mem(d / "embedding.hex", 8)
    fc1_w, fc1_b = read_mem(d / "fc1_weight.hex", 8), read_mem(d / "fc1_bias.hex", 32)
    fc2_w, fc2_b = read_mem(d / "fc2_weight.hex", 8), read_mem(d / "fc2_bias.hex", 32)
    tanh_lut = read_mem(d / "tanh_lut.hex", 8) if P["ACT_TANH"] else None
    token_to_ascii = read_mem(d / "token_to_ascii.hex", 8, signed=False)
    assert len(embedding) == V * E and len(fc1_w) == H * IN and len(fc2_w) == V * H
    assert len(fc1_b) == H and len(fc2_b) == V and (tanh_lut is None or len(tanh_lut) == 256)

    ctxs = read_mem(d / "test_context.hex", 8, signed=False)
    exp_hidden = read_mem(d / "test_hidden.hex", 8)
    exp_logits = read_mem(d / "test_logits.hex", 32)
    exp_argmax = read_mem(d / "test_argmax.hex", 8, signed=False)

    n_vectors = P["TEST_VECTORS"]
    for n in range(n_vectors):
        tokens = ctxs[n * C : (n + 1) * C]
        x = [embedding[tokens[p] * E + k] for p in range(C) for k in range(E)]

        hidden = []
        for o in range(H):
            acc1 = fc1_b[o]
            for i in range(IN):
                acc1 += fc1_w[o * IN + i] * x[i]
            assert -acc1_lim <= acc1 < acc1_lim, f"vector {n}: acc1[{o}] overflows ACC1_BITS"
            prod = acc1 * mult + (1 << (shift - 1))
            assert -prod_lim <= prod < prod_lim, f"vector {n}: requant product overflows"
            q = prod >> shift  # arithmetic shift
            if tanh_lut is not None:
                hidden.append(tanh_lut[max(-128, min(127, q)) & 0xFF])
            else:
                hidden.append(max(0, min(127, q)))
        assert hidden == exp_hidden[n * H : (n + 1) * H], f"vector {n}: hidden activations differ"

        logits = []
        for o in range(V):
            acc2 = fc2_b[o]
            for j in range(H):
                acc2 += fc2_w[o * H + j] * hidden[j]
            assert -acc2_lim <= acc2 < acc2_lim, f"vector {n}: acc2[{o}] overflows ACC2_BITS"
            logits.append(acc2)
        assert logits == exp_logits[n * V : (n + 1) * V], f"vector {n}: logits differ"

        best = 0  # first index wins ties, like a sequential comparator
        for o in range(1, V):
            if logits[o] > logits[best]:
                best = o
        assert best == exp_argmax[n], f"vector {n}: argmax {best} != {exp_argmax[n]}"

        if verbose and n < 8:
            text = "".join(chr(token_to_ascii[t]) for t in tokens)
            print(f"  {text!r:>8} -> {chr(token_to_ascii[best])!r}")

    if verbose:
        print(f"hardware check: {n_vectors}/{n_vectors} test vectors bit-exact, accumulators within params.vh widths")
    return n_vectors


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "model-weights")
