# nano-inference
A lightweight, low-latency inference engine running on a tiny Tang Nano 20k FPGA.

## Part 1: the model

A Bengio-style character-level language model (makemore MLP), trained on all
seven Harry Potter books and quantised to int8 so all of it fits in BRAM (data source credit: https://github.com/gastonstat/harry-potter-data).

```
4 previous chars -> embedding 64x16 -> concat (64) -> fc1 64->128 -> tanh -> fc2 128->64 -> logits
```

Weights: 1,024 (embedding) + 8,192 + 128 (fc1) + 8,192 + 64 (fc2) = 17,600.

### Training

```bash
pip install -r requirements.txt
python train.py                          # train and export to model-weights/
python train.py --tune 10                # Optuna TPE search first, save the best, then train with it
python train.py --tune 40 --tune-jobs 4  # same, running 4 trials at a time in worker processes
python train.py --hparams other.json     # use a specific hyperparameter file
python train.py --no-hparams             # ignore runs/best_hparams.json, use the built-in defaults
python -m model.hw_check model-weights   # hardware-style check of an export (train.py also runs it)
```

Hyperparameters come from, in order of precedence: `--tune` (the search's
best trial, also saved to `runs/best_hparams.json`), `--hparams FILE`,
`runs/best_hparams.json` if it exists, then the built-in defaults in
`model/trainer.py`. Individual flags such as `--lr` override all of them.
`--tune` resumes the study in `runs/optuna.db`, so its best covers earlier
trials too. Use a new `--study-name` when you change `--trial-epochs` /
`--trial-qat-epochs`, since losses from different trial budgets aren't
comparable.

One trial cannot keep a multi-core CPU busy with a model this small, so
`--tune-jobs N` runs N trials at once in worker processes that share the study
and split the CPU threads between them. The final training still runs once,
in the main process.

The built-in defaults come from an Optuna search (best post-QAT val loss 1.560
on a 1-epoch trial budget: tanh, batch 256, lr 1.77e-2).

The corpus is read from `harry-potter-data/csv-data-file/harry_potter_books.csv`
(override with `--data`). Text is lower-cased and folded onto a fixed 64-symbol
vocabulary (newline, space, a-z, 0-9, punctuation; see `model/vocab.py`) and
split train/val/test by whole chapters.

`train.py` trains in float, then fine-tunes with int8 fake quantisation (QAT),
converts to integers, exports hex files, reloads them and checks the reload is
bit-exact. It then runs the hardware-style check (`model/hw_check.py`), which
reads only the `.hex` files and `params.vh`, recomputes every golden test
vector with flat-memory addressing and plain integer math, and asserts the
accumulators fit the declared widths. It writes:

- `model-weights/`: FPGA image (below)
- `checkpoints/charmlp_float.pt`: float model before QAT
- `checkpoints/charmlp_qat.pt`: QAT model (fake-quant when `quantized=True`)
- `runs/`: Optuna study (`optuna.db`) and `best_hparams.json`

<!-- ### Integer pipeline (what the FPGA computes)

All tensors are symmetric per-tensor int8 with zero-point 0; biases are int32
at the accumulator scale.

1. `x[p*16 + d] = embedding[token[p]*16 + d]` for the 4 context tokens (int8).
2. `acc1[o] = fc1_bias[o] + sum_i fc1_weight[o*64 + i] * x[i]` (fits `ACC1_BITS`).
3. `pre = sat8((acc1 * REQ1_MULT + (1 << (REQ1_SHIFT-1))) >>> REQ1_SHIFT)`.
4. `h[o] = tanh_lut[pre as unsigned byte]` (int8, scale 1/127). With `--activation relu`
   there is no LUT and `h = clamp(step 3, 0, 127)`.
5. `logits[t] = fc2_bias[t] + sum_j fc2_weight[t*128 + j] * h[j]` (fits `ACC2_BITS`);
   the next character is `argmax(logits)`, or sample from `softmax(logits * logit_scale)`. -->

### `model-weights/`

One two's-complement word per line, loadable with `$readmemh`.

| file | words | bits | contents |
|---|---|---|---|
| `embedding.hex` | 64x16 | 8 | `addr = token*16 + d` |
| `fc1_weight.hex` | 128x64 | 8 | `addr = out*64 + position*16 + d` (position 0 = oldest char) |
| `fc1_bias.hex` | 128 | 32 | |
| `fc2_weight.hex` | 64x128 | 8 | `addr = out_token*128 + j` |
| `fc2_bias.hex` | 64 | 32 | |
| `tanh_lut.hex` | 256 | 8 | activation LUT addressed by the pre-activation byte |
| `ascii_to_token.hex` | 128 | 8 | UART input: ASCII to token (upper case folded, unknown to space) |
| `token_to_ascii.hex` | 64 | 8 | UART output: token to ASCII |
| `test_*.hex` | | | golden vectors: contexts, hidden activations, logits, argmax |
| `params.vh` | | | Verilog `localparam`s: sizes, `REQ1_MULT`/`REQ1_SHIFT`, accumulator widths |
| `model.json` | | | scales, layouts, metrics and hyperparameters |

### Code layout (`model/`)

| module | role |
|---|---|
| `vocab.py` | 64-symbol vocabulary, corpus text normalisation, ASCII tables |
| `data.py` | CSV loading, chapter split, batching |
| `net.py` | `CharMLP` (float + fake-quant forward), checkpoints |
| `quant.py` | fake-quant primitives, fixed-point requantisation |
| `int8_model.py` | `Int8Model`: integer-only golden model of the FPGA datapath |
| `trainer.py` | float training + QAT |
| `tune.py` | Optuna search over the training recipe |
| `export.py` | hex export / load |
| `hw_check.py` | standalone hardware-style verification of an export |
| `inference.py` | interchangeable backends (`TorchBackend`, `Int8Backend`) with `evaluate`/`generate` |

For benchmarking, every backend exposes `logits(contexts) -> (N, 64)`; an FPGA
backend only has to implement the same method to reuse `inference.evaluate`
and `inference.generate`:

```python
from model.inference import Int8Backend, TorchBackend, evaluate, generate
cpu = TorchBackend.from_checkpoint("checkpoints/charmlp_float.pt")
ref = Int8Backend.from_dir("model-weights")
```
