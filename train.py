"""Train the character-level CharMLP on the Harry Potter corpus, quantise it to
int8 and export $readmemh hex weights for the FPGA.

    python train.py                                  # tuned defaults -> model-weights/
    python train.py --tune 10                        # Optuna search first, then train with the best settings
    python train.py --hparams runs/best_hparams.json # reuse a previous search
    python -m model.hw_check model-weights           # re-run the hardware-style check on an export
"""

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from model import data, export, hw_check, inference, tune
from model.int8_model import Int8Model
from model.net import ACTIVATIONS, CharMLP, ModelConfig, save_checkpoint
from model.trainer import TrainConfig, fit, resolve_device

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(data.DEFAULT_CSV), help="harry_potter_books.csv")
    p.add_argument("--out", default=str(ROOT / "model-weights"), help="hex export directory")
    p.add_argument("--checkpoints", default=str(ROOT / "checkpoints"), help="PyTorch checkpoint directory")
    p.add_argument("--device", default="auto", help="auto (cuda, else cpu) | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=1337)

    g = p.add_argument_group("training (override defaults and tuned values)")
    g.add_argument("--hparams", help="JSON written by a previous --tune run")
    g.add_argument("--activation", choices=ACTIVATIONS)
    g.add_argument("--epochs", type=float)
    g.add_argument("--qat-epochs", type=float)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--qat-lr", type=float)
    g.add_argument("--weight-decay", type=float)

    g = p.add_argument_group("hyperparameter search")
    g.add_argument("--tune", type=int, default=0, metavar="N_TRIALS", help="run N Optuna trials before training")
    g.add_argument("--trial-epochs", type=float, default=1.0)
    g.add_argument("--trial-qat-epochs", type=float, default=0.25)
    g.add_argument("--tune-timeout", type=float, help="stop the search after this many seconds")
    g.add_argument("--study-name", default="charmlp")
    g.add_argument("--storage", default=f"sqlite:///{ROOT / 'runs' / 'optuna.db'}")
    g.add_argument("--hparams-out", default=str(ROOT / "runs" / "best_hparams.json"), help="where to save the best trial")

    g = p.add_argument_group("quantisation / export")
    g.add_argument("--mult-bits", type=int, default=16, help="width of the requantisation multiplier")
    g.add_argument("--test-vectors", type=int, default=32, help="golden input/output vectors to export")
    g.add_argument("--sample-chars", type=int, default=400, help="characters of sample text to print")
    return p.parse_args()


def configure(args, corpus) -> tuple[ModelConfig, TrainConfig, dict]:
    model_cfg, train_cfg = ModelConfig(), TrainConfig(seed=args.seed)
    hp = tune.load_hparams(args.hparams) if args.hparams else {}
    if args.tune:
        study = tune.run_study(
            corpus,
            model_cfg,
            train_cfg,
            n_trials=args.tune,
            trial_epochs=args.trial_epochs,
            trial_qat_epochs=args.trial_qat_epochs,
            storage=args.storage,
            study_name=args.study_name,
            timeout=args.tune_timeout,
            seed=args.seed,
        )
        hp = tune.best_hparams(study)
        tune.save_hparams(hp, args.hparams_out)
        print(f"[tune] best post-QAT val loss {study.best_value:.4f} with {hp}")
    model_cfg, train_cfg = tune.apply_hparams(hp, model_cfg, train_cfg)

    if args.activation:
        model_cfg = replace(model_cfg, activation=args.activation)
    overrides = {
        "epochs": args.epochs,
        "qat_epochs": args.qat_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "qat_lr": args.qat_lr,
        "weight_decay": args.weight_decay,
    }
    train_cfg = replace(train_cfg, **{k: v for k, v in overrides.items() if v is not None})
    return model_cfg, train_cfg, hp


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    t0 = time.time()
    corpus = data.build_corpus(args.data, context=ModelConfig().context, seed=args.seed)
    sizes = "  ".join(f"{s} {corpus.num_examples(s):,}" for s in ("train", "val", "test"))
    print(f"corpus: {sizes} chars ({time.time() - t0:.1f}s), device {device}")
    corpus_dev = corpus.to(device)

    model_cfg, train_cfg, hp = configure(args, corpus_dev)
    print(f"model: {asdict(model_cfg)}\ntrain: {asdict(train_cfg)}")

    model = CharMLP(model_cfg).to(device)
    print(f"weights: {model.num_weights()} = {sum(model.num_weights().values()):,}")
    result = fit(model, corpus_dev, train_cfg)
    model = model.cpu().eval()

    ckpt_dir = Path(args.checkpoints)
    float_model = CharMLP(model_cfg)
    float_model.load_state_dict(result["float_state"])
    save_checkpoint(ckpt_dir / "charmlp_float.pt", float_model, train=asdict(train_cfg), hparams=hp)
    save_checkpoint(ckpt_dir / "charmlp_qat.pt", model, train=asdict(train_cfg), hparams=hp)

    q = Int8Model.from_float(model, args.mult_bits)
    backends = {
        "float32": inference.TorchBackend(float_model, name="float32"),
        "fake-quant": inference.TorchBackend(model, name="fake-quant"),
        "int8": inference.Int8Backend(q),
    }
    metrics = {}
    print(f"\n{'model':<12} {'split':<5} {'loss':>7} {'bpc':>6} {'acc':>6}")
    for name, backend in backends.items():
        for split in ("val", "test"):
            m = inference.evaluate(backend, corpus, split)
            metrics.setdefault(name, {})[split] = m
            print(f"{name:<12} {split:<5} {m['loss']:7.4f} {m['bpc']:6.3f} {m['acc']:6.3f}")

    rng = np.random.default_rng(args.seed)
    ix = torch.from_numpy(rng.choice(corpus.num_examples("test"), args.test_vectors, replace=False))
    test_ctx, _ = data.windows(corpus.test, ix, corpus.context)
    hparams = {"model": asdict(model_cfg), "train": asdict(train_cfg), "tuned": hp}
    manifest = export.export(q, args.out, test_ctx.numpy(), metrics=metrics, hparams=hparams)

    # Reload purely from the hex files: the FPGA image must be bit-exact with the reference.
    q_hex = export.load(args.out)
    all_test, _ = data.windows(corpus.test, torch.arange(corpus.num_examples("test")), corpus.context)
    assert np.array_equal(q.int_logits(all_test.numpy()), q_hex.int_logits(all_test.numpy())), "hex round trip mismatch"
    acc = manifest["quant"]["accumulator_bits"]
    print(
        f"\nexported {manifest['num_weights']:,} weights to {args.out}/ (hex round trip bit-exact; "
        f"req1 = *{q.req1_mult} >>> {q.req1_shift}; acc1 {acc['acc1']}b, acc2 {acc['acc2']}b)"
    )
    hw_check.check(args.out)

    if args.sample_chars:
        prompt = "harry looked at "
        int8_hex = inference.Int8Backend(q_hex)
        print("\n--- float32, T=0.8 ---")
        print(inference.generate(backends["float32"], prompt, args.sample_chars, 0.8, seed=args.seed))
        print("\n--- int8 from hex, T=0.8 ---")
        print(inference.generate(int8_hex, prompt, args.sample_chars, 0.8, seed=args.seed))
        print("\n--- int8 from hex, greedy (argmax) ---")
        print(inference.generate(int8_hex, prompt, args.sample_chars // 2, 0.0))


if __name__ == "__main__":
    main()
