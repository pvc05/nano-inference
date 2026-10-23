"""Harry Potter corpus loading, chapter-level train/val/test split and batching."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from . import vocab

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "harry-potter-data" / "csv-data-file" / "harry_potter_books.csv"


@dataclass
class Chapter:
    book: str
    chapter: str
    text: str  # normalised


@dataclass
class Corpus:
    """Token-id streams (int64) for each split. Example i of a split is
    ids[i : i + context] -> ids[i + context]."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    context: int

    def split(self, name: str) -> torch.Tensor:
        return getattr(self, name)

    def num_examples(self, name: str) -> int:
        return len(self.split(name)) - self.context

    def to(self, device) -> "Corpus":
        return Corpus(self.train.to(device), self.val.to(device), self.test.to(device), self.context)


def load_chapters(csv_path: str | Path = DEFAULT_CSV) -> list[Chapter]:
    """Read the CSV (one wrapped line per row) and rebuild normalised chapters in order."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"corpus not found at {csv_path} (pass --data)")
    lines: dict[tuple[str, str], list[str]] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lines.setdefault((row["book"], row["chapter"]), []).append(row["text"])
    return [Chapter(book, chap, vocab.normalize(" ".join(rows))) for (book, chap), rows in lines.items()]


def _stream(chapters: list[Chapter], context: int) -> torch.Tensor:
    # Chapters are separated by a blank line; the leading newlines give the
    # first characters a full (boundary-padded) context.
    text = "\n" * context + "\n\n".join(ch.text for ch in chapters) + "\n"
    return torch.tensor(vocab.encode(text), dtype=torch.long)


def build_corpus(
    csv_path: str | Path = DEFAULT_CSV,
    context: int = 4,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 1337,
) -> Corpus:
    """Split whole chapters (shuffled with `seed`) into train/val/test streams."""
    chapters = load_chapters(csv_path)
    order = list(range(len(chapters)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(chapters) * val_frac))
    n_test = max(1, round(len(chapters) * test_frac))
    val_idx = sorted(order[:n_val])
    test_idx = sorted(order[n_val : n_val + n_test])
    train_idx = sorted(order[n_val + n_test :])
    pick = lambda idx: _stream([chapters[i] for i in idx], context)  # noqa: E731
    return Corpus(pick(train_idx), pick(val_idx), pick(test_idx), context)


def windows(ids: torch.Tensor, ix: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather (contexts, targets) for example indices `ix`: shapes (B, context), (B,)."""
    win = ids[ix.unsqueeze(1) + torch.arange(context + 1, device=ids.device)]
    return win[:, :-1], win[:, -1]


def random_batch(corpus: Corpus, split: str, batch_size: int, generator: torch.Generator | None = None):
    """Uniformly sampled examples. `generator` is a CPU generator, so runs are
    reproducible on any device."""
    ids = corpus.split(split)
    ix = torch.randint(0, len(ids) - corpus.context, (batch_size,), generator=generator)
    return windows(ids, ix.to(ids.device), corpus.context)


def iter_batches(corpus: Corpus, split: str, batch_size: int, limit: int | None = None):
    """Deterministic sequential pass over a split (optionally the first `limit` examples)."""
    ids = corpus.split(split)
    n = corpus.num_examples(split) if limit is None else min(limit, corpus.num_examples(split))
    for start in range(0, n, batch_size):
        ix = torch.arange(start, min(start + batch_size, n), device=ids.device)
        yield windows(ids, ix, corpus.context)
