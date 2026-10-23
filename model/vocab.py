"""Fixed 64-symbol character vocabulary and corpus text normalisation.

Token ids are 6 bits wide on the FPGA, so the vocabulary is exactly 64 symbols
and does not depend on the corpus. Order: newline, space, a-z, 0-9, then ASCII
punctuation (minus the rarely useful ` { } | ~ ^).
"""

import re

PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]_"
CHARS = "\n " + "abcdefghijklmnopqrstuvwxyz" + "0123456789" + PUNCTUATION
VOCAB_SIZE = len(CHARS)
assert VOCAB_SIZE == 64 and len(set(CHARS)) == 64

STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = dict(enumerate(CHARS))
NEWLINE = STOI["\n"]
SPACE = STOI[" "]


def encode(text: str) -> list[int]:
    """Map already-normalised text to token ids (unknown characters are skipped)."""
    return [STOI[c] for c in text if c in STOI]


def decode(ids) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def ascii_to_token_table() -> list[int]:
    """128-entry lookup table from a raw ASCII byte to a token id.

    Upper case folds to lower case, a backtick becomes a double quote and every
    other byte outside the vocabulary maps to space. Intended for a UART front end.
    """
    table = []
    for b in range(128):
        c = chr(b).lower()
        if c == "`":
            c = '"'
        table.append(STOI.get(c, SPACE))
    return table


# --- corpus normalisation -------------------------------------------------

# Character fixes applied to the raw CSV text before anything else. The
# backtick is the dialogue quote mark in this corpus; curly quotes are folded
# into it so the paragraph heuristics below see a single quote symbol.
_RAW_REPLACEMENTS = [
    ("¨C", "-"),  # mojibake en dash (book 7)
    ("—", "-"),
    ("–", "-"),
    ("…", "..."),
    ("“", "`"),
    ("”", "`"),
    ("‘", "'"),
    ("’", "'"),
]
_PAGE_MARKER = re.compile(r"\*\s*\d+")  # e.g. "* 44" in book 2
_PARAGRAPH_GAP = re.compile(r"\s{2,}")
# Books 6-7 lost most paragraph breaks: "...was smoke.The yew hedges..."
_MERGED_SENTENCE = re.compile(r"(?<=[a-z][.!?])(?=[A-Z])")


def _split_merged_quotes(text: str) -> str:
    """Re-insert paragraph breaks around dialogue quotes glued to sentences.

    Tracks open/closed quote parity (reset at every newline). A quote that
    closes and is immediately followed by a capital letter or another quote
    ends a paragraph; a quote that opens immediately after sentence-ending
    punctuation starts one.
    """
    out = []
    quote_open = False
    n = len(text)
    for i, c in enumerate(text):
        if c == "\n":
            quote_open = False
            out.append(c)
            continue
        if c != "`":
            out.append(c)
            continue
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < n else ""
        if quote_open:
            out.append(c)
            if nxt.isupper() or nxt == "`":
                out.append("\n")
                quote_open = False
                continue
        else:
            if prev and prev in ".!?" and i > 1 and text[i - 2].isalpha():
                out.append("\n")
            out.append(c)
        quote_open = not quote_open
    return "".join(out)


def normalize(text: str) -> str:
    """Turn one chapter of raw CSV text into a string over CHARS only."""
    for old, new in _RAW_REPLACEMENTS:
        text = text.replace(old, new)
    text = _PAGE_MARKER.sub(" ", text)
    text = _PARAGRAPH_GAP.sub("\n", text)
    text = _MERGED_SENTENCE.sub("\n", text)
    text = _split_merged_quotes(text)
    text = text.replace("`", '"').lower()
    text = "".join(c for c in text if c in STOI)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    return text.strip()
