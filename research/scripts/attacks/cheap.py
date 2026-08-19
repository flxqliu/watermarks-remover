#!/usr/bin/env python3
"""Cheap, deterministic, local text attacks (protocol §4.1, attack A7).

Implements the three "cheap baseline" attacks from
research/01-experiment-protocol.md §4.1 row A7 (prior-art anchor: Random
Walk / impossibility, ICML 2024), tracked as gap 05-A3 in
research/05-arxiv-readiness.md:

- ``synonym``: deterministic synonym substitution. Uses nltk WordNet when
  it is importable *and* its corpus data is present locally (the probe
  never downloads anything); otherwise falls back to a small built-in map
  of ~40 common English word pairs embedded in this file.
- ``delete``: random deletion of ~5-10% of words (``--delete-ratio``,
  clamped to [0.02, 0.20]).
- ``sentence-reorder``: shuffle sentences. Sentences split on ``[.!?]``
  followed by whitespace + a capital letter (fallback: split on ``". "``).

Every attack is fully deterministic given its seed: each function creates
a private ``random.Random(seed)`` and never touches the global RNG. Numbers
(``\\d+``) and URLs (``https?://\\S+``) are tokenized apart and are never
substituted, deleted, or split.

CLI::

    python3 research/scripts/attacks/cheap.py \
        --input FILE --output FILE \
        --attack {synonym,delete,sentence-reorder} [--seed N] [--delete-ratio F]

Word tokens are ASCII letter runs (optionally with internal apostrophes or
hyphens); non-ASCII words pass through every attack untouched.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

# Fraction of eligible (substitutable) word tokens the synonym attack
# actually replaces. Deterministic given the seed.
SYNONYM_RATE = 0.5

# Bounds for --delete-ratio / random_word_delete(ratio=...).
MIN_DELETE_RATIO = 0.02
MAX_DELETE_RATIO = 0.20

# Small curated map used when WordNet is unavailable: ~40 common English
# word pairs (same part of speech, common register, single-word targets).
_BUILTIN_SYNONYMS: dict[str, str] = {
    "happy": "glad",
    "sad": "unhappy",
    "angry": "mad",
    "afraid": "scared",
    "big": "large",
    "small": "little",
    "fast": "quick",
    "slow": "sluggish",
    "quick": "rapid",
    "beautiful": "pretty",
    "ugly": "hideous",
    "good": "fine",
    "bad": "poor",
    "great": "excellent",
    "nice": "pleasant",
    "awful": "terrible",
    "smart": "clever",
    "strong": "powerful",
    "weak": "feeble",
    "rich": "wealthy",
    "old": "ancient",
    "new": "novel",
    "young": "youthful",
    "difficult": "hard",
    "easy": "simple",
    "important": "significant",
    "interesting": "fascinating",
    "buy": "purchase",
    "begin": "start",
    "end": "finish",
    "help": "assist",
    "think": "believe",
    "see": "observe",
    "look": "glance",
    "talk": "speak",
    "say": "state",
    "get": "obtain",
    "give": "grant",
    "make": "create",
    "use": "employ",
    "show": "display",
    "tell": "inform",
    "ask": "inquire",
    "answer": "reply",
}

# Tokenizer used by the substitution and deletion attacks. Every character
# is matched by exactly one named alternative, so concatenating the values
# reproduces the input byte-for-byte; attacks can drop or rewrite tokens
# without losing anything else.
_TOKEN_RE = re.compile(
    r"(?P<url>https?://\S+)"
    r"|(?P<number>\d+(?:[.,]\d+)*)"
    r"|(?P<word>[A-Za-z]+(?:['-][A-Za-z]+)*)"
    r"|(?P<space>\s+)"
    r"|(?P<other>.)"
)

# Sentence boundaries: whitespace after terminal punctuation, followed by
# a capital letter. The punctuation itself is kept with the sentence via a
# lookbehind, so splits never drop it. The loose fallback also accepts a
# lowercase start but never splits a digit-preceded period, so decimals
# like "3. 14" are left intact.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])(\s+)(?=[A-Z])")
_SENT_BOUNDARY_LOOSE_RE = re.compile(r"(?<=(?<![0-9])[.!?])(\s+)(?=[A-Za-z])")


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Split *text* into ``(kind, value)`` tokens, preserving everything.

    Kinds: ``url``, ``number``, ``word``, ``space``, ``other`` (a single
    punctuation/unknown character). Concatenating the values reproduces
    *text* exactly.
    """
    return [(match.lastgroup or "other", match.group()) for match in _TOKEN_RE.finditer(text)]


@lru_cache(maxsize=1)
def _wordnet_available() -> bool:
    """True iff nltk WordNet is importable and its corpus data is present.

    Probes the local install only (``nltk.data.find``); never calls
    ``nltk.download``, so this works offline. Callers fall back to
    ``_BUILTIN_SYNONYMS`` when it returns False.
    """
    try:
        import nltk
    except ImportError:
        return False
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        return False
    return True


@lru_cache(maxsize=4096)
def _wordnet_synonyms(word: str) -> list[str]:
    """Sorted distinct lemma candidates for lowercase *word* (WordNet)."""
    from nltk.corpus import wordnet

    candidates: set[str] = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word:
                candidates.add(name)
    return sorted(candidates)


def _match_case(original: str, replacement: str) -> str:
    """Apply *original*'s casing to *replacement* (upper/title/lower)."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _synonym_for(word: str) -> str | None:
    """Deterministic synonym candidate for *word*, or None if none exists.

    Checks the built-in map first, then WordNet when available.
    """
    key = word.lower()
    mapped = _BUILTIN_SYNONYMS.get(key)
    if mapped is not None:
        return mapped
    if _wordnet_available():
        candidates = _wordnet_synonyms(key)
        if candidates:
            return candidates[0]
    return None


def synonym_substitute(text: str, seed: int) -> str:
    """Replace some substitutable words with synonyms (deterministic in seed).

    Each word token that has a synonym is replaced with probability
    ``SYNONYM_RATE``, decided by the seeded RNG; numbers and URLs are never
    touched.
    """
    rng = _seeded_rng(seed)
    out: list[str] = []
    for kind, value in _tokenize(text):
        if kind == "word":
            replacement = _synonym_for(value)
            if replacement is not None and rng.random() < SYNONYM_RATE:
                out.append(_match_case(value, replacement))
                continue
        out.append(value)
    return "".join(out)


def _clamp_delete_ratio(ratio: float) -> float:
    """Clamp *ratio* into [MIN_DELETE_RATIO, MAX_DELETE_RATIO]."""
    return max(MIN_DELETE_RATIO, min(MAX_DELETE_RATIO, ratio))


# S311 is deliberate: this is a reproducible text attack, not cryptography.
def _seeded_rng(seed: int) -> random.Random:
    """Private RNG seeded with *seed*; never the global RNG (determinism)."""
    return random.Random(seed)  # noqa: S311


def random_word_delete(text: str, seed: int, ratio: float = 0.07) -> str:
    """Delete a random subset of words (deterministic in seed).

    ``ratio`` is clamped to [0.02, 0.20]; the number deleted is
    ``round(ratio * n_words)``. Numbers and URLs are never deleted. An
    adjacent plain space is dropped with each deleted word so the text
    stays tidy.
    """
    ratio = _clamp_delete_ratio(ratio)
    tokens = _tokenize(text)
    word_indices = [i for i, (kind, _value) in enumerate(tokens) if kind == "word"]
    n_words = len(word_indices)
    if n_words == 0:
        return text
    rng = _seeded_rng(seed)
    n_delete = round(ratio * n_words)
    if n_delete <= 0:
        return text
    delete = set(rng.sample(word_indices, min(n_delete, n_words)))
    for i in sorted(delete):
        for j in (i - 1, i + 1):
            if (
                0 <= j < len(tokens)
                and j not in delete
                and tokens[j][0] == "space"
                and tokens[j][1] == " "
            ):
                delete.add(j)
                break
    kept = [value for i, (_kind, value) in enumerate(tokens) if i not in delete]
    return re.sub(r" {2,}", " ", "".join(kept))


def _split_sentences(text: str) -> tuple[list[str], list[str]]:
    """Split *text* into ``(sentences, separators)``.

    The first separator is ``""``; separator *i* is the whitespace that
    preceded sentence *i* in the original text, so rejoining
    ``sentences[0] + separators[1] + sentences[1] + ...`` reproduces *text*
    exactly (up to sentence order).
    """
    parts = _SENT_BOUNDARY_RE.split(text)
    if len(parts) <= 1:
        parts = _SENT_BOUNDARY_LOOSE_RE.split(text)
    return parts[0::2], ["", *parts[1::2]]


def sentence_reorder(text: str, seed: int) -> str:
    """Shuffle sentences (deterministic in seed), preserving every token.

    The multiset of characters - and therefore of word tokens - is
    unchanged; only the order of sentences (and which inter-sentence
    whitespace precedes which) is permuted.
    """
    sentences, separators = _split_sentences(text)
    if len(sentences) <= 1:
        return text
    rng = _seeded_rng(seed)
    rng.shuffle(sentences)
    parts = [sentences[0]]
    for i in range(1, len(sentences)):
        parts.append(separators[i])
        parts.append(sentences[i])
    return "".join(parts)


def apply_attack(text: str, attack: str, seed: int, delete_ratio: float = 0.07) -> str:
    """Apply the named attack (``synonym`` | ``delete`` | ``sentence-reorder``)."""
    if attack == "synonym":
        return synonym_substitute(text, seed)
    if attack == "delete":
        return random_word_delete(text, seed, delete_ratio)
    if attack == "sentence-reorder":
        return sentence_reorder(text, seed)
    raise ValueError(
        f"unknown attack {attack!r}; expected one of synonym, delete, sentence-reorder"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments (see module docstring)."""
    parser = argparse.ArgumentParser(
        description="Cheap deterministic text attacks (research/01-experiment-protocol.md §4.1 A7)."
    )
    parser.add_argument("--input", required=True, help="UTF-8 text file to read")
    parser.add_argument("--output", required=True, help="UTF-8 file to write the attacked text to")
    parser.add_argument(
        "--attack",
        required=True,
        choices=("synonym", "delete", "sentence-reorder"),
        help="attack to apply",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--delete-ratio",
        type=float,
        default=0.07,
        help="fraction of words to delete, clamped to [0.02, 0.20] (default: 0.07)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: read --input, attack, write --output. 0 on success."""
    args = parse_args(argv)
    try:
        text = Path(args.input).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cheap.py: cannot read {args.input!r}: {exc}", file=sys.stderr)
        return 1
    transformed = apply_attack(text, args.attack, args.seed, args.delete_ratio)
    try:
        Path(args.output).write_text(transformed, encoding="utf-8")
    except OSError as exc:
        print(f"cheap.py: cannot write {args.output!r}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
