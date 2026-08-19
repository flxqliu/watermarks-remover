"""Tests for research/scripts/attacks/cheap.py (protocol §4.1, attack A7).

All tests are fast and offline: nltk is never required (the module lazily
falls back to the built-in synonym map), and no network access happens.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from attacks import cheap

WORD_RE = re.compile(r"[A-Za-z]+")
PROTECTED_RE = re.compile(r"\d+(?:[.,]\d+)*|https?://\S+")


def words(text: str) -> list[str]:
    """Letter-run tokens of *text* (the module counts the same way)."""
    return WORD_RE.findall(text)


def protected(text: str) -> list[str]:
    """All numbers and URLs embedded in *text*."""
    return PROTECTED_RE.findall(text)


# A few sentences with words that are in the built-in map, a number, and a
# URL, so every attack exercises its protected-token path too.
BASE_TEXT = (
    "The happy scientist saw the fast result and told the angry manager. "
    "She could not believe the big number 42 in the https://example.com/x report. "
    "Everything seemed easy and important that day."
)

DELETION_TEXT = " ".join(
    ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    * 25
)

REORDER_TEXT = (
    "The first sentence has five words here. "
    "The second sentence talks about the moon. "
    "The third sentence mentions a large number 12345. "
    "The fourth sentence ends with a question? "
    "The fifth sentence closes the story."
)

# 20 words that are all keys of the built-in map; with SYNONYM_RATE=0.5 a
# fixed seed virtually always rewrites at least one of them.
SYNONYM_TEXT = (
    "The happy happy happy happy happy dog ran fast fast fast fast fast. "
    "The big big big big big house was old old old old old."
)


@pytest.mark.parametrize("attack", ["synonym", "delete", "sentence-reorder"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_determinism(attack: str, seed: int) -> None:
    """Same seed -> byte-identical output for every attack."""
    first = cheap.apply_attack(BASE_TEXT, attack, seed)
    second = cheap.apply_attack(BASE_TEXT, attack, seed)
    assert first == second


@pytest.mark.parametrize("ratio", [0.02, 0.07, 0.20])
def test_delete_ratio_in_bounds(ratio: float) -> None:
    """Surviving word fraction stays within ratio +- 0.02."""
    out = cheap.random_word_delete(DELETION_TEXT, seed=0, ratio=ratio)
    survived = len(words(out)) / len(words(DELETION_TEXT))
    assert 1 - ratio - 0.02 <= survived <= 1 - ratio + 0.02


def test_delete_ratio_clamped_to_max() -> None:
    """Ratios above 0.20 are clamped, not honored literally."""
    out = cheap.random_word_delete(DELETION_TEXT, seed=0, ratio=0.9)
    survived = len(words(out)) / len(words(DELETION_TEXT))
    assert 1 - 0.20 - 0.02 <= survived <= 1 - 0.20 + 0.02


def test_sentence_reorder_preserves_word_multiset() -> None:
    """Reorder shuffles sentences but keeps every character (and word)."""
    out = cheap.sentence_reorder(REORDER_TEXT, seed=3)
    assert Counter(out) == Counter(REORDER_TEXT)
    assert out != REORDER_TEXT


def test_sentence_reorder_single_sentence_is_identity() -> None:
    """A one-sentence text has nothing to reorder."""
    single = "Only one sentence lives here, and it stays put."
    assert cheap.sentence_reorder(single, seed=0) == single


def test_synonym_changes_at_least_one_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in map path changes >= 1 word (WordNet probed off)."""
    monkeypatch.setattr(cheap, "_wordnet_available", lambda: False)
    out = cheap.synonym_substitute(SYNONYM_TEXT, seed=0)
    assert out != SYNONYM_TEXT
    assert cheap._wordnet_available() is False  # patch in effect


@pytest.mark.parametrize("attack", ["synonym", "delete", "sentence-reorder"])
def test_numbers_and_urls_preserved(attack: str) -> None:
    """No attack alters or drops numbers or URLs."""
    out = cheap.apply_attack(BASE_TEXT, attack, seed=7)
    for item in protected(BASE_TEXT):
        assert item in out


def test_apply_attack_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        cheap.apply_attack("text", "bogus", seed=0)


def test_cli_round_trip(tmp_path: Path) -> None:
    """CLI reads a file, attacks it, writes output; rerun is identical."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    src.write_text(BASE_TEXT, encoding="utf-8")
    rc = cheap.main(
        [
            "--input",
            str(src),
            "--output",
            str(dst),
            "--attack",
            "delete",
            "--seed",
            "0",
            "--delete-ratio",
            "0.07",
        ]
    )
    assert rc == 0
    out_text = dst.read_text(encoding="utf-8")
    assert out_text != BASE_TEXT
    dst2 = tmp_path / "out2.txt"
    rc2 = cheap.main(
        [
            "--input",
            str(src),
            "--output",
            str(dst2),
            "--attack",
            "delete",
            "--seed",
            "0",
        ]
    )
    assert rc2 == 0
    assert dst2.read_text(encoding="utf-8") == out_text


def test_cli_missing_input_returns_nonzero(tmp_path: Path) -> None:
    """A missing input file is an error, not a crash."""
    dst = tmp_path / "out.txt"
    rc = cheap.main(
        [
            "--input",
            str(tmp_path / "missing.txt"),
            "--output",
            str(dst),
            "--attack",
            "synonym",
            "--seed",
            "0",
        ]
    )
    assert rc == 1


def test_parse_args_defaults() -> None:
    """Defaults: seed 0, delete ratio 0.07."""
    args = cheap.parse_args(["--input", "a", "--output", "b", "--attack", "delete"])
    assert args.seed == 0
    assert args.delete_ratio == 0.07

    with pytest.raises(SystemExit):
        cheap.parse_args(["--input", "a", "--output", "b", "--attack", "bogus"])
