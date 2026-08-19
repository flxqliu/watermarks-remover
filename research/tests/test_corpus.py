"""Corpus validation (gap 05-A6).

research/corpus/ must hold 25 factual, neutral prompts per language
(en/de/fr/es), 50-90 words each, index-aligned across languages (same
index = same topic, translated). The English set is self-written and
unique; the 8 legacy prompts from benchmarks/corpus/ are copied verbatim
as en/01..08.
"""

from __future__ import annotations

from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
LANGUAGES = ("en", "de", "fr", "es")
N_PROMPTS = 25
MIN_WORDS = 50
MAX_WORDS = 90


def _word_count(text: str) -> int:
    return len([t for t in text.split() if t.strip()])


def _load(lang: str) -> dict[int, str]:
    d = CORPUS / lang
    assert d.is_dir(), f"missing corpus dir {d}"
    files = sorted(p for p in d.glob("*.txt") if p.is_file())
    assert len(files) == N_PROMPTS, f"{lang}: expected {N_PROMPTS} files, got {len(files)}"
    out: dict[int, str] = {}
    for f in files:
        idx = int(f.stem)
        out[idx] = f.read_text(encoding="utf-8").strip()
    return out


def test_every_language_has_25_prompts() -> None:
    for lang in LANGUAGES:
        prompts = _load(lang)
        assert set(prompts) == set(range(1, N_PROMPTS + 1))


def test_word_counts_within_range() -> None:
    for lang in LANGUAGES:
        for idx, text in _load(lang).items():
            n = _word_count(text)
            assert MIN_WORDS <= n <= MAX_WORDS, f"{lang}/{idx:02d}.txt: {n} words"


def test_english_prompts_are_unique() -> None:
    en = _load("en")
    texts = list(en.values())
    assert len(set(texts)) == len(texts), "duplicate EN prompts"


def test_translations_actually_differ_from_english() -> None:
    en = _load("en")
    for lang in ("de", "fr", "es"):
        for idx, text in _load(lang).items():
            assert text != en[idx], f"{lang}/{idx:02d}.txt identical to EN (not a translation)"


def test_legacy_prompts_copied_verbatim() -> None:
    """en/01..08 match benchmarks/corpus/ byte-for-byte."""
    legacy = CORPUS.parents[1] / "benchmarks" / "corpus"
    legacy_files = sorted(p for p in legacy.glob("*.txt") if p.is_file())
    assert len(legacy_files) == 8
    en = _load("en")
    for i, f in enumerate(legacy_files, start=1):
        assert en[i] == f.read_text(encoding="utf-8").strip(), f"en/{i:02d}.txt != {f.name}"
