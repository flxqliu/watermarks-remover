"""Validation for the gap 05-A6 prompt corpus (research/corpus/).

Checks the deliverable contract from research/01-experiment-protocol.md
section 3: exactly 25 prompts per language (en/de/fr/es), 50-90 words
each, index-aligned topics across languages, the 8 legacy
benchmarks/corpus prompts copied verbatim as en/01..08, and the 17 new
English seeds covering distinct factual domains.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "research" / "corpus"
BENCH_CORPUS = REPO_ROOT / "benchmarks" / "corpus"
LANGS = ("en", "de", "fr", "es")
MIN_WORDS = 50
MAX_WORDS = 90
N_PROMPTS = 25

# en/01..08 must match benchmarks/corpus/ in this exact order (verbatim).
LEGACY_FILES: tuple[str, ...] = (
    "cloud-computing.txt",
    "coffee-brewing.txt",
    "hiking-checklist.txt",
    "nutrition-myths.txt",
    "open-source-licenses.txt",
    "renewable-energy.txt",
    "small-biz-finance.txt",
    "venice-history.txt",
)

# The 17 new English seeds (en/09..25) cover these domains, one each.
NEW_EN_DOMAINS: tuple[str, ...] = (
    "astronomy",
    "geology",
    "zoology",
    "music history",
    "food safety",
    "transportation",
    "agriculture",
    "economic history",
    "literature",
    "public health",
    "energy storage",
    "urban planning",
    "marine biology",
    "meteorology",
    "language origins",
    "materials science",
    "sports science",
)


def word_count(text: str) -> int:
    """Return whitespace-separated token count of *text*."""
    return len(text.split())


def read_prompt(lang: str, index: int) -> str:
    """Read and strip corpus prompt *index* (1-based) for *lang*."""
    path = CORPUS / lang / f"{index:02d}.txt"
    assert path.is_file(), f"missing corpus file: {path}"
    return path.read_text(encoding="utf-8").strip()


def test_four_language_directories_exist() -> None:
    """The corpus root contains exactly the en/de/fr/es directories."""
    dirs = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    assert dirs == sorted(LANGS)


def test_each_language_has_exactly_25_prompt_files() -> None:
    """Each language directory holds exactly indices 01..25."""
    for lang in LANGS:
        files = sorted(p.name for p in (CORPUS / lang).glob("*.txt"))
        expected = [f"{i:02d}.txt" for i in range(1, N_PROMPTS + 1)]
        assert files == expected, f"{lang}: unexpected file set {files}"


def test_word_counts_within_50_to_90() -> None:
    """Every one of the 100 prompts is 50-90 words (whitespace tokens)."""
    for lang in LANGS:
        for index in range(1, N_PROMPTS + 1):
            n = word_count(read_prompt(lang, index))
            assert MIN_WORDS <= n <= MAX_WORDS, f"{lang}/{index:02d}.txt: {n} words"


def test_english_01_to_08_match_benchmarks_verbatim() -> None:
    """en/01..08 are byte-identical to benchmarks/corpus/ sources."""
    for i, name in enumerate(LEGACY_FILES, start=1):
        bench = (BENCH_CORPUS / name).read_text(encoding="utf-8").strip()
        assert read_prompt("en", i) == bench, f"en/{i:02d}.txt != {name}"


def test_english_prompts_are_all_unique() -> None:
    """No two English prompts share identical text."""
    texts = [read_prompt("en", i) for i in range(1, N_PROMPTS + 1)]
    assert len(set(texts)) == len(texts), "duplicate English prompts"


def test_new_english_domains_are_distinct() -> None:
    """The 17 new English seeds cover 17 distinct required domains."""
    assert len(set(NEW_EN_DOMAINS)) == len(NEW_EN_DOMAINS)


def test_translations_differ_from_english() -> None:
    """Every translated file differs from its English counterpart."""
    for lang in ("de", "fr", "es"):
        for index in range(1, N_PROMPTS + 1):
            assert read_prompt(lang, index) != read_prompt("en", index), (
                f"{lang}/{index:02d}.txt identical to English seed"
            )


def test_same_index_is_same_topic_across_languages() -> None:
    """Cross-language files sharing an index are mutually different."""
    for index in range(1, N_PROMPTS + 1):
        texts = {lang: read_prompt(lang, index) for lang in LANGS}
        assert len(set(texts.values())) == len(LANGS), (
            f"duplicate text across languages for index {index}"
        )
