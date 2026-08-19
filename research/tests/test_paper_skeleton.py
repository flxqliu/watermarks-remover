"""Structural tests for the arXiv v1 paper skeleton (research/paper/, gaps C1/C2).

These tests validate the *skeleton*: the exact title, the section order
from research/02-paper-outline.md section 5, the input{} wiring, the
table/figure placeholders from 02 section 6, brace balance, and full
reference coverage of research/03-related-work.md sections A--E. They
intentionally assert nothing about experiment numbers, which do not
exist yet.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1] / "paper"
RELATED_WORK = Path(__file__).resolve().parents[1] / "03-related-work.md"

TITLE = (
    "How Fragile Are Deployed Text Watermarks? An Empirical Study of "
    "Layered Watermark Removal under Realistic User-Side Editing"
)

SECTIONS = [
    "Introduction",
    "Background and Related Work",
    "Threat Model and System",
    "Experimental Setup",
    "Results",
    "Analysis and Case Study",
    "Policy Discussion",
    "Limitations",
    "Ethics Statement",
    "Conclusion",
]

#: Every entry in research/03-related-work.md A--E as bib keys
#: (ID = first-author-lastname + arXiv year [+ short tag]).
EXPECTED_BIB_KEYS = [
    # A. Watermarking methods (the attack surface we test)
    "kirchenbauer2023kwg",
    "christ2023undetectable",
    "kuditipudi2023robust",
    "liu2023sir",
    "liu2023upv",
    "zhao2024permute",
    "hu2023unbiased",
    "wu2023dipmark",
    "huo2024tswatermark",
    "liu2024adaptive",
    "hou2023semstamp",
    "hou2024ksemstamp",
    "gu2025invisible",
    "wang2025morphmark",
    "yang2023blackbox",
    # B. Robustness, attacks, and limits
    "kirchenbauer2023reliability",
    "he2024xsir",
    "zhang2023sand",
    "jovanovic2024stealing",
    "pan2024waterseeker",
    "lu2024ewd",
    "liu2024crafted",
    "pan2025distillation",
    # C. Closest recent neighbors
    "han2025synthid",
    "omidi2026synthid",
    "tamim2026forensic",
    "harelcanada2025sandcastles",
    # D. Tools & surveys
    "pan2024markllm",
    "liu2023survey",
    # E. Non-arXiv sources
    "deepmind2024synthidtext",
    "c2pa2024spec",
    "euaiact2024",
]

#: Non-arXiv anchors that must appear in refs.bib (03 section E).
NON_ARXIV_ANCHORS = [
    "10.1038/s41586-024-08025-4",  # SynthID-Text, Nature 638, 625-632
    "c2pa.org/specifications",  # C2PA specification URL
    "2024/1689",  # EU AI Act Regulation (EU) 2024/1689, Art. 50
]


def _read(path: Path) -> str:
    assert path.is_file(), f"missing file: {path}"
    return path.read_text(encoding="utf-8")


def _strip_latex_comments(text: str) -> str:
    """Drop full-line comments and truncate at the first unescaped percent."""
    kept_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        kept: list[str] = []
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            kept.append(ch)
        kept_lines.append("".join(kept))
    return "\n".join(kept_lines)


def _brace_delta(text: str) -> int:
    """Unbalanced-brace count on comment-stripped text (open minus close)."""
    return text.count("{") - text.count("}")


def _bib_entries(text: str) -> dict[str, str]:
    """Map bib key -> full entry body for every entry in *text*.

    Runs on comment-stripped text and scans brace-balanced spans so that
    field values containing braces (e.g. double-braced organization
    authors) do not truncate the body.
    """
    entries: dict[str, str] = {}
    for match in re.finditer(r"@(\w+)\s*\{([^,]+),", text):
        key = match.group(2).strip()
        start = match.start()
        # Scan from the entry's own opening brace so the first field's
        # closing brace does not end the capture early.
        brace_pos = text.find("{", start)
        depth = 0
        for i in range(brace_pos, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    entries[key] = text[start : i + 1]
                    break
    return entries


# ---------------------------------------------------------------------
# Skeleton file inventory
# ---------------------------------------------------------------------


def test_required_paper_files_exist() -> None:
    for name in (
        "main.tex",
        "refs.bib",
        "abstract.tex",
        "ethics.tex",
        "acknowledgments.tex",
        "README.md",
    ):
        assert (PAPER_DIR / name).is_file(), f"missing research/paper/{name}"


# ---------------------------------------------------------------------
# main.tex structure
# ---------------------------------------------------------------------


def test_exact_title() -> None:
    main = _read(PAPER_DIR / "main.tex")
    match = re.search(r"\\title\{([^}]*)\}", main, re.DOTALL)
    assert match is not None, "no \\title{...} found"
    normalized = re.sub(r"\s+", " ", match.group(1)).strip()
    assert normalized == TITLE, f"title mismatch: {normalized!r}"


def test_single_author_placeholder() -> None:
    main = _read(PAPER_DIR / "main.tex")
    assert "\\author{Guillaume Meyer}" in main
    # No affiliation line should be active (only a commented example).
    active = _strip_latex_comments(main)
    assert "\\author{Guillaume Meyer}\\" not in active


def test_sections_in_order() -> None:
    main = _read(PAPER_DIR / "main.tex")
    found = re.findall(r"\\section\{([^}]*)\}", main)
    assert found == SECTIONS, f"section mismatch: {found}"


def test_acl_style_dropin_comment() -> None:
    main = _read(PAPER_DIR / "main.tex")
    assert "acl2024.sty" in main, "missing 'drop in acl2024.sty' comment"


def test_inputs_wired() -> None:
    main = _read(PAPER_DIR / "main.tex")
    for name in ("abstract", "ethics", "acknowledgments"):
        assert f"\\input{{{name}}}" in main, f"missing \\input{{{name}}}"


def test_bibliography_wired() -> None:
    main = _read(PAPER_DIR / "main.tex")
    assert "\\bibliographystyle{plainnat}" in main
    assert "\\bibliography{refs}" in main


def test_table_placeholders() -> None:
    main = _read(PAPER_DIR / "main.tex")
    for number in range(1, 8):
        assert f"\\input{{tables/t{number}}}" in main, f"missing tables/t{number}"
        assert f"tab:t{number}" in main, f"missing label tab:t{number}"


def test_figure_placeholders() -> None:
    main = _read(PAPER_DIR / "main.tex")
    for number in range(1, 7):
        assert f"figures/f{number}" in main, f"missing figures/f{number}"
        assert f"fig:f{number}" in main, f"missing label fig:f{number}"


def test_todo_comments_present() -> None:
    main = _read(PAPER_DIR / "main.tex")
    assert main.count("TODO") >= 10, "expected TODO markers for real numbers"


# ---------------------------------------------------------------------
# refs.bib coverage
# ---------------------------------------------------------------------


def test_bib_keys_match_03_expected_set() -> None:
    bib = _read(PAPER_DIR / "refs.bib")
    entries = _bib_entries(_strip_latex_comments(bib))
    assert set(entries) == set(EXPECTED_BIB_KEYS), (
        f"bib keys diverge from 03: missing={set(EXPECTED_BIB_KEYS) - set(entries)} "
        f"extra={set(entries) - set(EXPECTED_BIB_KEYS)}"
    )


def test_every_arxiv_id_from_03_present() -> None:
    related = _read(RELATED_WORK)
    bib = _read(PAPER_DIR / "refs.bib")
    ids = re.findall(r"arXiv \*\*\d{4}\.\d{5}\*\*", related)
    assert ids, "no arXiv IDs parsed from 03-related-work.md"
    for arxiv_id in ids:
        digits = re.sub(r"[^0-9.]", "", arxiv_id)
        assert digits in bib, f"arXiv id {digits} missing from refs.bib"


def test_non_arxiv_anchors_present() -> None:
    bib = _read(PAPER_DIR / "refs.bib")
    for anchor in NON_ARXIV_ANCHORS:
        assert anchor in bib, f"missing non-arXiv anchor {anchor!r}"


def test_arxiv_entries_use_eprint_fields() -> None:
    bib = _read(PAPER_DIR / "refs.bib")
    entries = _bib_entries(_strip_latex_comments(bib))
    arxiv_keys = [
        k
        for k in EXPECTED_BIB_KEYS
        if k not in ("deepmind2024synthidtext", "c2pa2024spec", "euaiact2024")
    ]
    for key in arxiv_keys:
        assert "eprint" in entries[key], f"{key} is missing eprint"
        assert "archivePrefix" in entries[key], f"{key} is missing archivePrefix"


def test_reverify_comment_present() -> None:
    bib = _read(PAPER_DIR / "refs.bib")
    assert "re-verify at submission" in bib


# ---------------------------------------------------------------------
# Braces and abstract length
# ---------------------------------------------------------------------


def test_braces_balanced() -> None:
    for name in ("main.tex", "abstract.tex", "ethics.tex", "acknowledgments.tex", "refs.bib"):
        text = _read(PAPER_DIR / name)
        delta = _brace_delta(_strip_latex_comments(text))
        assert delta == 0, f"{name}: unbalanced braces (delta={delta})"


def test_abstract_word_count() -> None:
    abstract = _strip_latex_comments(_read(PAPER_DIR / "abstract.tex"))
    words = len(abstract.split())
    assert 120 <= words <= 190, f"abstract word count out of range: {words}"


# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------


def test_readme_build_and_generators() -> None:
    readme = _read(PAPER_DIR / "README.md")
    assert "pdflatex" in readme and "bibtex" in readme
    assert "make_tables.py" in readme and "make_figures.py" in readme
    assert "acl2024.sty" in readme
