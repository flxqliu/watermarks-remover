"""Tests for research/scripts/make_tables.py and make_figures.py (gap 05-B3).

Runs the generators against synthetic and empty results dirs using only
pytest + stdlib (no matplotlib required): the figures script is exercised
with the matplotlib import mocked to fail.

Run with:  python3 -m pytest research/tests/test_make_tables.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_figures
import make_tables


def _run_tables(results: Path, out: Path) -> int:
    return make_tables.main(["--results-dir", str(results), "--out-dir", str(out)])


def _write_metrics(cond: Path, attacks: dict) -> None:
    cond.mkdir(parents=True, exist_ok=True)
    (cond / "metrics.json").write_text(json.dumps(attacks), encoding="utf-8")


def _synthetic_metrics() -> dict:
    """A minimal per-attack metrics dict covering all four T2 columns."""
    return {
        "none": {"auroc": 0.990, "tpr_at_fpr": {"0.01": 0.900}},
        "layerA": {"auroc": 0.970, "tpr_at_fpr": {"0.01": 0.870}},
        "paraphrase:3": {"auroc": 0.610, "tpr_at_fpr": {"0.01": 0.400}},
        "layerA+paraphrase:3": {"auroc": 0.520, "tpr_at_fpr": {"0.01": 0.310}},
    }


def _make_synthetic_results(root: Path) -> None:
    _write_metrics(root / "kgw-d2-L300-T0.7-en-s1-p0", _synthetic_metrics())
    _write_metrics(root / "synthid-L300-T0.7-en-s1-p0", _synthetic_metrics())


def test_static_tables_with_empty_results_dir(tmp_path) -> None:
    """T1 (taxonomy) and T7 (baseline template) need no results data."""
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    assert _run_tables(results, out) == 0
    for name in ("t1", "t2", "t3", "t4", "t5", "t6", "t7"):
        assert (out / "tables" / f"{name}.tex").is_file()
    md = (out / "tables" / "tables.md").read_text(encoding="utf-8")
    assert "### T1" in md and "### T7" in md
    t1 = (out / "tables" / "t1.tex").read_text(encoding="utf-8")
    assert "clean\\_text.py" in t1  # LaTeX-escaped underscore
    assert "Layer A" in t1
    assert "layered contribution" in t1  # capitalized in the taxonomy row
    assert all(f"A{i}" in t1 for i in range(9))  # A0..A8 present
    assert "clean_text.py" in md  # unescaped in markdown
    t7 = (out / "tables" / "t7.tex").read_text(encoding="utf-8")
    assert "TBD (re-verify at submission)" in t7
    assert "\\cite{" in t7


def test_data_dependent_tables_degrade_to_no_data(tmp_path) -> None:
    """T2-T6 emit an explicit 'no data' row when inputs are missing."""
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    assert _run_tables(results, out) == 0
    for name in ("t2", "t3", "t4", "t5", "t6"):
        tex = (out / "tables" / f"{name}.tex").read_text(encoding="utf-8")
        assert "no data" in tex
    md = (out / "tables" / "tables.md").read_text(encoding="utf-8")
    assert "no data" in md


def test_t2_matrix_reads_metrics(tmp_path) -> None:
    """T2 maps attacks to the pre/A-only/B-only/A+B columns from metrics.json."""
    results = tmp_path / "results"
    _make_synthetic_results(results)
    out = tmp_path / "out"
    assert _run_tables(results, out) == 0
    t2 = (out / "tables" / "t2.tex").read_text(encoding="utf-8")
    assert "0.990" in t2 and "0.900" in t2
    assert "KGW" in t2 and "SynthID-Text" in t2
    assert "pre-attack" in t2 and "A+B" in t2
    # B-family row: its own value lands in the B-only column.
    assert "A3 paraphrase:3 & 0.990 & 0.970 & 0.610 & ---" in t2
    # Layered row: its own value lands in the A+B column.
    assert "A8 layerA+paraphrase:3 & 0.990 & 0.970 & --- & 0.520" in t2
    md = (out / "tables" / "tables.md").read_text(encoding="utf-8")
    assert "0.610" in md


def test_t6_attack_cost_from_attacked_jsonl(tmp_path) -> None:
    """T6 normalizes tokens/seconds/USD per 1k words from attacked.jsonl."""
    results = tmp_path / "results"
    cond = results / "kgw-d2-L300-T0.7-en-s1-p0"
    cond.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "attack": "paraphrase:3",
            "original": "word " * 100,
            "candidate": "word " * 95,
            "stats": {"tokens_in": 300, "tokens_out": 250},
            "seconds": 12.5,
            "usd": 0.0015,
        },
    ]
    (cond / "attacked.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = tmp_path / "out"
    assert _run_tables(results, out) == 0
    t6 = (out / "tables" / "t6.tex").read_text(encoding="utf-8")
    assert "paraphrase:3" in t6
    assert "3,000" in t6  # 300 tokens / 100 words * 1000
    assert "0.0150" in t6  # 0.0015 USD / 100 words * 1000


def test_figures_exit_zero_when_matplotlib_missing(tmp_path, monkeypatch, capsys) -> None:
    """With the matplotlib import mocked, the figures script warns and exits 0."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    code = make_figures.main(["--results-dir", str(results), "--out-dir", str(out)])
    assert code == 0
    captured = capsys.readouterr()
    assert "matplotlib" in (captured.out + captured.err).lower()
    figs = list((out / "figures").glob("f*")) if (out / "figures").is_dir() else []
    assert figs == []


def test_figures_render_with_synthetic_data(tmp_path) -> None:
    """All six figures are emitted when matplotlib and data are available."""
    pytest.importorskip("matplotlib")
    results = tmp_path / "results"
    cond = results / "kgw-d2-L300-T0.7-en-s1-p0"
    metrics = _synthetic_metrics()
    metrics["none"]["roc_points"] = {"fpr": [0.0, 0.01, 1.0], "tpr": [0.0, 0.9, 1.0]}
    metrics["layerA+paraphrase:3"]["roc_points"] = {
        "fpr": [0.0, 0.01, 1.0],
        "tpr": [0.0, 0.31, 1.0],
    }
    _write_metrics(cond, metrics)
    (cond / "quality.jsonl").write_text(
        json.dumps({"attack": "paraphrase:3", "ppl": 24.5, "ppl_delta": 2.1})
        + "\n"
        + json.dumps({"attack": "none", "ppl": 22.4})
        + "\n",
        encoding="utf-8",
    )
    (cond / "attacked.jsonl").write_text(
        json.dumps(
            {
                "attack": "layerA+paraphrase:3",
                "original": "alpha beta " * 50,
                "candidate": "gamma delta " * 40,
                "score_before": 2.3,
                "score_after": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_metrics(results / "synthid-L300-T0.7-en-s1-p0", _synthetic_metrics())
    out = tmp_path / "out"
    code = make_figures.main(["--results-dir", str(results), "--out-dir", str(out)])
    assert code == 0
    for i in range(1, 7):
        assert (out / "figures" / f"f{i}.png").is_file()
