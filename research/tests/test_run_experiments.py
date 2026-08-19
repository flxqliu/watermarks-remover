"""Integration tests for the v1 experiment orchestrator (gap 05-A4).

These exercise the design constants, the locked matrix, results helpers,
and the report stage without spawning MarkLLM/API workers (the full run
is a multi-week CPU/API job driven by the smoke test in 01 §9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_experiments as rx


def test_locked_matrix_sizes() -> None:
    cells = list(
        rx.iter_cells(
            list(rx.SCHEMES),
            rx.LENGTHS,
            rx.TEMPS,
            rx.LANGUAGES,
            rx.SEEDS,
            rx.PROMPTS,
        )
    )
    assert len(cells) == 3500


def test_conditions_count() -> None:
    conds = list(rx.iter_conditions(list(rx.SCHEMES), rx.LENGTHS, rx.TEMPS, rx.LANGUAGES))
    # 14 EN core + 4 temp axis + 4 length axis + 6 multilingual.
    assert len(conds) == 28


def test_cell_allowed_restrictions() -> None:
    # EN core: every scheme at length 100/300, temp 0.7.
    for scheme in rx.SCHEMES:
        assert rx.cell_allowed(scheme, 100, 0.7, "en")
        assert rx.cell_allowed(scheme, 300, 0.7, "en")
    # temp axis: core 4 only.
    assert rx.cell_allowed("synthid", 300, 1.0, "en")
    assert not rx.cell_allowed("sir", 300, 1.0, "en")
    assert not rx.cell_allowed("exp", 300, 1.0, "en")
    # length axis: core 4 only.
    assert rx.cell_allowed("kgw-d1", 500, 0.7, "en")
    assert not rx.cell_allowed("unigram", 500, 0.7, "en")
    # multilingual: kgw-d2 + synthid only.
    assert rx.cell_allowed("kgw-d2", 300, 0.7, "de")
    assert rx.cell_allowed("synthid", 300, 0.7, "fr")
    assert not rx.cell_allowed("kgw-d1", 300, 0.7, "es")
    assert not rx.cell_allowed("sir", 300, 0.7, "de")


def test_scheme_cli_map_covers_all_schemes() -> None:
    assert set(rx.SCHEME_CLI) == set(rx.SCHEMES)
    assert rx.SCHEME_CLI["kgw-d1"] == "kgw"
    assert rx.SCHEME_CLI["synthid"] == "synthid"
    assert rx.SCHEME_CLI["sir"] == "sir"


def test_cheap_expands_to_three_subattacks() -> None:
    assert rx._attack_rows("cheap") == [
        "cheap:synonym",
        "cheap:delete",
        "cheap:reorder",
    ]
    assert rx._attack_rows("layerA") == ["layerA"]


def test_configs_exist_for_every_scheme(tmp_path) -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    for scheme in rx.SCHEMES:
        cfg = cfg_dir / rx.SCHEMES[scheme][2]
        assert cfg.is_file(), f"missing {cfg}"
        data = json.loads(cfg.read_text("utf-8"))
        assert data["algorithm_name"] == rx.SCHEMES[scheme][1]


def test_quality_pairs_skip_none_and_respect_cap(tmp_path) -> None:
    rows = [
        {
            "ok": True,
            "seed": 1,
            "prompt_idx": 1,
            "attack": "none",
            "original": "a",
            "candidate": "a",
        },
        {
            "ok": True,
            "seed": 2,
            "prompt_idx": 2,
            "attack": "layerA",
            "original": "b",
            "candidate": "c",
        },
        {
            "ok": True,
            "seed": 3,
            "prompt_idx": 3,
            "attack": "paraphrase:3",
            "original": "d",
            "candidate": "e",
        },
        {
            "ok": False,
            "seed": 4,
            "prompt_idx": 4,
            "attack": "humanize",
            "original": "f",
            "candidate": "g",
        },
    ]
    att = tmp_path / "attacked.jsonl"
    att.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cond = rx.Condition("kgw-d2", 300, 0.7, "en")
    pairs = rx._quality_pairs(cond, att, cap=2)
    assert len(pairs) == 2
    assert {p["attack"] for p in pairs} == {"layerA", "paraphrase:3"}
    assert all(p["condition"] == cond.id for p in pairs)


def test_report_writes_manifest_and_markdown(tmp_path) -> None:
    class FakeArgs:
        rewrite_backend = "openai-compatible"
        rewrite_model = "m"
        force = False
        corpus_dir = "research/corpus"

    args = FakeArgs()
    from run_experiments import RunContext

    ctx = RunContext(args, tmp_path / "markllm")  # upstream not needed for report
    ctx.upstream = tmp_path / "markllm"  # avoid resolving the real checkout
    out = tmp_path / "results"
    out.mkdir()
    rx.stage_report(out, ctx)
    assert (out / "manifest.json").is_file()
    assert (out / "report.md").is_file()
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["design"]["prompts"] == 25
    assert set(manifest["pins"]) >= {"repo_commit", "markllm_commit"}
