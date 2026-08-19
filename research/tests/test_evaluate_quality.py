"""Unit tests for research/scripts/evaluate_quality.py (gap 05-B2).

Pure helpers (Levenshtein, length drift, number/URL survival, row
assembly) are tested directly. Model-dependent code paths are gated on
torch being importable and never touch the network: they use tiny local
models, stubs, and monkeypatched fake modules only.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_quality as eq
import pytest

try:
    import torch
except ImportError:
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


def _long_text(n: int, chunk: str = "abcdefghij ") -> str:
    return (chunk * (n // len(chunk) + 1))[:n]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_levenshtein_distance_known_pairs() -> None:
    assert eq.levenshtein_distance("", "") == 0
    assert eq.levenshtein_distance("abc", "abc") == 0
    assert eq.levenshtein_distance("kitten", "sitting") == 3
    assert eq.levenshtein_distance("a", "") == 1
    assert eq.levenshtein_distance("", "abc") == 3
    assert eq.levenshtein_distance("abc", "ab") == 1


def test_levenshtein_pct_known_pairs() -> None:
    assert eq.levenshtein_pct("", "") == 0.0
    assert eq.levenshtein_pct("abc", "abc") == 0.0
    assert eq.levenshtein_pct("kitten", "sitting") == pytest.approx(50.0)
    assert eq.levenshtein_pct("a", "") == 100.0
    assert eq.levenshtein_pct("abc", "ab") == pytest.approx(100.0 / 3)


def test_truncate_pair_long_original() -> None:
    original = _long_text(5000)
    candidate = "short"
    o, c, truncated = eq.truncate_pair(original, candidate, max_chars=4000)
    assert truncated is True
    assert len(o) == 4000
    assert o == original[:4000]
    assert c == "short"  # shorter text is cut to max_chars but never padded


def test_truncate_pair_short_noop() -> None:
    o, c, truncated = eq.truncate_pair("hello", "world", max_chars=4000)
    assert truncated is False
    assert (o, c) == ("hello", "world")


def test_length_drift() -> None:
    assert eq.length_drift("ab", "abcd") == 1.0
    assert eq.length_drift("ab", "a") == -0.5
    assert eq.length_drift("", "abc") == 3.0
    assert eq.length_drift("hello", "hello") == 0.0


def test_numbers_preserved() -> None:
    assert eq.numbers_preserved("no numbers here", "anything") == 1.0
    assert eq.numbers_preserved("a 42 b 7", "a 42 c") == 0.5
    assert eq.numbers_preserved("a 42 b 7", "x 99") == 0.0
    assert eq.numbers_preserved("12 34", "34 12") == 1.0


def test_urls_preserved() -> None:
    assert eq.urls_preserved("plain text", "anything") == 1.0
    original = "see https://example.com/a and http://x.y/z"
    assert eq.urls_preserved(original, "see https://example.com/a") == 0.5
    assert eq.urls_preserved(original, original) == 1.0
    assert eq.urls_preserved(original, "nothing") == 0.0


def test_ppl_from_loss() -> None:
    assert eq.ppl_from_loss(0.0) == pytest.approx(1.0)
    assert eq.ppl_from_loss(math.log(10.0)) == pytest.approx(10.0)


def test_compute_rouge_l_fmeasure() -> None:
    class _FakeScore:
        fmeasure = 0.75

    assert eq.compute_rouge_l_fmeasure(_FakeScore()) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Row assembly (no models involved: all model metrics skipped)
# ---------------------------------------------------------------------------


def test_compute_row_metrics_all_pure_metrics_present() -> None:
    row = {
        "condition": "c",
        "scheme": "kgw",
        "seed": 1,
        "prompt_idx": 0,
        "attack": "paraphrase",
        "original": "the fox jumps over 42 lazy dogs",  # len 31
        "candidate": "the fox leaps over 42 dogs",  # len 26, lev dist 8
    }
    metrics, notes = eq.compute_row_metrics(
        row, skip={"ppl", "bertscore", "rouge", "sbert"}, device="cpu"
    )
    assert set(metrics) == set(eq.METRIC_KEYS)
    assert metrics["ppl"] is None
    assert metrics["bertscore"] is None
    assert metrics["rouge_l"] is None
    assert metrics["sbert_cosine"] is None
    # metrics are rounded to 4 decimals by the script
    assert metrics["length_drift"] == pytest.approx(round(-5 / 31, 4))
    assert metrics["levenshtein_pct"] == pytest.approx(round(8 / 31 * 100, 4))
    assert metrics["numbers_preserved"] == 1.0
    assert metrics["urls_preserved"] == 1.0
    assert notes == []


def test_compute_row_metrics_missing_texts_all_none() -> None:
    metrics, notes = eq.compute_row_metrics(
        {"original": None, "candidate": "x"}, skip=set(), device="cpu"
    )
    assert set(metrics) == set(eq.METRIC_KEYS)
    assert all(value is None for value in metrics.values())
    assert any("missing" in note for note in notes)


def test_compute_row_metrics_levenshtein_truncation_note() -> None:
    row = {"original": _long_text(5000), "candidate": "short"}
    metrics, notes = eq.compute_row_metrics(
        row, skip={"ppl", "bertscore", "rouge", "sbert"}, device="cpu"
    )
    assert any("truncated" in note for note in notes)
    assert 0.0 <= metrics["levenshtein_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Lazy loaders: failure contract + caching, via monkeypatched imports
# (deterministic, no network, independent of installed packages)
# ---------------------------------------------------------------------------


def test_rouge_loader_records_warning_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    eq._MODEL_CACHE.pop("rouge", None)
    eq._FAILED.discard("rouge")
    monkeypatch.setitem(sys.modules, "rouge_score", None)  # force ImportError
    assert eq._get_rouge_scorer() is None
    assert any(w["metric"] == "rouge" for w in eq._warnings())


def test_ppl_loader_records_warning_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    eq._MODEL_CACHE.pop("ppl", None)
    eq._FAILED.discard("ppl")
    monkeypatch.setitem(sys.modules, "transformers", None)  # force ImportError
    assert eq._get_ppl_models() is None
    assert any(w["metric"] == "ppl" for w in eq._warnings())


def test_failed_loader_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    eq._MODEL_CACHE.pop("sbert", None)
    eq._FAILED.discard("sbert")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert eq._get_sbert_model("cpu") is None
    warning_count = len(eq._warnings())
    assert eq._get_sbert_model("cpu") is None  # cached failure, no new warning
    assert len(eq._warnings()) == warning_count


def test_sbert_loader_caches_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    eq._MODEL_CACHE.pop("sbert", None)
    eq._FAILED.discard("sbert")
    calls: list[tuple[str, str]] = []

    class _FakeST:
        def __init__(self, model_id: str, device: str) -> None:
            calls.append((model_id, device))

    fake = types.SimpleNamespace(SentenceTransformer=_FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    first = eq._get_sbert_model("cpu")
    second = eq._get_sbert_model("cpu")
    assert first is not None and first is second
    assert calls == [("all-MiniLM-L6-v2", "cpu")]


def test_ppl_loader_caches_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    eq._MODEL_CACHE.pop("ppl", None)
    eq._FAILED.discard("ppl")
    calls: list[str] = []

    class _FakeModel:
        def eval(self) -> None:
            calls.append("eval")

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str) -> _FakeTokenizer:
            calls.append(model_id)
            return cls()

    class _FakeAuto:
        @classmethod
        def from_pretrained(cls, model_id: str) -> _FakeModel:
            calls.append(model_id)
            return _FakeModel()

    fake = types.SimpleNamespace(AutoModelForCausalLM=_FakeAuto, AutoTokenizer=_FakeTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    first = eq._get_ppl_models()
    second = eq._get_ppl_models()
    assert first is not None and first is second
    assert len(first) == 2  # (model, tokenizer)
    assert calls.count("gpt2-large") == 2  # tokenizer + model
    assert "eval" in calls


# ---------------------------------------------------------------------------
# CLI end-to-end (model metrics skipped: runs without torch)
# ---------------------------------------------------------------------------


def test_main_end_to_end(tmp_path: Path) -> None:
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    warn = tmp_path / "warnings.jsonl"
    rows = [
        {
            "condition": "c",
            "scheme": "kgw",
            "seed": 1,
            "prompt_idx": 0,
            "attack": "none",
            "original": "a 42 b",
            "candidate": "a 42 c",
        },
        {
            "condition": "c",
            "scheme": "kgw",
            "seed": 2,
            "prompt_idx": 1,
            "attack": "none",
            "original": "hello https://example.com/x",
            "candidate": "hello",
        },
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    rc = eq.main(
        [
            "--input",
            str(inp),
            "--out",
            str(out),
            "--skip",
            "ppl,bertscore,rouge,sbert",
            "--warnings-out",
            str(warn),
        ]
    )
    assert rc == 0

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    for row in written:
        assert set(eq.METRIC_KEYS) <= set(row)
        assert row["ppl"] is None
        assert row["bertscore"] is None
        assert row["rouge_l"] is None
        assert row["sbert_cosine"] is None
    assert written[0]["numbers_preserved"] == 1.0
    assert written[1]["numbers_preserved"] == 1.0  # no numbers -> preserved
    assert written[1]["urls_preserved"] == 0.0  # URL dropped
    # warnings file is valid JSONL (possibly empty)
    assert all(line.strip() for line in warn.read_text(encoding="utf-8").splitlines())


def test_main_respects_limit(tmp_path: Path) -> None:
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [{"original": f"text {i}", "candidate": f"text {i}"} for i in range(3)]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    rc = eq.main(
        [
            "--input",
            str(inp),
            "--out",
            str(out),
            "--skip",
            "ppl,bertscore,rouge,sbert",
            "--limit",
            "2",
        ]
    )
    assert rc == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2


# ---------------------------------------------------------------------------
# Model-dependent math (torch-gated; offline: tiny local models / stubs)
# ---------------------------------------------------------------------------


@needs_torch
def test_compute_ppl_offline_small_model() -> None:
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        pytest.skip("transformers not installed")

    class _StubTokenizer:
        """Maps chars to ids < vocab_size; mimics the HF tokenizer contract."""

        def __init__(self, vocab_size: int) -> None:
            self.vocab_size = vocab_size

        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            ids = [ord(ch) % self.vocab_size for ch in text][:max_length]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }

    config = GPT2Config(vocab_size=64, n_positions=128, n_embd=16, n_layer=1, n_head=1)
    model = GPT2LMHeadModel(config).eval()
    ppl = eq.compute_ppl(model, _StubTokenizer(64), "hello world", device="cpu", max_length=16)
    assert math.isfinite(ppl)
    assert ppl > 0.0


@needs_torch
def test_compute_bertscore_f1() -> None:
    f1 = torch.tensor([0.847])
    assert eq.compute_bertscore_f1(f1) == pytest.approx(0.847)


@needs_torch
def test_compute_sbert_cosine_identical() -> None:
    emb = torch.tensor([1.0, 2.0, 3.0])
    assert eq.compute_sbert_cosine(emb, emb) == pytest.approx(1.0)


@needs_torch
def test_compute_sbert_cosine_orthogonal() -> None:
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert eq.compute_sbert_cosine(a, b) == pytest.approx(0.0, abs=1e-6)
