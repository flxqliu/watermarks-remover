"""Tests for the Panoptes adapter contract (panoptes_adapter.py + manifest)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from text_unicode import clean_text

import panoptes_adapter

_ADAPTER_ENV = (
    "WATERMARKS_REMOVER_ADAPTER_LAYER_B",
    "WATERMARKS_REWRITE_BACKEND",
    "WATERMARKS_REWRITE_MODEL",
    "WATERMARKS_REWRITE_BASE_URL",
    "WATERMARKS_REWRITE_API_KEY",
    "WATERMARKS_REWRITE_ALLOW_REMOTE",
    "WATERMARKS_REWRITE_REASONING_EFFORT",
    "WATERMARKS_REWRITE_CANDIDATES",
    "WATERMARKS_REWRITE_LOOPS",
)

# Invisible carriers, kept as explicit escapes so they stay visible to reviewers.
ZWSP = chr(0x200B)
SOFT_HYPHEN = chr(0x00AD)
EM_SPACE = chr(0x2003)
IDEOGRAPHIC_SPACE = chr(0x3000)
TAG_LATIN_A = chr(0xE0041)  # TAG LATIN CAPITAL LETTER A
RLO = chr(0x202E)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ADAPTER_ENV:
        monkeypatch.delenv(name, raising=False)


# Cases mirror tests/test_clean_text.py.
PARITY_CASES = [
    f"Hello{ZWSP}World{SOFT_HYPHEN}!",
    f"a{EM_SPACE}b{IDEOGRAPHIC_SPACE}c",
    f"hi{TAG_LATIN_A}there",
    f"ab{RLO}ef",  # override: stripped by default
    # Legitimate bidi marks (LRI U+2066, PDI U+2069, RLM U+200F): preserved by default.
    "السعر " + chr(0x2066) + "123 USD" + chr(0x2069) + chr(0x200F),
    "plain ASCII stays put",
    (
        f"The committee{ZWSP}approved the{SOFT_HYPHEN}budget at{EM_SPACE}14:00.\n"
        f"Minutes{ZWSP}were{TAG_LATIN_A}circulated."
    ),
]


def test_manifest_is_valid_and_resolves() -> None:
    data = json.loads((ROOT / "panoptes.adapter.json").read_text(encoding="utf-8"))
    assert data["kind"] == "watermark-remover"
    assert data["requires_network"] is False
    entry = data["entry"]
    assert entry["module"] == "panoptes_adapter"
    module = importlib.import_module(entry["module"])
    assert callable(getattr(module, entry["callable"]))


@pytest.mark.parametrize("raw", PARITY_CASES)
def test_transform_matches_clean_text(raw: str) -> None:
    expected, _stats = clean_text(raw)
    assert panoptes_adapter.transform(raw) == expected


def test_transform_strips_invisible_watermark_carriers() -> None:
    raw = f"hi{TAG_LATIN_A} the{ZWSP}re"
    assert panoptes_adapter.transform(raw) == "hi there"


def test_transform_idempotent() -> None:
    for raw in PARITY_CASES:
        once = panoptes_adapter.transform(raw)
        assert panoptes_adapter.transform(once) == once


def test_transform_preserves_legitimate_bidi_marks() -> None:
    raw = "السعر " + chr(0x2066) + "123 USD" + chr(0x2069) + chr(0x200F)
    assert panoptes_adapter.transform(raw) == raw


def test_transform_empty_string() -> None:
    assert panoptes_adapter.transform("") == ""


def test_transform_large_input() -> None:
    base = "The quick brown fox jumps over the lazy dog. "
    raw = base * 24_000 + ZWSP * 100  # ~1 MiB with trailing zero-width marks
    expected, _stats = clean_text(raw)
    out = panoptes_adapter.transform(raw)
    assert out == expected
    assert ZWSP not in out


def test_layer_b_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "llama3.2")

    def _boom(text: str, **kwargs: object) -> tuple[str, dict]:
        raise AssertionError("rewrite must not run unless Layer B is opted in")

    monkeypatch.setitem(sys.modules, "rewrite_text", SimpleNamespace(rewrite=_boom))
    assert panoptes_adapter.transform(f"a{ZWSP}b") == "ab"


def _install_fake_rewrite(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _fake(text: str, **kwargs: object) -> tuple[str, dict]:
        calls.append({"text": text, **kwargs})
        return "REWRITTEN", {}

    monkeypatch.setitem(sys.modules, "rewrite_text", SimpleNamespace(rewrite=_fake))
    return calls


def test_layer_b_opt_in_routes_to_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATERMARKS_REMOVER_ADAPTER_LAYER_B", "1")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "llama3.2")
    calls = _install_fake_rewrite(monkeypatch)

    assert panoptes_adapter.transform(f"a{ZWSP}b") == "REWRITTEN"
    assert len(calls) == 1
    call = calls[0]
    assert call["text"] == "ab"  # Layer A runs first
    assert call["backend"] == "ollama"
    assert call["model"] == "llama3.2"
    assert call["layer_a_after"] is True
    assert call["allow_remote"] is False
    assert call["candidates"] == 1
    assert call["max_loops"] == 1
    assert call["reasoning_effort"] == "none"
    assert call["api_key"] is None


def test_layer_b_maps_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATERMARKS_REMOVER_ADAPTER_LAYER_B", "true")
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "sk-test")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    monkeypatch.setenv("WATERMARKS_REWRITE_REASONING_EFFORT", "off")
    monkeypatch.setenv("WATERMARKS_REWRITE_CANDIDATES", "3")
    monkeypatch.setenv("WATERMARKS_REWRITE_LOOPS", "2")
    calls = _install_fake_rewrite(monkeypatch)

    panoptes_adapter.transform("x")
    call = calls[0]
    assert call["backend"] == "openai-compatible"
    assert call["model"] == "deepseek-v4-flash"
    assert call["base_url"] == "https://api.deepseek.com"
    assert call["api_key"] == "sk-test"
    assert call["allow_remote"] is True
    assert call["reasoning_effort"] is None  # "off" omits the parameter
    assert call["candidates"] == 3
    assert call["max_loops"] == 2


def test_layer_b_rejects_print_prompt_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATERMARKS_REMOVER_ADAPTER_LAYER_B", "1")
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "print-prompt")
    with pytest.raises(RuntimeError, match="print-prompt"):
        panoptes_adapter.transform("x")


def test_layer_b_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATERMARKS_REMOVER_ADAPTER_LAYER_B", "1")
    with pytest.raises(RuntimeError, match="WATERMARKS_REWRITE_MODEL"):
        panoptes_adapter.transform("x")
