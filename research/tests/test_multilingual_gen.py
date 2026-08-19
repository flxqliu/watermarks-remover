"""Unit tests for research/scripts/multilingual_gen.py (gap 05-A5).

Covers the chat-template formatting (fake tokenizer), arg parsing, the
watermark/serve JSON schemas (MarkLLM backend monkeypatched away), and the
seeding path. Model/torch-dependent tests are guarded: torch imports happen
in try/except and torch-requiring tests are marked skipif so the suite runs
in the repo venv (pytest, no torch) without network access.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import multilingual_gen as mg

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch is not installed in this environment"
)


class _FakeTokenizer:
    """Minimal stand-in: chat_template attribute + apply_chat_template."""

    def __init__(
        self, chat_template: str | None = "<fake template>", formatted: str = "<FORMATTED>"
    ):
        self.chat_template = chat_template
        self.formatted = formatted
        self.calls: list[tuple[list, dict]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append(
            (messages, {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt})
        )
        return self.formatted


class _FakeWM:
    """Stand-in watermark instance: config.gen_kwargs + generate/detect methods."""

    def __init__(self, tokenizer: object | None = None):
        self.config = SimpleNamespace(
            gen_kwargs={},
            generation_tokenizer=tokenizer
            if tokenizer is not None
            else _FakeTokenizer(chat_template=None),
        )
        self.detect_calls: list[tuple[str, bool]] = []

    def generate_watermarked_text(self, prompt):
        return "WM:" + prompt

    def generate_unwatermarked_text(self, prompt):
        return "UW:" + prompt

    def detect_watermark(self, text, return_dict=True):
        self.detect_calls.append((text, return_dict))
        return {"is_watermarked": True, "score": 3.5}


def _make_upstream(tmp_path: Path) -> Path:
    """A fake MarkLLM checkout (only the watermark/ dir main() checks)."""
    upstream = tmp_path / "MarkLLM"
    (upstream / "watermark").mkdir(parents=True)
    return upstream


def _watermark_argv(upstream: Path, config: Path, prompt: Path, **overrides) -> list[str]:
    argv = [
        "watermark",
        "--markllm-dir",
        str(upstream),
        "--scheme",
        "kgw",
        "--config",
        str(config),
        "--prompt",
        str(prompt),
        "--seed",
        "3",
        "--max-new-tokens",
        "300",
        "--temperature",
        "0.7",
        "--top-p",
        "0.95",
        "--lang",
        "de",
        "--doc-id",
        "de-1",
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        argv[argv.index(flag) + 1] = str(value)
    return argv


def _patch_backend(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """Replace _load_algorithm/_generate so no torch/MarkLLM is needed."""

    def fake_load(
        upstream, alg, config, model, device, offline=False, temperature=None, top_p=None
    ):
        captured["load"] = {
            "alg": alg,
            "model": model,
            "device": device,
            "offline": offline,
            "temperature": temperature,
            "top_p": top_p,
        }
        return _FakeWM(tokenizer=_FakeTokenizer())

    def fake_generate(wm, prompt, seed, max_new_tokens, min_length=0, need_unwatermarked=True):
        captured["generate"] = {
            "prompt": prompt,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
        }
        return "WM|" + prompt, "UW|" + prompt

    monkeypatch.setattr(mg, "_load_algorithm", fake_load)
    monkeypatch.setattr(mg, "_generate", fake_generate)


# --------------------------------------------------------------------------
# Chat-template formatting (no torch, no network)
# --------------------------------------------------------------------------


def test_chat_template_formats_single_user_turn():
    tok = _FakeTokenizer(
        chat_template="{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
    )
    out = mg.apply_chat_template(tok, "Erzähle mir über Berlin.")
    assert out == "<FORMATTED>"
    (messages, kwargs) = tok.calls[0]
    assert messages == [{"role": "user", "content": "Erzähle mir über Berlin."}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}


@pytest.mark.parametrize("template", [None, ""])
def test_chat_template_returns_raw_prompt_when_unset(template):
    tok = _FakeTokenizer(chat_template=template)
    assert mg.apply_chat_template(tok, "raw prompt") == "raw prompt"
    assert tok.calls == []


def test_chat_template_missing_attribute_returns_raw_prompt():
    class _NoTemplate:
        pass

    assert mg.apply_chat_template(_NoTemplate(), "raw prompt") == "raw prompt"


# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------


def test_watermark_parser_full_args():
    args = mg.build_parser().parse_args(
        _watermark_argv(Path("/fake/MarkLLM"), Path("/fake/cfg.json"), Path("/fake/p.txt"))
    )
    assert args.cmd == "watermark"
    assert args.scheme == "kgw"
    assert args.seed == 3
    assert args.max_new_tokens == 300
    assert args.temperature == 0.7
    assert args.top_p == 0.95
    assert args.lang == "de"
    assert args.doc_id == "de-1"
    assert args.device == "auto"
    assert args.model == mg.DEFAULT_MODEL  # --model omitted -> Qwen2.5-1.5B


def test_watermark_parser_model_override():
    argv = _watermark_argv(Path("/fake/MarkLLM"), Path("/fake/cfg.json"), Path("/fake/p.txt"))
    argv += ["--model", "Qwen/Qwen2.5-0.5B-Instruct"]
    args = mg.build_parser().parse_args(argv)
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.parametrize(
    "missing",
    ["--scheme", "--config", "--prompt", "--seed", "--max-new-tokens", "--lang", "--doc-id"],
)
def test_watermark_parser_requires_args(missing):
    argv = _watermark_argv(Path("/fake/MarkLLM"), Path("/fake/cfg.json"), Path("/fake/p.txt"))
    argv.remove(missing)
    with pytest.raises(SystemExit):
        mg.build_parser().parse_args(argv)


def test_watermark_parser_rejects_non_multilingual_lang():
    argv = _watermark_argv(Path("/fake/MarkLLM"), Path("/fake/cfg.json"), Path("/fake/p.txt"))
    argv[argv.index("--lang") + 1] = "en"
    with pytest.raises(SystemExit):
        mg.build_parser().parse_args(argv)


def test_serve_parser_common_args():
    args = mg.build_parser().parse_args(
        [
            "serve",
            "--markllm-dir",
            "/fake/MarkLLM",
            "--scheme",
            "synthid",
            "--config",
            "/fake/c.json",
        ]
    )
    assert args.cmd == "serve"
    assert args.scheme == "synthid"
    assert args.model == mg.DEFAULT_MODEL


# --------------------------------------------------------------------------
# watermark subcommand: JSON schema on stdout
# --------------------------------------------------------------------------


def test_cmd_watermark_json_schema(monkeypatch, tmp_path, capsys):
    upstream = _make_upstream(tmp_path)
    config = tmp_path / "cfg.json"
    config.write_text('{"algorithm_name": "KGW"}', encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Erzähle mir über Berlin.\n", encoding="utf-8")
    captured: dict = {}
    _patch_backend(monkeypatch, captured)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # pins.hf_revision fails fast, no network

    rc = mg.main(_watermark_argv(upstream, config, prompt))

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1  # exactly one JSON object on stdout
    out = json.loads(lines[0])
    assert out["ok"] is True
    assert out["doc_id"] == "de-1"
    assert out["lang"] == "de"
    assert out["seed"] == 3
    assert out["model"] == mg.DEFAULT_MODEL
    assert out["scheme"] == "kgw"
    assert out["config"] == str(config.resolve())
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.95
    assert out["watermarked"] == "WM|<FORMATTED>"  # chat template applied
    assert out["unwatermarked"] == "UW|<FORMATTED>"
    assert isinstance(out["pins"], dict)
    assert "markllm_commit" in out["pins"]
    assert "hf_revision" in out["pins"]
    assert out["pins"]["hf_revision"] is None  # offline -> no hub call
    # The MarkLLM prompt was the chat-formatted string, not the raw prompt.
    assert captured["generate"]["prompt"] == "<FORMATTED>"
    assert captured["generate"]["seed"] == 3
    assert captured["generate"]["max_new_tokens"] == 300
    # temperature/top_p were folded into the load call (like the service).
    assert captured["load"]["temperature"] == 0.7
    assert captured["load"]["top_p"] == 0.95


def test_cmd_watermark_omitted_temp_top_p_default_none(monkeypatch, tmp_path, capsys):
    upstream = _make_upstream(tmp_path)
    config = tmp_path / "cfg.json"
    config.write_text('{"algorithm_name": "KGW"}', encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hola, ¿qué tal?", encoding="utf-8")
    captured: dict = {}
    _patch_backend(monkeypatch, captured)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    argv = _watermark_argv(upstream, config, prompt)
    for flag in ("--temperature", "--top-p"):
        idx = argv.index(flag)
        del argv[idx : idx + 2]  # remove flag and its value
    argv[argv.index("--lang") + 1] = "es"
    argv[argv.index("--doc-id") + 1] = "es-1"

    rc = mg.main(argv)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["temperature"] is None
    assert out["top_p"] is None
    assert captured["load"]["temperature"] is None
    assert captured["load"]["top_p"] is None


def test_cmd_watermark_missing_prompt_file(monkeypatch, tmp_path, capsys):
    upstream = _make_upstream(tmp_path)
    config = tmp_path / "cfg.json"
    config.write_text('{"algorithm_name": "KGW"}', encoding="utf-8")
    argv = _watermark_argv(upstream, config, tmp_path / "nope.txt")

    rc = mg.main(argv)
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert "not a file" in out["error"]
    assert out["doc_id"] == "de-1"


def test_main_missing_markllm_dir(monkeypatch, capsys):
    monkeypatch.delenv("MARKLLM_DIR", raising=False)
    argv = _watermark_argv(Path("/nonexistent/MarkLLM"), Path("/fake/c.json"), Path("/fake/p.txt"))
    rc = mg.main(argv)
    assert rc == 3
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert "MarkLLM not configured" in out["error"]


# --------------------------------------------------------------------------
# serve subcommand: JSON-lines protocol
# --------------------------------------------------------------------------


def test_handle_serve_request_watermark_applies_kwargs():
    wm = _FakeWM()
    resp = mg._handle_serve_request(
        wm,
        {
            "op": "watermark",
            "id": 7,
            "prompt": "Bonjour Paris",
            "seed": None,
            "max_new_tokens": 100,
            "min_length": 5,
            "temperature": 0.8,
            "top_p": 0.9,
        },
    )
    assert resp["ok"] is True
    assert resp["id"] == 7
    assert resp["watermarked"] == "WM:Bonjour Paris"
    assert resp["unwatermarked"] == "UW:Bonjour Paris"
    assert wm.config.gen_kwargs["max_new_tokens"] == 100
    assert wm.config.gen_kwargs["min_length"] == 5
    assert wm.config.gen_kwargs["temperature"] == 0.8
    assert wm.config.gen_kwargs["top_p"] == 0.9


def test_handle_serve_request_bad_prompt_keeps_worker_alive():
    wm = _FakeWM()
    resp = mg._handle_serve_request(wm, {"op": "watermark", "id": 9, "prompt": ""})
    assert resp == {"ok": False, "id": 9, "error": "'prompt' must be a non-empty string"}


def test_handle_serve_request_unknown_op():
    wm = _FakeWM()
    resp = mg._handle_serve_request(wm, {"op": "frobnicate", "id": 1})
    assert resp["ok"] is False
    assert resp["id"] == 1
    assert "unknown op" in resp["error"]


def test_handle_serve_request_exit():
    wm = _FakeWM()
    assert mg._handle_serve_request(wm, {"op": "exit", "id": 0}) == {"ok": True, "id": 0}


def test_handle_serve_request_detect():
    wm = _FakeWM()
    resp = mg._handle_serve_request(
        wm, {"op": "detect", "id": 5, "text": "Hola Madrid"}, threshold=0.52
    )
    assert resp == {
        "ok": True,
        "id": 5,
        "is_watermarked": True,
        "score": 3.5,
        "threshold": 0.52,
    }
    assert wm.detect_calls == [("Hola Madrid", True)]  # return_dict=True


@pytest.mark.parametrize("bad", [None, "", 42])
def test_handle_serve_request_detect_bad_text(bad):
    wm = _FakeWM()
    resp = mg._handle_serve_request(wm, {"op": "detect", "id": 6, "text": bad})
    assert resp["ok"] is False
    assert resp["id"] == 6
    assert "non-empty string" in resp["error"]


def test_handle_serve_request_detect_score_not_float():
    class _OddWM(_FakeWM):
        def detect_watermark(self, text, return_dict=True):
            return {"is_watermarked": False, "score": "n/a"}

    resp = mg._handle_serve_request(_OddWM(), {"op": "detect", "id": 8, "text": "x"})
    assert resp == {"ok": True, "id": 8, "is_watermarked": False, "score": None, "threshold": None}


def test_threshold_from_config(tmp_path):
    kgw = tmp_path / "kgw.json"
    kgw.write_text('{"algorithm_name": "KGW", "z_threshold": 4.0}', encoding="utf-8")
    assert mg._threshold_from_config(kgw) == 4.0
    synthid = tmp_path / "synthid.json"
    synthid.write_text('{"algorithm_name": "SynthID", "threshold": 0.52}', encoding="utf-8")
    assert mg._threshold_from_config(synthid) == 0.52
    no_threshold = tmp_path / "none.json"
    no_threshold.write_text('{"algorithm_name": "KGW"}', encoding="utf-8")
    assert mg._threshold_from_config(no_threshold) is None
    assert mg._threshold_from_config(tmp_path / "missing.json") is None


def test_cmd_serve_jsonlines_protocol(monkeypatch, tmp_path, capsys):
    upstream = _make_upstream(tmp_path)
    config = tmp_path / "cfg.json"
    config.write_text('{"algorithm_name": "SynthID"}', encoding="utf-8")
    monkeypatch.setattr(mg, "_load_algorithm", lambda *a, **k: _FakeWM())

    requests = [
        {"op": "watermark", "id": 1, "prompt": "Hola Madrid"},
        {"op": "watermark", "id": 2, "prompt": ""},  # bad request: worker must live on
        {"op": "exit", "id": 3},
    ]
    stream = "\n".join(json.dumps(r) for r in requests) + "\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stream))

    argv = ["serve", "--markllm-dir", str(upstream), "--scheme", "synthid", "--config", str(config)]
    rc = mg.main(argv)

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4  # ready + 3 responses
    ready = json.loads(lines[0])
    assert ready["ready"] is True
    assert ready["scheme"] == "synthid"
    assert ready["model"] == mg.DEFAULT_MODEL
    assert ready["device"] in ("cpu", "cuda")  # auto; never mps
    r1 = json.loads(lines[1])
    assert r1["ok"] is True and r1["id"] == 1
    assert r1["watermarked"] == "WM:Hola Madrid"
    r2 = json.loads(lines[2])
    assert r2["ok"] is False and r2["id"] == 2
    r3 = json.loads(lines[3])
    assert r3["ok"] is True and r3["id"] == 3


def test_cmd_serve_detect_request(monkeypatch, tmp_path, capsys):
    upstream = _make_upstream(tmp_path)
    config = tmp_path / "cfg.json"
    config.write_text('{"algorithm_name": "SynthID", "threshold": 0.52}', encoding="utf-8")
    monkeypatch.setattr(mg, "_load_algorithm", lambda *a, **k: _FakeWM())

    requests = [
        {"op": "watermark", "id": 1, "prompt": "Hola Madrid"},
        {"op": "detect", "id": 2, "text": "Hola Madrid"},
        {"op": "detect", "id": 3, "text": ""},  # bad detect: worker must live on
        {"op": "exit", "id": 4},
    ]
    stream = "\n".join(json.dumps(r) for r in requests) + "\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stream))

    argv = ["serve", "--markllm-dir", str(upstream), "--scheme", "synthid", "--config", str(config)]
    rc = mg.main(argv)

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5  # ready + 4 responses
    assert json.loads(lines[0])["ready"] is True
    r1 = json.loads(lines[1])
    assert r1["ok"] is True and r1["id"] == 1 and r1["watermarked"] == "WM:Hola Madrid"
    r2 = json.loads(lines[2])
    assert r2 == {"ok": True, "id": 2, "is_watermarked": True, "score": 3.5, "threshold": 0.52}
    r3 = json.loads(lines[3])
    assert r3["ok"] is False and r3["id"] == 3
    r4 = json.loads(lines[4])
    assert r4 == {"ok": True, "id": 4}


def test_cmd_serve_invalid_json_line(monkeypatch, tmp_path, capsys):
    upstream = _make_upstream(tmp_path)
    config = tmp_path / "cfg.json"
    config.write_text('{"algorithm_name": "KGW"}', encoding="utf-8")
    monkeypatch.setattr(mg, "_load_algorithm", lambda *a, **k: _FakeWM())

    stream = "this is not json\n" + json.dumps({"op": "exit", "id": 0}) + "\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stream))

    argv = ["serve", "--markllm-dir", str(upstream), "--scheme", "kgw", "--config", str(config)]
    rc = mg.main(argv)

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])["ready"] is True
    assert json.loads(lines[1]) == {"ok": False, "error": "invalid JSON request"}
    assert json.loads(lines[2]) == {"ok": True, "id": 0}


def test_main_serve_missing_markllm_dir_ready_false(monkeypatch, capsys):
    monkeypatch.delenv("MARKLLM_DIR", raising=False)
    argv = ["serve", "--markllm-dir", "/nonexistent", "--scheme", "kgw", "--config", "/fake/c.json"]
    rc = mg.main(argv)
    assert rc == 3
    out = json.loads(capsys.readouterr().out)
    assert out["ready"] is False


# --------------------------------------------------------------------------
# Generation internals
# --------------------------------------------------------------------------


def test_generate_seeds_via_torch_and_sets_kwargs(monkeypatch):
    calls: list[int] = []

    class _FakeTorch:
        @staticmethod
        def manual_seed(seed):
            calls.append(seed)

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    wm = _FakeWM()
    watermarked, unwatermarked = mg._generate(wm, "prompt", seed=7, max_new_tokens=50, min_length=5)

    assert watermarked == "WM:prompt"
    assert unwatermarked == "UW:prompt"
    assert calls == [7]
    assert wm.config.gen_kwargs == {"max_new_tokens": 50, "min_length": 5}


@requires_torch
def test_generate_with_real_torch(monkeypatch):
    """Seed path with real torch installed (skipped when torch is absent)."""
    wm = _FakeWM()
    watermarked, unwatermarked = mg._generate(wm, "prompt", seed=42, max_new_tokens=10)
    assert (watermarked, unwatermarked) == ("WM:prompt", "UW:prompt")
    assert wm.config.gen_kwargs["max_new_tokens"] == 10


@requires_torch
def test_resolve_device_auto_never_mps():
    device = mg.resolve_device("auto")
    assert device in ("cpu", "cuda")
    assert device != "mps"
