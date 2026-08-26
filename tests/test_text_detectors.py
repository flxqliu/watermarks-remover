"""Tests for text_detectors.py (vendor/research text-watermark detectors)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import text_detectors


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "WATERMARKS_MARKLLM_SCHEME",
        "MARKLLM_DIR",
        "WATERMARKS_MARKLLM_TIMEOUT",
        "PANOPTES_API_URL",
        "PANOPTES_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


# --- MarkLLM ---------------------------------------------------------------


def test_markllm_unconfigured():
    assert text_detectors.MarkLLMTextDetector().available() is False
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is False
    assert "MARKLLM_DIR" in report["error"]


def test_markllm_success(monkeypatch):
    monkeypatch.setenv("MARKLLM_DIR", "/fake/MarkLLM")
    payload = {"is_watermarked": True, "score": 4.2, "threshold": 4.0}
    monkeypatch.setattr(
        text_detectors.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=json.dumps(payload)),
    )
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert "research harness" in report["note"]


def test_markllm_unavailable_exit3(monkeypatch):
    monkeypatch.setenv("MARKLLM_DIR", "/fake/MarkLLM")
    monkeypatch.setattr(
        text_detectors.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 3, stdout="", stderr="missing deps"),
    )
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is False
    assert "missing deps" in report["error"]


def test_markllm_scheme_env(monkeypatch):
    monkeypatch.setenv("MARKLLM_DIR", "/fake/MarkLLM")
    monkeypatch.setenv("WATERMARKS_MARKLLM_SCHEME", "synthid")
    seen = {}

    def fake_run(*args, **kwargs):
        seen["argv"] = args[0]
        return subprocess.CompletedProcess(args[0], 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    text_detectors.MarkLLMTextDetector().detect("hello")
    assert "--scheme" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--scheme") + 1] == "synthid"


def test_markllm_prefers_checkout_venv(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    if os.name == "nt":
        venv_python = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = upstream / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    assert det.available() is True
    det.detect("hello")
    assert seen["argv"][0] == str(venv_python)


def test_markllm_falls_back_to_sys_executable(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    det.detect("hello")
    assert seen["argv"][0] == sys.executable


def test_markllm_ctor_overrides_passed_to_adapter(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(
        scheme="synthid",
        upstream_dir=str(upstream),
        model="opt-1.3b",
        timeout=5,
    )
    det.detect("hello")
    argv = seen["argv"]
    assert argv[argv.index("--scheme") + 1] == "synthid"
    assert argv[argv.index("--model") + 1] == "opt-1.3b"
    assert argv[argv.index("--upstream-dir") + 1] == str(upstream.resolve())


def test_markllm_ctor_available_with_override(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    assert det.available() is True


def test_markllm_preexec_default_off(monkeypatch):
    monkeypatch.delenv("WATERMARKS_MARKLLM_RLIMIT_AS", raising=False)
    assert text_detectors._markllm_preexec() is None


def test_markllm_preexec_env(monkeypatch):
    if os.name != "posix":
        pytest.skip("preexec_fn is POSIX-only")
    monkeypatch.setenv("WATERMARKS_MARKLLM_RLIMIT_AS", "0x40000000")
    assert callable(text_detectors._markllm_preexec())


def test_markllm_detect_applies_rlimit(monkeypatch, tmp_path):
    if os.name != "posix":
        pytest.skip("preexec_fn is POSIX-only")
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    monkeypatch.setenv("WATERMARKS_MARKLLM_RLIMIT_AS", "1073741824")
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["preexec_fn"] = kwargs.get("preexec_fn")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"available": true}')

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    assert report["available"] is True
    assert callable(captured["preexec_fn"])


def _spin_worker_server(response):
    import socketserver
    import threading

    class _H(socketserver.BaseRequestHandler):
        def handle(self):
            f = self.request.makefile("r", encoding="utf-8")
            f.readline()
            self.request.sendall((json.dumps(response) + "\n").encode("utf-8"))

    class _S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = _S(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_markllm_detector_uses_worker_port(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    srv = _spin_worker_server({"ok": True, "is_watermarked": True, "score": 2.0, "threshold": 0.5})
    monkeypatch.setenv("WATERMARKS_MARKLLM_PORT", str(srv.server_address[1]))
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    srv.shutdown()
    srv.server_close()
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["score"] == 2.0
    assert seen == []  # no cold-start subprocess
    assert "worker" in report["note"]


def test_markllm_detector_worker_fallback_to_subprocess(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    monkeypatch.setenv("WATERMARKS_MARKLLM_PORT", "1")  # nothing listening
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"is_watermarked": True, "score": 1.0})
        )

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert len(calls) == 1  # fell back to the subprocess


def test_run_all_text_detectors_can_exclude_markllm(monkeypatch):
    monkeypatch.setattr(
        text_detectors,
        "MarkLLMTextDetector",
        lambda: pytest.fail("must not construct MarkLLM when excluded"),
    )
    reports = text_detectors.run_all_text_detectors("hello", include_markllm=False)
    assert len(reports) == 3  # gumbel + claude placeholder + panoptes
    assert {r["detector"] for r in reports} == {"gumbel", "claude-text", "panoptes"}


def test_run_all_text_detectors_injects_markllm_instance(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream), scheme="synthid")
    seen: list = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    text_detectors.run_all_text_detectors("hello", markllm=det)
    markllm_cmd = next(c for c in seen if "--scheme" in c)
    assert markllm_cmd[markllm_cmd.index("--scheme") + 1] == "synthid"


# --- Claude placeholder ----------------------------------------------------


def test_claude_placeholder():
    det = text_detectors.ClaudeTextDetector()
    assert det.available() is False
    report = det.detect("hello")
    assert report["available"] is False
    assert "WATERMARKS_CLAUDE_API_KEY" in report["error"]


# --- Registry --------------------------------------------------------------


def test_detector_status_keys():
    status = text_detectors.detector_status()
    assert set(status) == {"markllm", "gumbel", "claude-text", "panoptes"}


def test_run_all_text_detectors_length():
    reports = text_detectors.run_all_text_detectors("hello")
    assert len(reports) == 4  # markllm + gumbel + claude placeholder + panoptes
    assert all("detector" in r for r in reports)


def test_run_text_detectors_filters_unavailable():
    reports = text_detectors.run_text_detectors("hello")
    assert reports == []


# --- Panoptes cross-detector -------------------------------------------------


def _spin_panoptes(payload=None, *, raw=None, delay=0.0):
    """Loopback HTTP fake for the Panoptes /api/v1/analyze endpoint."""
    import http.server
    import threading
    import time

    captured: dict = {}

    class _H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            captured["path"] = self.path
            captured["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            if delay:
                time.sleep(delay)
            body = raw if raw is not None else json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, captured


def _panoptes_url(srv) -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}"


def _panoptes_payload() -> dict:
    return {
        "summary": {"ai_generation": 0.91, "ai_participation": 0.95},
        "watermarks": [
            {"scheme": "kgw-v1", "status": "tested", "z": 7.13, "p_value": 1e-12},
        ],
    }


def test_panoptes_unconfigured():
    det = text_detectors.PanoptesTextDetector()
    assert det.available() is False
    report = det.detect("hello")
    assert report["available"] is False
    assert "PANOPTES_API_URL" in report["error"]


def test_panoptes_rejects_non_http_scheme(monkeypatch):
    monkeypatch.setenv("PANOPTES_API_URL", "file:///etc/passwd")
    det = text_detectors.PanoptesTextDetector()
    assert det.available() is False
    assert det.detect("hello")["available"] is False


def test_panoptes_success_maps_analysis_response():
    srv, captured = _spin_panoptes(_panoptes_payload())
    try:
        det = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv))
        assert det.available() is True
        report = det.detect("some text")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["score"] == 7.13
    assert report["kgw"]["p_value"] == 1e-12
    assert report["ai_generation"] == 0.91
    assert report["ai_participation"] == 0.95
    assert "demo key" in report["note"]
    assert captured["path"] == "/api/v1/analyze"
    assert json.loads(captured["body"]) == {"text": "some text", "content_type": "prose"}


def test_panoptes_kgw_not_tested_yields_unknown():
    payload = _panoptes_payload()
    payload["watermarks"][0].update({"status": "insufficient_data", "z": None, "p_value": None})
    srv, _ = _spin_panoptes(payload)
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is True
    assert report["is_watermarked"] is None
    assert report["score"] is None
    assert report["ai_generation"] == 0.91


def test_panoptes_unreachable_fails_soft():
    det = text_detectors.PanoptesTextDetector(url="http://127.0.0.1:1", timeout=2.0)
    assert det.available() is True  # well-formed URL; reachability is a detect-time concern
    report = det.detect("hello")
    assert report["available"] is False
    assert "failed" in report["error"]


def test_panoptes_timeout_fails_soft():
    srv, _ = _spin_panoptes(_panoptes_payload(), delay=1.5)
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv), timeout=0.2).detect(
            "x"
        )
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is False
    assert "failed" in report["error"]


def test_panoptes_malformed_json_fails_soft():
    srv, _ = _spin_panoptes(raw=b"not json")
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is False
    assert report["error"].startswith("bad panoptes JSON")


def test_panoptes_oversize_response_fails_soft():
    raw = b" " * (text_detectors.PANOPTES_MAX_RESPONSE_BYTES + 10)
    srv, _ = _spin_panoptes(raw=raw)
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is False
    assert "exceeds" in report["error"]


def test_panoptes_redirect_is_refused():
    """A 302 must surface as an error, never a followed redirect (text would leak)."""
    import http.server
    import threading

    class _R(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/never")
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _R)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("secret")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is False
    assert "failed" in report["error"]


def test_panoptes_ctor_url_overrides_env(monkeypatch):
    monkeypatch.setenv("PANOPTES_API_URL", "http://127.0.0.1:1")
    det = text_detectors.PanoptesTextDetector(url="http://example.invalid:9/")
    assert det._base_url() == "http://example.invalid:9"  # trailing slash stripped


def test_panoptes_zero_p_value_is_positive():
    """A tested KGW p_value of 0.0 is a positive, not a missing value."""
    payload = _panoptes_payload()
    payload["watermarks"][0]["p_value"] = 0.0
    srv, _ = _spin_panoptes(payload)
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["kgw"]["p_value"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("summary", ["invalid"]), ("watermarks", 7)],
)
def test_panoptes_malformed_nested_fields_fail_soft(field, value):
    payload = _panoptes_payload()
    payload[field] = value
    srv, _ = _spin_panoptes(payload)
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is False
    assert report["error"].startswith("bad panoptes response")


def test_panoptes_non_numeric_kgw_field_fails_soft():
    payload = _panoptes_payload()
    payload["watermarks"][0]["p_value"] = "high"
    srv, _ = _spin_panoptes(payload)
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is False
    assert "non-numeric" in report["error"]


@pytest.mark.parametrize("bad", ["-1", "nan", "inf"])
def test_panoptes_invalid_env_timeout_falls_back(monkeypatch, bad):
    """Garbage PANOPTES_TIMEOUT values fall back to the default, never reach urllib."""
    monkeypatch.setenv("PANOPTES_TIMEOUT", bad)
    srv, _ = _spin_panoptes(_panoptes_payload())
    try:
        report = text_detectors.PanoptesTextDetector(url=_panoptes_url(srv)).detect("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["available"] is True


def test_panoptes_invalid_ctor_timeout_fails_soft():
    det = text_detectors.PanoptesTextDetector(url="http://127.0.0.1:1", timeout=-1.0)
    report = det.detect("x")
    assert report["available"] is False
    assert "invalid panoptes timeout" in report["error"]


def test_cross_detectors_registry():
    assert sorted(text_detectors.CROSS_DETECTORS) == ["panoptes"]
    assert text_detectors.CROSS_DETECTORS["panoptes"]().name == "panoptes"
