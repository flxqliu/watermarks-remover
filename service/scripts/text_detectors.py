#!/usr/bin/env python3
"""Research text-watermark detectors behind one interface.

Detects statistical (Layer B) text watermarks using research or vendor
detectors. Every detector implements the same small protocol:

    name: str                 stable identifier (surfaced in /capabilities)
    available() -> bool       configured and usable right now
    detect(text) -> dict      JSON-safe report; never raises

Reports follow the fail-soft contract: a detector that is unconfigured,
times out, or errors returns {"available": False, "error": ...} and can
never block cleaning.

Detectors:

- markllm — research harness (KGW / SynthID schemes) via
  detect_text_watermark.py, activated by MARKLLM_DIR. Same-config-only
  detection; not a vendor oracle.
- gumbel — model-free same-key replay of the keyed-Gumbel (Aaronson EXP)
  scheme (detect_gumbel.py), activated by WATERMARKS_GUMBEL_KEY. Stdlib-only;
  valid only against the same key, tokenizer, and PRF layout used at
  generation (self-hosted engines such as arbi-serve); not a vendor oracle.
- claude-text — placeholder for Anthropic's announced text-watermark
  detection API. Reports unavailable until a public endpoint exists; the
  interface it must implement is already defined here.
- panoptes — independent second opinion from a Panoptes workbench over
  HTTP, activated by PANOPTES_API_URL. KGW is scoped to Panoptes' baked
  demo key; the calibrated AI posteriors are the watermark-agnostic
  cross-check.

Vendor note (Aug 2026): Google retired SynthID text watermarking on the
Generative Language API — API text output is no longer watermarked and
DETECT_TEXT_WATERMARK is rejected on current (3.x) models. The former
gemini-synthid-text detector was removed for this reason; a vendor seam can
be re-added if Google exposes detection again (e.g. via Vertex AI).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from detect_gumbel import DEFAULT_THRESHOLD, DEFAULT_WINDOW, detect_text

DEFAULT_MARKLLM_SCHEME = "kgw"
DEFAULT_MARKLLM_TIMEOUT = 600.0


class TextDetector(Protocol):
    name: str

    def available(self) -> bool: ...

    def detect(self, text: str) -> dict[str, Any]: ...


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    # urllib raises ValueError on non-finite or negative timeouts; fall back.
    return value if math.isfinite(value) and value > 0 else default


def _worker_port() -> int | None:
    """Loopback port of a resident MarkLLM serve worker (WATERMARKS_MARKLLM_PORT)."""
    raw = os.environ.get("WATERMARKS_MARKLLM_PORT", "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        return None
    return port if 0 < port < 65536 else None


def _detect_via_worker(port: int, text: str, timeout: float) -> dict[str, Any]:
    """One detect request to a resident MarkLLM serve worker over loopback TCP."""
    import socket as _socket

    with _socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.sendall((json.dumps({"op": "detect", "text": text}) + "\n").encode("utf-8"))
        f = conn.makefile("r", encoding="utf-8")
        line = f.readline()
    if not line:
        raise RuntimeError("worker closed without a response")
    try:
        resp = json.loads(line)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"worker emitted non-JSON: {line[:120]!r}") from e
    if not isinstance(resp, dict) or not resp.get("ok"):
        raise RuntimeError(resp.get("error") or "worker detect failed")
    return resp


# ---------------------------------------------------------------------------
# MarkLLM (open-source research harness: KGW / SynthID schemes)
# ---------------------------------------------------------------------------


def _venv_python(upstream: Path) -> Path | None:
    """Prefer the MarkLLM checkout's venv interpreter, if it exists."""
    if os.name == "nt":
        candidate = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = upstream / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _markllm_preexec() -> Callable[[], None] | None:
    """Optional RLIMIT_AS guard for the MarkLLM child; None means "no limit".

    torch/CUDA usually needs large address spaces, so this is opt-in via
    WATERMARKS_MARKLLM_RLIMIT_AS (byte count, hex/octal allowed). POSIX only;
    on Windows preexec_fn must stay None.
    """
    raw = os.environ.get("WATERMARKS_MARKLLM_RLIMIT_AS")
    if not raw or os.name != "posix":
        return None
    try:
        limit = int(raw, 0)
    except ValueError:
        return None

    def _apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return _apply


class MarkLLMTextDetector:
    """Same-config-only research detection via detect_text_watermark.py.

    Constructor overrides (scheme, upstream_dir, model, timeout) take
    precedence over the environment, so callers such as rewrite_text.py can
    keep CLI flags driving the harness. When the MarkLLM checkout has a
    venv, its interpreter runs the child process; otherwise the current
    interpreter is used (the service image bundles the harness deps).
    """

    name = "markllm"

    def __init__(
        self,
        *,
        scheme: str | None = None,
        upstream_dir: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._scheme = scheme
        self._upstream_dir = upstream_dir
        self._model = model
        self._timeout = timeout

    def available(self) -> bool:
        upstream = self._upstream_dir or os.environ.get("MARKLLM_DIR", "").strip()
        return bool(upstream)

    def detect(self, text: str) -> dict[str, Any]:
        upstream = self._upstream_dir or os.environ.get("MARKLLM_DIR", "").strip()
        scheme = (
            self._scheme
            or os.environ.get("WATERMARKS_MARKLLM_SCHEME", "")
            or DEFAULT_MARKLLM_SCHEME
        )
        report: dict[str, Any] = {
            "detector": self.name,
            "scheme": scheme,
            "vendor": "open-llm",
            "available": False,
        }
        if not upstream:
            report["error"] = "MARKLLM_DIR not set"
            return report

        timeout = (
            self._timeout
            if self._timeout is not None
            else _env_float("WATERMARKS_MARKLLM_TIMEOUT", DEFAULT_MARKLLM_TIMEOUT)
        )

        # Reuse a resident serve worker (WATERMARKS_MARKLLM_PORT) when one is
        # up — avoids a ~20s torch+model cold start per detect. Falls back to
        # a one-shot subprocess if the worker is unreachable.
        port = _worker_port()
        if port is not None:
            try:
                resp = _detect_via_worker(port, text, timeout)
                return {
                    **report,
                    "available": True,
                    "is_watermarked": bool(resp["is_watermarked"]),
                    "score": resp.get("score"),
                    "threshold": resp.get("threshold"),
                    "note": "detected via resident MarkLLM serve worker",
                }
            except Exception as e:
                report["error"] = f"MarkLLM worker detect failed ({e}); falling back"

        script = Path(__file__).resolve().parent / "detect_text_watermark.py"
        venv_python = _venv_python(Path(upstream).expanduser().resolve())
        python = str(venv_python) if venv_python is not None else sys.executable

        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
            f.write(text)
            tmp = f.name

        cmd = [python, str(script), "detect", tmp, "--scheme", scheme, "--json"]
        if self._model:
            cmd += ["--model", self._model]
        if self._upstream_dir:
            cmd += ["--upstream-dir", str(Path(upstream).expanduser().resolve())]

        try:
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    preexec_fn=_markllm_preexec(),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                report["error"] = "MarkLLM detection timed out"
                return report
            if r.returncode == 3:
                report["error"] = (r.stderr or "").strip()[:400] or "MarkLLM unavailable"
                return report
            if r.returncode != 0:
                report["error"] = (r.stderr or "").strip()[:400] or f"MarkLLM exit {r.returncode}"
                return report
            try:
                payload = json.loads(r.stdout or "{}")
            except json.JSONDecodeError as e:
                report["error"] = f"bad MarkLLM JSON: {e}"
                return report
        finally:
            with contextlib.suppress(OSError):
                Path(tmp).unlink()

        if not isinstance(payload, dict):
            report["error"] = "bad MarkLLM response"
            return report
        payload["available"] = True
        payload["detector"] = self.name
        payload["note"] = (
            "MarkLLM is a research harness: detection is only valid against the "
            "same scheme config and keys used at generation; not a vendor detector."
        )
        return payload


# ---------------------------------------------------------------------------
# Keyed-Gumbel (Aaronson EXP) — model-free same-key replay
# ---------------------------------------------------------------------------


class GumbelTextDetector:
    """Same-key replay of the keyed-Gumbel (Aaronson EXP) text watermark.

    Model-free: implements the detection arithmetic of the ARBI keyed-Gumbel
    technical report (Sections 2-3) — replay u = PRF(Hash(key, window), token)
    from the text alone and test the Gamma tail — so it needs no GPU, model,
    or logits. Detection is valid only against the SAME key, tokenizer, and
    PRF layout used at generation (self-hosted engines such as arbi-serve);
    it is not a vendor oracle. Key from WATERMARKS_GUMBEL_KEY (env) or the
    constructor override.
    """

    name = "gumbel"
    vendor = "self-hosted"

    def __init__(
        self,
        *,
        key: str | None = None,
        window: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self._key = key
        self._window = window
        self._threshold = threshold

    def _key_env(self) -> str | None:
        if self._key:
            return self._key
        return os.environ.get("WATERMARKS_GUMBEL_KEY", "").strip() or None

    def available(self) -> bool:
        return self._key_env() is not None

    def detect(self, text: str) -> dict[str, Any]:
        key = self._key_env()
        report: dict[str, Any] = {
            "detector": self.name,
            "scheme": "exp",
            "vendor": self.vendor,
            "available": False,
        }
        if key is None:
            report["error"] = "WATERMARKS_GUMBEL_KEY not set"
            return report
        try:
            payload = detect_text(
                text,
                key,
                window=self._window or DEFAULT_WINDOW,
                threshold=self._threshold or DEFAULT_THRESHOLD,
            )
        except Exception as e:  # fail-soft contract: never raise
            report["error"] = f"keyed-Gumbel detection failed: {e}"
            return report
        payload["detector"] = self.name
        payload["note"] = (
            "same-key replay of the keyed-Gumbel (Aaronson EXP) watermark: valid "
            "only with the same key, tokenizer, and PRF layout used at generation; "
            "not a vendor detector."
        )
        return payload


# ---------------------------------------------------------------------------
# Claude (Anthropic) — announced detector API, not yet public
# ---------------------------------------------------------------------------


class ClaudeTextDetector:
    """Placeholder for Anthropic's announced text-watermark detection API.

    Anthropic has announced a watermark detection API for Claude-generated
    text; no public endpoint exists yet. When it ships, set
    WATERMARKS_CLAUDE_API_KEY, flip available() to check it, and fill in
    detect() against the documented endpoint.
    """

    name = "claude-text"
    vendor = "anthropic"

    def available(self) -> bool:
        return False

    def detect(self, text: str) -> dict[str, Any]:
        return {
            "detector": self.name,
            "vendor": self.vendor,
            "available": False,
            "error": (
                "Anthropic has announced a text-watermark detection API for "
                "Claude; no public endpoint is available yet. When it ships, "
                "set WATERMARKS_CLAUDE_API_KEY and implement ClaudeTextDetector."
            ),
        }


# ---------------------------------------------------------------------------
# Panoptes — independent cross-detector workbench (HTTP)
# ---------------------------------------------------------------------------

PANOPTES_DEFAULT_TIMEOUT = 30.0
PANOPTES_MAX_RESPONSE_BYTES = 1 << 20  # 1 MiB, mirroring MAX_CONFIG_BYTES discipline


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects.

    urllib's default handler re-sends request headers on 301/302/303; for a
    user-configured analysis endpoint a silent redirect would forward
    submitted text to an unvalidated host. Any 3xx now surfaces as HTTPError.
    (Local copy: rewrite_text._NoRedirect cannot be imported here without a
    circular import.)
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


class PanoptesTextDetector:
    """Independent second opinion from a Panoptes workbench over HTTP.

    Activated by PANOPTES_API_URL (e.g. http://127.0.0.1:8000 from
    ``panoptes up``). Posts the text to /api/v1/analyze and maps the
    response onto the detector contract:

    - is_watermarked / score come from Panoptes' KGW adapter, which is keyed
      to Panoptes' baked demo key — it detects only watermarks made with
      that key, so it stays silent on third-party (e.g. MarkLLM) marks.
    - ai_generation / ai_participation are the calibrated, watermark-agnostic
      posteriors: the informative cross-check on whether a rewrite still
      reads as machine-generated.

    Fail-soft like every detector here: unreachable, slow, oversize, or
    malformed responses return available=False with an error; never raises.
    """

    name = "panoptes"
    vendor = "marketstandard"

    def __init__(self, *, url: str | None = None, timeout: float | None = None) -> None:
        self._url = url
        self._timeout = timeout

    def _base_url(self) -> str | None:
        raw = self._url if self._url is not None else os.environ.get("PANOPTES_API_URL", "")
        raw = raw.strip()
        if not raw:
            return None
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.hostname:
            return None
        return raw.rstrip("/")

    def available(self) -> bool:
        return self._base_url() is not None

    def detect(self, text: str) -> dict[str, Any]:
        report: dict[str, Any] = {
            "detector": self.name,
            "scheme": "kgw",
            "vendor": self.vendor,
            "available": False,
        }
        base = self._base_url()
        if base is None:
            report["error"] = "PANOPTES_API_URL not set or not a valid http(s) URL"
            return report
        timeout = (
            self._timeout
            if self._timeout is not None
            else _env_float("PANOPTES_TIMEOUT", PANOPTES_DEFAULT_TIMEOUT)
        )
        if not math.isfinite(timeout) or timeout <= 0:
            report["error"] = f"invalid panoptes timeout: {timeout!r}"
            return report
        body = json.dumps({"text": text, "content_type": "prose"}).encode("utf-8")
        # S310: URL scheme is restricted to http/https in _base_url above.
        req = urllib.request.Request(  # noqa: S310
            f"{base}/api/v1/analyze",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(req, timeout=timeout) as resp:
                payload = resp.read(PANOPTES_MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
            report["error"] = f"panoptes analyze failed: {e}"
            return report
        if len(payload) > PANOPTES_MAX_RESPONSE_BYTES:
            report["error"] = f"panoptes response exceeds {PANOPTES_MAX_RESPONSE_BYTES} bytes"
            return report
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            report["error"] = f"bad panoptes JSON: {e}"
            return report
        if not isinstance(data, dict):
            report["error"] = "bad panoptes response"
            return report

        summary = data.get("summary")
        watermarks = data.get("watermarks")
        if summary is not None and not isinstance(summary, dict):
            report["error"] = "bad panoptes response: summary is not an object"
            return report
        if watermarks is not None and not isinstance(watermarks, list):
            report["error"] = "bad panoptes response: watermarks is not a list"
            return report
        kgw = next(
            (w for w in (watermarks or []) if isinstance(w, dict) and w.get("scheme") == "kgw-v1"),
            None,
        )
        kgw_tested = kgw is not None and kgw.get("status") == "tested"
        z = kgw.get("z") if kgw_tested else None
        p_value = kgw.get("p_value") if kgw_tested else None
        if any(v is not None and not isinstance(v, (int, float)) for v in (z, p_value)):
            report["error"] = "bad panoptes response: non-numeric kgw field"
            return report
        report["available"] = True
        report["is_watermarked"] = (p_value is not None and p_value < 0.05) if kgw_tested else None
        report["score"] = z
        report["kgw"] = (
            {"status": kgw.get("status"), "z": kgw.get("z"), "p_value": kgw.get("p_value")}
            if kgw is not None
            else None
        )
        report["ai_generation"] = (summary or {}).get("ai_generation")
        report["ai_participation"] = (summary or {}).get("ai_participation")
        report["note"] = (
            "Independent Panoptes analysis. KGW is scoped to Panoptes' baked demo "
            "key — it detects only watermarks made with that key, not third-party "
            "(e.g. MarkLLM) marks; the AI posteriors are the watermark-agnostic signal."
        )
        return report


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def all_detectors(
    markllm: MarkLLMTextDetector | None = None,
    *,
    include_markllm: bool = True,
    gumbel: GumbelTextDetector | None = None,
    include_gumbel: bool = True,
) -> list[TextDetector]:
    detectors: list[TextDetector] = []
    if include_markllm:
        detectors.append(markllm or MarkLLMTextDetector())
    if include_gumbel:
        detectors.append(gumbel or GumbelTextDetector())
    detectors.append(ClaudeTextDetector())
    detectors.append(PanoptesTextDetector())
    return detectors


# Detectors selectable via bench_synthid_text.py --cross-detect <name>:
# independent second opinions scored alongside the same-config detection.
CROSS_DETECTORS: dict[str, Callable[[], TextDetector]] = {"panoptes": PanoptesTextDetector}


def detector_status() -> dict[str, bool]:
    """Configured/usable status per detector (for /capabilities)."""
    return {d.name: d.available() for d in all_detectors()}


def run_all_text_detectors(
    text: str,
    *,
    markllm: MarkLLMTextDetector | None = None,
    include_markllm: bool = True,
    gumbel: GumbelTextDetector | None = None,
    include_gumbel: bool = True,
) -> list[dict[str, Any]]:
    """Run every detector (including unavailable ones, with reasons).

    markllm injects a caller-parameterized MarkLLM detector (e.g. one
    driven by rewrite_text.py CLI flags); pass include_markllm=False to
    exclude the MarkLLM harness entirely. Same for gumbel.
    """
    return [
        d.detect(text)
        for d in all_detectors(
            markllm,
            include_markllm=include_markllm,
            gumbel=gumbel,
            include_gumbel=include_gumbel,
        )
    ]


def run_text_detectors(
    text: str,
    *,
    markllm: MarkLLMTextDetector | None = None,
    include_markllm: bool = True,
    gumbel: GumbelTextDetector | None = None,
    include_gumbel: bool = True,
) -> list[dict[str, Any]]:
    """Run only the detectors that are configured and usable."""
    return [
        d.detect(text)
        for d in all_detectors(
            markllm,
            include_markllm=include_markllm,
            gumbel=gumbel,
            include_gumbel=include_gumbel,
        )
        if d.available()
    ]
