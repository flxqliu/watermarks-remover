#!/usr/bin/env python3
"""Vendor and research text-watermark detectors behind one interface.

Detects statistical (Layer B) text watermarks using vendor-provided or
research detectors. Every detector implements the same small protocol:

    name: str                 stable identifier (surfaced in /capabilities)
    available() -> bool       configured and usable right now
    detect(text) -> dict      JSON-safe report; never raises

Reports follow the fail-soft contract: a detector that is unconfigured,
times out, or errors returns {"available": False, "error": ...} and can
never block cleaning.

Detectors:

- gemini-synthid-text — Google's official SynthID-text detector, called
  through the Gemini API (taskType DETECT_TEXT_WATERMARK). Activated by
  WATERMARKS_GEMINI_API_KEY. User text is sent to Google only when the
  operator sets that key.
- markllm — optional research harness (KGW / SynthID schemes) via
  detect_text_watermark.py, activated by MARKLLM_DIR. Same-config-only
  detection; not a vendor oracle.
- claude-text — placeholder for Anthropic's announced text-watermark
  detection API. Reports unavailable until a public endpoint exists; the
  interface it must implement is already defined here.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

GEMINI_DETECT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TIMEOUT = 30.0
DEFAULT_GEMINI_MAX_CHARS = 1_000_000
DEFAULT_MARKLLM_SCHEME = "kgw"
DEFAULT_MARKLLM_TIMEOUT = 600.0


class DetectorError(RuntimeError):
    """A detector call failed (network, HTTP error, timeout)."""


class TextDetector(Protocol):
    name: str

    def available(self) -> bool: ...

    def detect(self, text: str) -> dict[str, Any]: ...


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Gemini (Google's official SynthID-text detector)
# ---------------------------------------------------------------------------

_WATERMARKED_VERDICTS = ("watermarked", "ai-generated", "ai generated", "likely ai")


def _verdict_is_watermarked(verdict: str | None) -> bool | None:
    """Map the detector model's free-text verdict to a boolean, or None."""
    if not verdict:
        return None
    low = verdict.strip().lower()
    if low.startswith(("unlikely", "no", "not")):
        return False
    return any(marker in low for marker in _WATERMARKED_VERDICTS)


def _extract_numeric_score(candidate: dict[str, Any], top: dict[str, Any]) -> float | None:
    """Pull a numeric watermark score from any of the known response shapes."""
    for container in (candidate, top):
        for key in (
            "syntheticTextScore",
            "synthetic_text_score",
            "watermarkScore",
            "watermark_score",
            "score",
        ):
            value = container.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    attribution = candidate.get("attributionMetadata") or {}
    if isinstance(attribution, dict):
        for key in ("syntheticTextScore", "synthetic_text_score", "score"):
            value = attribution.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        st = attribution.get("syntheticText")
        if isinstance(st, dict):
            for key in ("score", "confidence"):
                value = st.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def parse_gemini_detect_response(data: dict[str, Any]) -> dict[str, Any]:
    """Parse a generateContent response from a DETECT_TEXT_WATERMARK call.

    The endpoint can answer with either a free-text verdict
    ("Likely AI-generated") or a structured score; both shapes are handled
    defensively so upstream schema changes degrade to an error report
    instead of a crash.
    """
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    if not isinstance(candidate, dict):
        candidate = {}

    if not candidate:
        feedback = data.get("promptFeedback") or {}
        block = feedback.get("blockReason")
        if block:
            raise DetectorError(f"Gemini blocked the request: {block}")
        raise DetectorError("Gemini returned no candidates")

    verdict: str | None = None
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    if parts and isinstance(parts[0], dict):
        verdict = parts[0].get("text")

    score = _extract_numeric_score(candidate, data)
    is_watermarked = _verdict_is_watermarked(verdict)
    if is_watermarked is None and score is not None:
        is_watermarked = score >= 0.5

    if verdict is None and score is None:
        raise DetectorError(
            f"unexpected Gemini response (no verdict or score): {json.dumps(data)[:400]}"
        )

    raw = {
        key: candidate[key]
        for key in ("attributionMetadata", "finishReason", "index")
        if candidate.get(key) is not None
    }
    return {
        "is_watermarked": is_watermarked,
        "score": score,
        "verdict": verdict,
        "raw": raw,
    }


def _post_json(url: str, body: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    """POST *body* to *url*, retrying once on transient failures."""
    if urlparse(url).scheme not in ("http", "https"):
        raise DetectorError(f"refusing non-http(s) Gemini endpoint: {url}")
    # S310: URL scheme is restricted to http/https just above.
    req = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    last_err = "Gemini API call failed"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise DetectorError("non-object Gemini response")
                return payload
        except urllib.error.HTTPError as e:
            last_err = f"Gemini API HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code not in (429, 500, 502, 503, 504):
                raise DetectorError(last_err) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = f"Gemini API unreachable: {e}"
        if attempt == 0:
            time.sleep(1.0)
    raise DetectorError(last_err)


class GeminiSynthIDTextDetector:
    """Google's official SynthID-text detector via the Gemini API."""

    name = "gemini-synthid-text"
    vendor = "google"

    def available(self) -> bool:
        return bool(os.environ.get("WATERMARKS_GEMINI_API_KEY", "").strip())

    def detect(self, text: str) -> dict[str, Any]:
        api_key = os.environ.get("WATERMARKS_GEMINI_API_KEY", "").strip()
        if not api_key:
            return {
                "detector": self.name,
                "vendor": self.vendor,
                "available": False,
                "error": "WATERMARKS_GEMINI_API_KEY not set",
            }

        max_chars = _env_int("WATERMARKS_GEMINI_MAX_CHARS", DEFAULT_GEMINI_MAX_CHARS)
        if len(text) > max_chars:
            return {
                "detector": self.name,
                "vendor": self.vendor,
                "available": True,
                "skipped": True,
                "reason": f"text longer than {max_chars} chars",
                "is_watermarked": None,
            }

        model = (
            os.environ.get("WATERMARKS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL
        )
        timeout = _env_float("WATERMARKS_GEMINI_TIMEOUT", DEFAULT_GEMINI_TIMEOUT)
        url = GEMINI_DETECT_URL.format(model=model)
        body = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"taskType": "DETECT_TEXT_WATERMARK"},
        }
        report: dict[str, Any] = {
            "detector": self.name,
            "vendor": self.vendor,
            "model": model,
            "available": True,
        }
        try:
            data = _post_json(url, body, api_key, timeout)
        except DetectorError as e:
            report["available"] = False
            report["error"] = str(e)
            return report
        try:
            parsed = parse_gemini_detect_response(data)
        except DetectorError as e:
            report["available"] = False
            report["error"] = str(e)
            return report
        report.update(parsed)
        return report


# ---------------------------------------------------------------------------
# MarkLLM (open-source research harness: KGW / SynthID schemes)
# ---------------------------------------------------------------------------


class MarkLLMTextDetector:
    """Same-config-only research detection via detect_text_watermark.py."""

    name = "markllm"

    def available(self) -> bool:
        return bool(os.environ.get("MARKLLM_DIR", "").strip())

    def detect(self, text: str) -> dict[str, Any]:
        upstream = os.environ.get("MARKLLM_DIR", "").strip()
        scheme = (
            os.environ.get("WATERMARKS_MARKLLM_SCHEME", DEFAULT_MARKLLM_SCHEME)
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

        script = Path(__file__).resolve().parent / "detect_text_watermark.py"
        timeout = _env_float("WATERMARKS_MARKLLM_TIMEOUT", DEFAULT_MARKLLM_TIMEOUT)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
            f.write(text)
            tmp = f.name
        try:
            try:
                r = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "detect",
                        tmp,
                        "--scheme",
                        scheme,
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
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
# Registry
# ---------------------------------------------------------------------------


def all_detectors() -> list[TextDetector]:
    return [GeminiSynthIDTextDetector(), MarkLLMTextDetector(), ClaudeTextDetector()]


def detector_status() -> dict[str, bool]:
    """Configured/usable status per detector (for /capabilities)."""
    return {d.name: d.available() for d in all_detectors()}


def run_all_text_detectors(text: str) -> list[dict[str, Any]]:
    """Run every detector (including unavailable ones, with reasons)."""
    return [d.detect(text) for d in all_detectors()]


def run_text_detectors(text: str) -> list[dict[str, Any]]:
    """Run only the detectors that are configured and usable."""
    return [d.detect(text) for d in all_detectors() if d.available()]
