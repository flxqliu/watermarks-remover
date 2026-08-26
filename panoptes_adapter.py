"""Panoptes adapter — expose watermarks-remover to the Panoptes workbench.

Implements the Panoptes watermark-remover adapter contract
(https://github.com/marketstandard/Panoptes — docs/testing-external-repos.md):

    watermark-remover contract:  transform(text: str) -> str

Default path: the deterministic Layer A Unicode-hygiene scrub
(service/scripts/text_unicode.clean_text) — strips invisible/format Unicode
(zero-width spaces, tag characters, private-use carriers) and normalizes
space homoglyphs. No model, no network, no API key.

Opt-in Layer B: set WATERMARKS_REMOVER_ADAPTER_LAYER_B=1 to follow Layer A
with the statistical-rewrite pass (service/scripts/rewrite_text.rewrite),
configured through the same WATERMARKS_REWRITE_* env vars as
rewrite_text.py. Layer B needs a real rewrite backend (default: ollama on
loopback); the CLI's print-prompt backend is rejected here because it
returns a rewrite prompt, not cleaned text.

Sandbox note: Panoptes evaluate-repo scrubs env vars containing API_KEY /
TOKEN / etc. before running an adapter, so WATERMARKS_REWRITE_API_KEY never
reaches this process there — under evaluate-repo, Layer B is limited to
loopback backends that need no key (e.g. a local Ollama).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _REPO_ROOT / "service" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from text_unicode import clean_text  # noqa: E402

_LAYER_B_ENV = "WATERMARKS_REMOVER_ADAPTER_LAYER_B"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Conservative general-purpose rewrite strength; the CLI default.
_LAYER_B_STRENGTH = "paraphrase"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


def _layer_b(text: str) -> str:
    """Run the Layer B statistical rewrite, configured via WATERMARKS_REWRITE_*."""
    from rewrite_text import rewrite  # lazy: keep `import panoptes_adapter` side-effect-light

    backend = os.environ.get("WATERMARKS_REWRITE_BACKEND", "").strip() or "ollama"
    if backend == "print-prompt":
        raise RuntimeError(
            "WATERMARKS_REWRITE_BACKEND=print-prompt cannot drive the Panoptes adapter: "
            "it returns a rewrite prompt, not cleaned text. Use ollama or openai-compatible."
        )
    model = os.environ.get("WATERMARKS_REWRITE_MODEL") or None
    if model is None:
        raise RuntimeError(
            f"Layer B requested ({_LAYER_B_ENV}=1) but WATERMARKS_REWRITE_MODEL is unset"
        )
    effort = os.environ.get("WATERMARKS_REWRITE_REASONING_EFFORT", "none")
    rewritten, _info = rewrite(
        text,
        backend=backend,
        model=model,
        base_url=os.environ.get("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
        api_key=os.environ.get("WATERMARKS_REWRITE_API_KEY"),
        strength=_LAYER_B_STRENGTH,
        lang="French",
        original_lang="English",
        timeout=120.0,
        layer_a_after=True,  # scrub any invisibles the rewrite model emits
        temperature=0.9,
        candidates=_int_env("WATERMARKS_REWRITE_CANDIDATES", 1),
        max_loops=_int_env("WATERMARKS_REWRITE_LOOPS", 1),
        allow_remote=_flag("WATERMARKS_REWRITE_ALLOW_REMOTE"),
        reasoning_effort=None if effort == "off" else effort,
    )
    return rewritten


def transform(text: str) -> str:
    """Return ``text`` with watermarks removed (Layer A; opt-in Layer B)."""
    cleaned, _stats = clean_text(text)
    if _flag(_LAYER_B_ENV):
        return _layer_b(cleaned)
    return cleaned
