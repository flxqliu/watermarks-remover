#!/usr/bin/env python3
"""CPU-feasible multilingual (DE/FR/ES) watermark generation (gap 05-A5).

The EN core of the v1 study (research/01-experiment-protocol.md §2) uses
facebook/opt-1.3b, which is English-only. Multilingual cells (de/fr/es)
therefore use a CPU-feasible multilingual generator -- Qwen/Qwen2.5-1.5B-
Instruct (fallback: 0.5B) -- reported explicitly as a model-holdout factor
in the paper (§3): scheme x language, not confounded with the EN core.

Like service/scripts/detect_text_watermark.py, this script does NOT vendor
upstream code: it imports AutoWatermark from a user-provided THU-BPM/MarkLLM
checkout at runtime (--markllm-dir / $MARKLLM_DIR) and mirrors that script's
_load_algorithm exactly, so multilingual generation and same-config detection
share identical MarkLLM wiring (scheme config + keys + TransformersConfig
defaults).

Because Qwen2.5-Instruct tokenizers ship a chat template, each prompt is
formatted as a single user turn before generation
(tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
tokenize=False, add_generation_prompt=True)); the generated text then
includes the (formatted) prompt prefix, consistent with the EN harness.
Tokenizers without a chat template receive the raw prompt.

Subcommands:
  watermark  one-shot generation: emit a single JSON object on stdout
  serve      persistent JSON-lines stdin/stdout worker (model loaded once)

Exit codes: 0 success, 1 runtime error, 2 bad input/usage, 3 unavailable
(no MarkLLM checkout / missing deps / bad config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import pins  # noqa: E402  (reproducibility pins, gap 05-A7)

# CLI scheme name -> MarkLLM algorithm name (config/{ALG}.json). The
# multilingual grid (01 §2) uses only KGW (gamma=.5, delta=2) and
# SynthID-Text; the mapping mirrors detect_text_watermark.SCHEMES.
SCHEMES = {
    "kgw": "KGW",
    "synthid": "SynthID",
    "synthid-text": "SynthID",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LANGUAGES = ("de", "fr", "es")

# Algorithm configs are ~200 B (KGW/SynthID). Cap well above that so a
# crafted or accidental huge file is refused before upstream reads it.
MAX_CONFIG_BYTES = 1 << 20


class _Unavailable(RuntimeError):
    """Backend present but unusable (unconfigured checkout, missing deps)."""


def resolve_upstream(raw: str | None) -> Path | None:
    """Resolve a MarkLLM checkout path, or None when absent/invalid."""
    if not raw:
        return None
    upstream = Path(raw).expanduser().resolve()
    if not upstream.is_dir():
        return None
    return upstream


def resolve_device(raw: str | None) -> str:
    """Resolve the 'auto' device hint to a concrete torch device."""
    if raw and raw != "auto":
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Never auto-select mps: MarkLLM builds torch.Generator(device=...),
        # which supports only cpu/cuda and raises RuntimeError on 'mps'
        # (Apple Silicon). Fall through to cpu. Pass --device mps to override.
    except Exception:  # noqa: S110 - optional torch device detection
        pass
    return "cpu"


def _load_algorithm(
    upstream: Path,
    alg: str,
    config: Path,
    model: str,
    device: str,
    offline: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
):
    """Import the MarkLLM checkout and build an AutoWatermark instance.

    Mirrors service/scripts/detect_text_watermark.py::_load_algorithm exactly
    (same imports, TransformersConfig defaults, gen_kwargs folding) so
    multilingual generation and the service detector share identical wiring.
    The checkout is imported at runtime via sys.path -- never vendored.

    temperature/top_p (when not None) are folded into the generation kwargs
    so the v1 study's temperature factor (01 §2) is reproducible.
    """
    gen_kwargs_extra: dict[str, float] = {}
    if temperature is not None:
        gen_kwargs_extra["temperature"] = temperature
    if top_p is not None:
        gen_kwargs_extra["top_p"] = top_p
    sys.path.insert(0, str(upstream))
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import AutoWatermark
    except ImportError as e:
        raise _Unavailable(f"MarkLLM dependencies missing: {e}") from e

    # --offline: never contact the HF hub. local_files_only makes transformers
    # fail fast instead of hanging, and HF_HUB_OFFLINE covers the lower-level
    # hub calls. Custom-code execution is not possible either way: transformers
    # only honors auto_map/trust_remote_code when explicitly enabled, which is
    # never done here.
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    load_kwargs = {"local_files_only": True} if offline else {}

    tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
    lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs).to(device)
    transformers_config = TransformersConfig(
        model=lm,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        min_length=0,
        do_sample=True,
        no_repeat_ngram_size=4,
        **gen_kwargs_extra,
    )
    return AutoWatermark.load(
        alg,
        algorithm_config=str(config),
        transformers_config=transformers_config,
    )


def _resolve_config(upstream: Path, alg: str, config: str | None) -> Path:
    """Resolve the algorithm config JSON, with size + existence checks."""
    path = Path(config).expanduser().resolve() if config else upstream / "config" / f"{alg}.json"
    if not path.is_file():
        raise _Unavailable(f"MarkLLM config not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as e:
        raise _Unavailable(f"cannot stat MarkLLM config {path}: {e}") from e
    if size > MAX_CONFIG_BYTES:
        raise _Unavailable(f"MarkLLM config too large ({size} bytes > {MAX_CONFIG_BYTES}): {path}")
    return path


def _threshold_from_config(config: Path) -> float | None:
    """Detection threshold from the algorithm config (KGW/SynthID), if any."""
    try:
        data = json.loads(config.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("threshold", "z_threshold"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    """Format *prompt* as a single user turn when the tokenizer has a chat template.

    Qwen2.5-Instruct tokenizers ship a chat_template attribute; wrapping the
    prompt in one user turn with the generation prompt appended matches how
    the instruct model is normally driven (and mirrors the EN harness, where
    opt-1.3b has no template and the raw prompt is used).
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate(
    wm: Any,
    prompt: str,
    seed: int | None,
    max_new_tokens: int,
    min_length: int = 0,
    need_unwatermarked: bool = True,
) -> tuple[str, str | None]:
    """Generate watermarked (and optionally unwatermarked) text for *prompt*."""
    if seed is not None:
        import torch

        torch.manual_seed(seed)
    wm.config.gen_kwargs["max_new_tokens"] = max_new_tokens
    wm.config.gen_kwargs["min_length"] = min_length
    watermarked = wm.generate_watermarked_text(prompt)
    unwatermarked = wm.generate_unwatermarked_text(prompt) if need_unwatermarked else None
    return watermarked, unwatermarked


def _read_prompt(path: str) -> str:
    """Read the prompt file (or stdin for '-'); strip surrounding whitespace."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return text.strip()


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON object to stdout (single line, flushed)."""
    print(json.dumps(payload), flush=True)


def _cmd_watermark(args: argparse.Namespace) -> int:
    """One-shot watermark generation; emits a single JSON object on stdout.

    Success:  {"ok": true, "doc_id", "lang", "seed", "model", "scheme",
               "config", "temperature", "top_p", "watermarked",
               "unwatermarked", "pins": {"markllm_commit", "hf_revision", ...}}
    Failure:  {"ok": false, "error": str, ...} (context keys included)
    """
    context: dict[str, Any] = {
        "doc_id": args.doc_id,
        "lang": args.lang,
        "seed": args.seed,
        "model": args.model,
        "scheme": args.scheme,
        "config": str(args.config),
    }
    if args.prompt != "-" and not Path(args.prompt).is_file():
        _emit({**context, "ok": False, "error": f"not a file: {args.prompt}"})
        return 2
    try:
        prompt = _read_prompt(args.prompt)
    except (OSError, UnicodeDecodeError) as e:
        _emit({**context, "ok": False, "error": f"cannot read prompt: {e}"})
        return 2

    device = resolve_device(args.device)
    try:
        config = _resolve_config(args.upstream_dir, SCHEMES[args.scheme], args.config)
        wm = _load_algorithm(
            args.upstream_dir,
            SCHEMES[args.scheme],
            config,
            args.model,
            device,
            offline=args.offline,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        gen_prompt = apply_chat_template(wm.config.generation_tokenizer, prompt)
        watermarked, unwatermarked = _generate(
            wm, gen_prompt, args.seed, args.max_new_tokens, args.min_length
        )
    except _Unavailable as e:
        _emit({**context, "ok": False, "error": str(e)})
        return 3
    except Exception as e:
        _emit({**context, "ok": False, "error": f"generation error: {e}"})
        return 1

    _emit(
        {
            "ok": True,
            **context,
            "config": str(config),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "watermarked": watermarked,
            "unwatermarked": unwatermarked,
            "watermarked_chars": len(watermarked),
            "unwatermarked_chars": len(unwatermarked) if unwatermarked is not None else None,
            "pins": {
                "markllm_commit": pins.markllm_commit(args.upstream_dir),
                "hf_revision": pins.hf_revision(args.model),
                "repo_commit": pins.repo_commit(),
            },
        }
    )
    return 0


def _detect_payload(wm: Any, text: str, threshold: float | None) -> dict[str, Any]:
    """Same-config detection payload (is_watermarked/score/threshold).

    Mirrors detect_text_watermark.py::_detect_payload: the loaded
    AutoWatermark instance detects with the same scheme config + keys used at
    generation, so a resident worker can score texts without a second model.
    """
    result = wm.detect_watermark(text, return_dict=True)
    is_watermarked = bool(result.get("is_watermarked", False))
    score = result.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    return {
        "is_watermarked": is_watermarked,
        "score": score,
        "threshold": threshold,
    }


def _handle_serve_request(
    wm: Any, req: dict[str, Any], threshold: float | None = None
) -> dict[str, Any]:
    """Handle one JSON-lines request; never raises (responds with ok:false).

    Supports watermark (generate) and detect ops; *threshold* is the
    algorithm-config detection threshold read once at serve load.
    """
    rid = req.get("id")
    op = req.get("op")
    if op == "exit":
        return {"ok": True, "id": rid}
    try:
        if op == "watermark":
            prompt = req.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("'prompt' must be a non-empty string")
            # Per-request generation knobs (temperature factor of the v1 study).
            for key in ("temperature", "top_p"):
                value = req.get(key)
                if isinstance(value, (int, float)):
                    wm.config.gen_kwargs[key] = float(value)
            gen_prompt = apply_chat_template(wm.config.generation_tokenizer, prompt)
            watermarked, unwatermarked = _generate(
                wm,
                gen_prompt,
                req.get("seed"),
                req.get("max_new_tokens", 200),
                req.get("min_length", 0),
                need_unwatermarked=True,
            )
            return {
                "ok": True,
                "id": rid,
                "watermarked": watermarked,
                "unwatermarked": unwatermarked,
                "watermarked_chars": len(watermarked),
                "unwatermarked_chars": len(unwatermarked),
            }
        if op == "detect":
            text = req.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("'text' must be a non-empty string")
            det = _detect_payload(wm, text, threshold)
            return {"ok": True, "id": rid, **det}
        return {"ok": False, "id": rid, "error": f"unknown op {op!r}"}
    except Exception as e:  # a bad request must not kill the worker
        return {"ok": False, "id": rid, "error": str(e)}


def _cmd_serve(args: argparse.Namespace) -> int:
    """Serve watermark/detect requests over JSON-lines stdin/stdout.

    Loads the MarkLLM model once and keeps it resident so callers (e.g. the
    v1 orchestrator) can generate AND detect many multilingual docs without
    paying the torch + model load cost per call. Protocol (identical to
    detect_text_watermark.py serve):

      first stdout line:  {"ready": true, "scheme", "model", "device", ...}
      request:  {"op": "watermark", "id": N, "prompt": str, "seed": int|None,
                 "max_new_tokens": int, "min_length": int,
                 "temperature": float|None, "top_p": float|None}
                {"op": "detect", "id": N, "text": str}
                {"op": "exit", "id": N}
      response: {"ok": true, "id": N, ...} | {"ok": false, "id": N, "error": str}

    Per-request "temperature"/"top_p" override the worker's generation kwargs
    (the v1 study's temperature factor); omitted keys keep the current value.
    Detect runs same-config detection on the resident model (the threshold is
    read once from the algorithm config). Errors on one request never kill the
    worker; {"op": "exit"} ends it.
    """
    device = resolve_device(args.device)
    try:
        config = _resolve_config(args.upstream_dir, SCHEMES[args.scheme], args.config)
        threshold = _threshold_from_config(config)
        wm = _load_algorithm(
            args.upstream_dir,
            SCHEMES[args.scheme],
            config,
            args.model,
            device,
            offline=args.offline,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except _Unavailable as e:
        _emit({"ready": False, "error": str(e)})
        return 3
    except Exception as e:
        _emit({"ready": False, "error": f"serve load error: {e}"})
        return 1

    _emit(
        {
            "ready": True,
            "scheme": args.scheme,
            "model": args.model,
            "device": device,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit({"ok": False, "error": "invalid JSON request"})
            continue
        if not isinstance(req, dict):
            _emit({"ok": False, "error": "request must be a JSON object"})
            continue
        _emit(_handle_serve_request(wm, req, threshold))
        if req.get("op") == "exit":
            return 0
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    """Shared flags for every subcommand (MarkLLM + decoding options)."""
    p.add_argument(
        "--markllm-dir",
        type=Path,
        default=None,
        help="MarkLLM checkout root (default: $MARKLLM_DIR)",
    )
    p.add_argument(
        "--scheme",
        required=True,
        choices=sorted(SCHEMES),
        help="Watermark scheme to use (kgw, synthid)",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Algorithm config JSON (e.g. research/configs/KGW-d2.json)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MARKLLM_MODEL", DEFAULT_MODEL),
        help=f"HF causal LM for generation (default: $MARKLLM_MODEL or {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Never contact the HF hub: load the model from the local cache "
        "only (fails fast if not cached)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (default: unset -> MarkLLM/HF default)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Generation nucleus-sampling top-p (default: unset -> MarkLLM/HF default)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (two subcommands: watermark, serve)."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    wm = sub.add_parser(
        "watermark", help="Generate watermarked + unwatermarked text for one prompt"
    )
    wm.add_argument("--prompt", required=True, help="Prompt file, or - for stdin")
    wm.add_argument("--seed", type=int, required=True, help="RNG seed (fixed 1..5 per cell, 01 §3)")
    wm.add_argument(
        "--max-new-tokens", type=int, required=True, help="Tokens to generate (length factor)"
    )
    wm.add_argument(
        "--min-length", type=int, default=0, help="Minimum total length in tokens (default: 0)"
    )
    wm.add_argument("--lang", required=True, choices=LANGUAGES, help="Target language: de, fr, es")
    wm.add_argument("--doc-id", required=True, help="Stable document id recorded in the manifest")
    _add_common(wm)
    wm.set_defaults(handler=_cmd_watermark)

    serve = sub.add_parser(
        "serve", help="Persistent JSON-lines stdin/stdout worker (model loaded once)"
    )
    _add_common(serve)
    serve.set_defaults(handler=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    fail_key = "ready" if args.cmd == "serve" else "ok"
    raw_upstream = args.markllm_dir or os.environ.get("MARKLLM_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)
    if upstream is None:
        _emit(
            {
                fail_key: False,
                "error": "MarkLLM not configured: set MARKLLM_DIR or pass --markllm-dir",
            }
        )
        return 3
    if not (upstream / "watermark").is_dir():
        _emit(
            {
                fail_key: False,
                "error": f"MarkLLM checkout incomplete (no watermark/ dir): {upstream}",
            }
        )
        return 3
    args.upstream_dir = upstream
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
