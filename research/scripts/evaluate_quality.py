#!/usr/bin/env python3
"""Text-quality metrics for the watermark-removal study (gap 05-B2).

Implements research/01-experiment-protocol.md §5.3 for paired
(original, candidate) texts:

  ppl             GPT-2 large mean-token perplexity of the *candidate*
                  (NEVER the generator -- the generator is opt-1.3b /
                  Qwen2.5, protocol §5.3 "never score with the
                  generator")
  bertscore       BERTScore F1, deberta-xlarge-mnli,
                  rescale_with_baseline=True
  rouge_l         ROUGE-L F1 (rouge-score lib, no stemmer / no nltk)
  sbert_cosine    SBERT cosine similarity, all-MiniLM-L6-v2
  levenshtein_pct edit distance % of original length (pure-python DP)
  length_drift    (len(candidate) - len(original)) / len(original)
  numbers_preserved / urls_preserved   regex set overlap, identical to
                  service/scripts/bench_synthid_text.py helpers

Robustness rules (the run must never crash on a model problem):

  * every model/tool is lazy-loaded on first use and cached;
  * each load is wrapped in its own try/except -- on failure that
    metric is set to None for every row, a warning is recorded, and
    the run continues;
  * per-row computation failures degrade to None + warning;
  * Levenshtein is capped: if len(original) > 4000 both texts are
    truncated to 4000 chars and the row gains a "notes" entry.

CLI:

  python3 research/scripts/evaluate_quality.py --input FILE --out FILE
      [--limit N] [--device cpu] [--skip ppl,bertscore,rouge,sbert]
      [--warnings-out FILE]

Input JSONL rows: {"condition", "scheme", "seed", "prompt_idx",
"attack", "original", "candidate"}. Output rows = input row + the eight
metric keys above (None when skipped/unavailable) and, when relevant, a
"notes" list.

The heavy dependencies (torch, transformers, bert-score,
sentence-transformers, rouge-score) are imported lazily inside the
loaders so the pure helpers and the CLI remain usable without them;
install them from research/requirements-quality.txt (a SEPARATE env
from the MarkLLM env, protocol §7).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Output metric keys, in the order they are appended to each output row.
METRIC_KEYS: tuple[str, ...] = (
    "ppl",
    "bertscore",
    "rouge_l",
    "sbert_cosine",
    "levenshtein_pct",
    "length_drift",
    "numbers_preserved",
    "urls_preserved",
)

#: Model-backed metric names accepted by --skip (output key for rouge is
#: "rouge_l"; the skip name is "rouge").
MODEL_METRICS: tuple[str, ...] = ("ppl", "bertscore", "rouge", "sbert")

#: Cap for Levenshtein inputs: DP is O(n*m), 4000 chars keeps the
#: worst case at ~16M cell updates per pair (protocol §5.3, "edit
#: magnitude" on realistic user-edited text).
LEVENSHTEIN_MAX_CHARS: int = 4000

#: Module-level caches. _MODEL_CACHE holds loaded models/tools; _FAILED
#: holds metric names whose load already failed (metric -> None forever
#: in this process); _WARNINGS is the deduplicated warning log.
_MODEL_CACHE: dict[Any, Any] = {}
_FAILED: set[str] = set()
_WARNINGS: list[dict[str, str | None]] = []


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without any model installed)
# ---------------------------------------------------------------------------


def levenshtein_distance(a: str, b: str) -> int:
    """Character-level edit distance via space-optimized dynamic programming."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein_pct(original: str, candidate: str) -> float:
    """Edit distance as a percentage of the original length.

    dist / max(len(original), 1) * 100 -- 0.0 when identical, 100.0
    when every original character must be replaced.
    """
    return levenshtein_distance(original, candidate) / max(len(original), 1) * 100.0


def truncate_pair(
    original: str, candidate: str, max_chars: int = LEVENSHTEIN_MAX_CHARS
) -> tuple[str, str, bool]:
    """Truncate *both* texts to *max_chars* when the original is longer.

    Returns (original', candidate', truncated). Keeps Levenshtein
    cost bounded; the caller records a note when truncated is True.
    """
    if len(original) <= max_chars:
        return original, candidate, False
    return original[:max_chars], candidate[:max_chars], True


def length_drift(original: str, candidate: str) -> float:
    """Signed relative length change: (len(c) - len(o)) / max(len(o), 1)."""
    return (len(candidate) - len(original)) / max(len(original), 1)


def numbers_preserved(original: str, candidate: str) -> float:
    """Fraction of original numbers (regex \\d+) surviving in the candidate.

    Mirrors service/scripts/bench_synthid_text.py._numbers_preserved.
    """
    a = set(re.findall(r"\d+", original))
    if not a:
        return 1.0
    b = set(re.findall(r"\d+", candidate))
    return len(a & b) / len(a)


def urls_preserved(original: str, candidate: str) -> float:
    """Fraction of original URLs (regex https?://\\S+) surviving.

    Mirrors service/scripts/bench_synthid_text.py._urls_preserved.
    """
    a = set(re.findall(r"https?://\S+", original))
    if not a:
        return 1.0
    b = set(re.findall(r"https?://\S+", candidate))
    return len(a & b) / len(a)


def ppl_from_loss(loss: float) -> float:
    """Perplexity from a mean-token cross-entropy loss: exp(loss)."""
    return float(math.exp(loss))


# ---------------------------------------------------------------------------
# Metric math on top of loaded models (injected objects => unit-testable)
# ---------------------------------------------------------------------------


def compute_ppl(
    model: Any, tokenizer: Any, text: str, *, device: str = "cpu", max_length: int = 1024
) -> float:
    """Mean-token perplexity of *text* under a causal-LM model.

    The tokenizer must accept (text, return_tensors="pt",
    truncation=True, max_length=...) and return an input_ids /
    attention_mask dict. Labels are shifted internally by
    transformers, so the loss is the standard per-token NLL.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return ppl_from_loss(float(outputs.loss))


def compute_bertscore_f1(f1: Any) -> float:
    """Scalar F1 out of the BERTScore F1 tensor (shape [batch])."""
    return float(f1.reshape(-1)[0].item())


def compute_sbert_cosine(emb_a: Any, emb_b: Any) -> float:
    """Cosine similarity between two sentence embeddings."""
    import torch

    a = emb_a.flatten().unsqueeze(0).float()
    b = emb_b.flatten().unsqueeze(0).float()
    return float(torch.nn.functional.cosine_similarity(a, b).item())


def compute_rouge_l_fmeasure(result: Any) -> float:
    """ROUGE-L F1 out of a rouge-score Score (has .fmeasure)."""
    return float(result.fmeasure)


# ---------------------------------------------------------------------------
# Lazy loaders: one try/except each, cache on success, _FAILED on failure
# ---------------------------------------------------------------------------


def _get_ppl_models() -> tuple[Any, Any] | None:
    """Load (tokenizer, model) for gpt2-large once; None on failure."""
    if "ppl" in _MODEL_CACHE:
        return _MODEL_CACHE["ppl"]
    if "ppl" in _FAILED:
        return None
    model_id = "gpt2-large"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
    except Exception as exc:
        _FAILED.add("ppl")
        _warn("ppl", f"failed to load {model_id}: {exc}")
        return None
    _MODEL_CACHE["ppl"] = (model, tokenizer)
    return _MODEL_CACHE["ppl"]


def _get_bert_score_fn() -> Callable[..., Any] | None:
    """Load the bert_score score callable once; None on failure."""
    if "bertscore" in _MODEL_CACHE:
        return _MODEL_CACHE["bertscore"]
    if "bertscore" in _FAILED:
        return None
    try:
        from bert_score import score as bert_score_score  # type: ignore[import-not-found]
    except Exception as exc:
        _FAILED.add("bertscore")
        _warn("bertscore", f"bert-score unavailable: {exc}")
        return None
    _MODEL_CACHE["bertscore"] = bert_score_score
    return bert_score_score


def _get_rouge_scorer() -> Any | None:
    """Load a ROUGE-L scorer once (pure python, no nltk); None on failure."""
    if "rouge" in _MODEL_CACHE:
        return _MODEL_CACHE["rouge"]
    if "rouge" in _FAILED:
        return None
    try:
        from rouge_score import rouge_scorer  # type: ignore[import-not-found]

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except Exception as exc:
        _FAILED.add("rouge")
        _warn("rouge", f"rouge-score unavailable: {exc}")
        return None
    _MODEL_CACHE["rouge"] = scorer
    return scorer


def _get_sbert_model(device: str) -> Any | None:
    """Load all-MiniLM-L6-v2 on *device* once (per device); None on failure."""
    cache_key: Any = ("sbert", device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if "sbert" in _FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    except Exception as exc:
        _FAILED.add("sbert")
        _warn("sbert", f"failed to load all-MiniLM-L6-v2: {exc}")
        return None
    _MODEL_CACHE[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Per-metric compute wrappers
# ---------------------------------------------------------------------------


def _ppl_for_text(text: str, device: str) -> float | None:
    loaded = _get_ppl_models()
    if loaded is None:
        return None
    model, tokenizer = loaded
    return compute_ppl(model, tokenizer, text, device=device)


def _bertscore_for_pair(original: str, candidate: str, device: str) -> float | None:
    score_fn = _get_bert_score_fn()
    if score_fn is None:
        return None
    _, _, f1 = score_fn(
        [candidate],
        [original],
        lang="en",
        model_type="deberta-xlarge-mnli",
        rescale_with_baseline=True,
        device=device,
    )
    return compute_bertscore_f1(f1)


def _rouge_l_for_pair(original: str, candidate: str) -> float | None:
    scorer = _get_rouge_scorer()
    if scorer is None:
        return None
    result = scorer.score(original, candidate)
    return compute_rouge_l_fmeasure(result["rougeL"])


def _sbert_for_pair(original: str, candidate: str, device: str) -> float | None:
    model = _get_sbert_model(device)
    if model is None:
        return None
    embeddings = model.encode([original, candidate], convert_to_tensor=True)
    return compute_sbert_cosine(embeddings[0], embeddings[1])


# ---------------------------------------------------------------------------
# Warning log + row metric assembly
# ---------------------------------------------------------------------------


def _warn(metric: str | None, message: str) -> None:
    """Record a deduplicated warning (also surfaced on stderr by main)."""
    record: dict[str, str | None] = {"metric": metric, "message": message}
    if record not in _WARNINGS:
        _WARNINGS.append(record)


def _warnings() -> list[dict[str, str | None]]:
    return list(_WARNINGS)


def _safe(metric: str, fn: Callable[[], Any]) -> Any:
    """Run a model-backed computation; degrade to None instead of crashing."""
    try:
        return fn()
    except Exception as exc:
        _warn(metric, f"{metric}: computation failed: {exc}")
        return None


def _round4(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def compute_row_metrics(
    row: dict[str, Any], skip: set[str], device: str
) -> tuple[dict[str, float | None], list[str]]:
    """One input row's metrics plus any notes (e.g. Levenshtein truncation).

    skip holds MODEL_METRICS names to leave None; model metrics whose
    load already failed are also None (see _FAILED). Never raises for
    model problems -- every model-backed computation degrades via _safe.
    """
    metrics: dict[str, float | None] = {key: None for key in METRIC_KEYS}
    notes: list[str] = []
    original = row.get("original")
    candidate = row.get("candidate")
    if not isinstance(original, str) or not isinstance(candidate, str):
        _warn(None, "row missing string 'original'/'candidate'; all metrics None")
        notes.append("missing original/candidate; all metrics None")
        return metrics, notes

    metrics["length_drift"] = _round4(length_drift(original, candidate))
    metrics["numbers_preserved"] = _round4(numbers_preserved(original, candidate))
    metrics["urls_preserved"] = _round4(urls_preserved(original, candidate))

    o, c = original, candidate
    if len(original) > LEVENSHTEIN_MAX_CHARS:
        o, c, _ = truncate_pair(original, candidate, LEVENSHTEIN_MAX_CHARS)
        notes.append(f"levenshtein: both texts truncated to {LEVENSHTEIN_MAX_CHARS} chars")
    metrics["levenshtein_pct"] = _round4(levenshtein_pct(o, c))

    if "ppl" not in skip:
        metrics["ppl"] = _round4(_safe("ppl", lambda: _ppl_for_text(candidate, device)))
    if "bertscore" not in skip:
        metrics["bertscore"] = _round4(
            _safe("bertscore", lambda: _bertscore_for_pair(original, candidate, device))
        )
    if "rouge" not in skip:
        metrics["rouge_l"] = _round4(_safe("rouge", lambda: _rouge_l_for_pair(original, candidate)))
    if "sbert" not in skip:
        metrics["sbert_cosine"] = _round4(
            _safe("sbert", lambda: _sbert_for_pair(original, candidate, device))
        )
    return metrics, notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Streaming CLI: read input JSONL, append metric keys, write output JSONL."""
    parser = argparse.ArgumentParser(
        description=(
            "Per-row text-quality metrics (research/01-experiment-protocol.md §5.3): "
            "PPL (gpt2-large), BERTScore, ROUGE-L, SBERT cosine, Levenshtein %, "
            "length drift, number/URL survival."
        )
    )
    parser.add_argument("--input", required=True, help="input JSONL, one row per line")
    parser.add_argument("--out", required=True, help="output JSONL (input row + metric keys)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    parser.add_argument(
        "--device", default="cpu", help="torch device for model inference (default: cpu)"
    )
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated model metrics to leave None: ppl,bertscore,rouge,sbert",
    )
    parser.add_argument("--warnings-out", default=None, help="optional JSONL file for warnings")
    args = parser.parse_args(argv)

    skip = {part.strip() for part in args.skip.split(",") if part.strip()}
    unknown = skip - set(MODEL_METRICS)
    if unknown:
        parser.error(
            f"unknown --skip values: {', '.join(sorted(unknown))} "
            f"(valid: {', '.join(MODEL_METRICS)})"
        )
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    in_path = Path(args.input)
    if not in_path.is_file():
        parser.error(f"input file not found: {in_path}")

    _WARNINGS.clear()
    processed = 0
    with (
        in_path.open("r", encoding="utf-8") as src,
        Path(args.out).open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if args.limit is not None and processed >= args.limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _warn(None, f"skipping malformed JSON line: {exc}")
                continue
            if not isinstance(row, dict):
                _warn(None, "skipping non-object JSON row")
                continue
            metrics, notes = compute_row_metrics(row, skip, args.device)
            out_row = dict(row)
            out_row.update(metrics)
            if notes:
                existing = out_row.get("notes")
                if isinstance(existing, list):
                    out_row["notes"] = existing + notes
                elif existing is None:
                    out_row["notes"] = notes
                else:
                    out_row["notes"] = [existing, *notes]
            dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            processed += 1

    records = _warnings()
    for record in records:
        metric = record["metric"] or "row"
        print(f"warning: {metric}: {record['message']}", file=sys.stderr)
    if args.warnings_out:
        with Path(args.warnings_out).open("w", encoding="utf-8") as warn_file:
            for record in records:
                warn_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"processed {processed} rows; {len(records)} warnings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
