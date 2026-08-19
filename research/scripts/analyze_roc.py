#!/usr/bin/env python3
"""ROC-based detection metrics with an empirical null (gap 05-B1).

Computes AUROC and TPR@FPR for watermark detection scores, calibrating
every FPR threshold against the *empirical* null score distribution and
never against a parametric (e.g. standard-normal) assumption
(research/01-experiment-protocol.md §5.1-5.2). SynthID tournament
scores, in particular, are not standard normal, so z-score calibration
would be wrong; the empirical null is mandatory (protocol §5.2).

CLI
---
    python3 research/scripts/analyze_roc.py --scores scores.jsonl \
        --out metrics.json [--n-bootstrap 10000] [--seed 1] \
        [--fpr-targets 0.001,0.01,0.1] [--attack NAME]

scores.jsonl rows (one JSON object per line):

    {"condition": str, "attack": str, "kind": "watermarked"|"control"|"attacked",
     "score": float, "is_watermarked": bool, "ok": bool}

Rows with ok != true or a missing/non-finite score are skipped. For each
attack the signal distribution is kind == "watermarked" when the attack
is "none" (untouched originals) and kind == "attacked" otherwise; the
null distribution is kind == "control" under the same attack
(unwatermarked texts run through the same attack pipeline).

Conventions
-----------
- AUROC is rank-based (Mann-Whitney U with tie-averaged ranks), so no
  scikit-learn dependency is needed; ties are handled exactly. The value
  is None (plus a warning) when scores are degenerate (all identical) or
  a group is empty.
- For a target FPR alpha, the operating threshold is read off the
  empirical null: the smallest null-derived score t with
  P(null >= t) <= alpha — the ROC point whose FPR is the largest one not
  exceeding the budget (ties count toward the positive side). If no null
  score reaches FPR <= alpha (coarse null), the strictest threshold (max
  null score) is used and a warning is emitted.
- 95% bootstrap CIs (percentile interval) are computed over
  --n-bootstrap resamples of both signal and null with a seeded RNG;
  thresholds are re-derived from the resampled null, so the CI covers
  null-sampling variability too.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """1-based average ranks of *values*; tied entries share the mean rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.size, dtype=float)
    n = values.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auroc(signal: np.ndarray, null: np.ndarray) -> float | None:
    """Rank-based AUROC; None when undefined (empty group or constant scores)."""
    if signal.size == 0 or null.size == 0:
        return None
    if np.unique(np.concatenate([signal, null])).size <= 1:
        return None
    combined = np.concatenate([signal, null])
    ranks = _rankdata_average(combined)
    n_s, n_n = signal.size, null.size
    sum_signal_ranks = ranks[:n_s].sum()
    auc = (sum_signal_ranks - n_s * (n_s + 1) / 2.0) / (n_s * n_n)
    return float(auc)


def _fpr_thresholds(null: np.ndarray, alphas: list[float]) -> list[float]:
    """One operating threshold per target FPR, read off the empirical null.

    For each alpha the threshold is the smallest score t taken from the
    null itself with P(null >= t) <= alpha (ties count positive). When no
    null score fits the budget, the strictest null score (its max) is
    used, giving the smallest achievable FPR.
    """
    if null.size == 0:
        return [0.0] * len(alphas)
    sorted_null = np.sort(null)
    unique = np.unique(sorted_null)
    counts = null.size - np.searchsorted(sorted_null, unique, side="left")
    thresholds: list[float] = []
    for alpha in alphas:
        budget = math.floor(alpha * null.size)
        ok = counts <= budget
        if not ok.any():
            thresholds.append(float(unique[-1]))
        else:
            thresholds.append(float(unique[int(np.flatnonzero(ok)[0])]))
    return thresholds


def _roc_points(signal: np.ndarray, null: np.ndarray) -> list[list[float]]:
    """Empirical ROC points [[fpr, tpr], ...] (monotone, endpoints included)."""
    if signal.size == 0 or null.size == 0:
        return [[0.0, 0.0], [1.0, 1.0]]
    thresholds = np.unique(np.concatenate([signal, null]))[::-1]
    points: list[list[float]] = [[0.0, 0.0]]
    for t in thresholds:
        point = [float((null >= t).mean()), float((signal >= t).mean())]
        if point != points[-1]:
            points.append(point)
    if points[-1] != [1.0, 1.0]:
        points.append([1.0, 1.0])
    return points


def _as_score(row: dict[str, Any]) -> float | None:
    """Parse the row's score; None when missing or non-finite."""
    if "score" not in row:
        return None
    try:
        score = float(row["score"])
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def compute_roc_metrics(
    signal_scores: list[float],
    null_scores: list[float],
    fpr_targets: list[float],
    n_bootstrap: int = 10000,
    seed: int = 1,
) -> dict[str, Any]:
    """ROC metrics (AUROC, TPR@FPR, bootstrap CIs) for one attack.

    Args:
        signal_scores: detector scores of watermarked (or attacked) texts.
        null_scores: detector scores of unwatermarked controls, used as the
            empirical null for FPR calibration (protocol §5.2).
        fpr_targets: FPR targets in (0, 1] at which TPR is reported.
        n_bootstrap: number of bootstrap resamples for the 95% CIs.
        seed: RNG seed for the bootstrap resamples.

    Returns:
        A dict with n_signal, n_null, auroc, tpr_at_fpr (keyed by str(fpr)),
        auroc_ci95, tpr_ci95, roc_points and warnings.
    """
    if not fpr_targets:
        raise ValueError("fpr_targets must not be empty")
    for fpr in fpr_targets:
        if not 0.0 < fpr <= 1.0:
            raise ValueError(f"fpr target must be in (0, 1], got {fpr!r}")

    signal = np.asarray(signal_scores, dtype=float)
    null = np.asarray(null_scores, dtype=float)
    n_s, n_n = int(signal.size), int(null.size)

    warnings: list[str] = []
    if n_n < 10:
        warnings.append(f"n_null={n_n} < 10: empirical FPR calibration is coarse")
    if n_s == 0:
        warnings.append("no signal scores")
    if n_n == 0:
        warnings.append("no null scores; FPR cannot be calibrated from the empirical null")
    if n_s > 0 and n_n > 0 and np.unique(np.concatenate([signal, null])).size <= 1:
        warnings.append("degenerate: all scores identical; AUROC undefined (None)")

    auroc = _auroc(signal, null)
    if auroc is not None and not 0.0 <= auroc <= 1.0:
        warnings.append(f"auroc={auroc:.6f} outside [0, 1] (numerical noise)")

    tpr_at_fpr: dict[str, float | None] = {}
    if n_s > 0 and n_n > 0:
        for fpr, threshold in zip(fpr_targets, _fpr_thresholds(null, fpr_targets), strict=True):
            achieved = float((null >= threshold).mean())
            if achieved > fpr:
                warnings.append(
                    f"FPR target {fpr}: achieved FPR {achieved:.4f} "
                    "(null too coarse; strictest null threshold used)"
                )
            tpr_at_fpr[str(fpr)] = float((signal >= threshold).mean())
    else:
        tpr_at_fpr = {str(fpr): None for fpr in fpr_targets}

    auroc_ci95: list[float | None] = [None, None]
    tpr_ci95: dict[str, list[float | None]] = {str(fpr): [None, None] for fpr in fpr_targets}
    if n_bootstrap >= 2 and n_s > 0 and n_n > 0:
        rng = np.random.default_rng(seed)
        aucs: list[float] = []
        tpr_boot: dict[str, list[float]] = {str(fpr): [] for fpr in fpr_targets}
        for _ in range(n_bootstrap):
            sb = signal[rng.integers(0, n_s, size=n_s)]
            nb = null[rng.integers(0, n_n, size=n_n)]
            boot_auc = _auroc(sb, nb)
            if boot_auc is not None:
                aucs.append(boot_auc)
            for fpr, threshold in zip(fpr_targets, _fpr_thresholds(nb, fpr_targets), strict=True):
                tpr_boot[str(fpr)].append(float((sb >= threshold).mean()))
        if len(aucs) >= 2:
            auroc_ci95 = [float(x) for x in np.percentile(aucs, [2.5, 97.5])]
        else:
            warnings.append("bootstrap AUROC CI unavailable (resamples degenerate)")
        for fpr in fpr_targets:
            vals = tpr_boot[str(fpr)]
            if len(vals) >= 2:
                tpr_ci95[str(fpr)] = [float(x) for x in np.percentile(vals, [2.5, 97.5])]
    elif n_bootstrap < 2:
        warnings.append("bootstrap skipped (n_bootstrap < 2)")

    return {
        "n_signal": n_s,
        "n_null": n_n,
        "auroc": auroc,
        "tpr_at_fpr": tpr_at_fpr,
        "auroc_ci95": auroc_ci95,
        "tpr_ci95": tpr_ci95,
        "roc_points": _roc_points(signal, null),
        "warnings": warnings,
    }


def analyze_scores(
    rows: list[dict[str, Any]],
    fpr_targets: list[float],
    n_bootstrap: int = 10000,
    seed: int = 1,
    attack: str | None = None,
) -> dict[str, Any]:
    """Group score rows by attack and compute ROC metrics per attack.

    Rows with ok != true or a missing/non-finite score are skipped. For
    each attack, signal is kind == "watermarked" (attack "none") or
    kind == "attacked" (any other attack); null is kind == "control" under
    the same attack. When *attack* is given, only that attack is reported.
    """
    condition: str | None = None
    valid: list[dict[str, Any]] = []
    for row in rows:
        if condition is None and row.get("condition") is not None:
            condition = row["condition"]
        if row.get("ok") is not True:
            continue
        score = _as_score(row)
        if score is None:
            continue
        parsed = dict(row)
        parsed["score"] = score
        valid.append(parsed)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        groups.setdefault(str(row.get("attack")), []).append(row)
    if attack is not None:
        groups = {name: group for name, group in groups.items() if name == attack}

    per_attack: dict[str, dict[str, Any]] = {}
    for name, group in groups.items():
        signal_kind = "watermarked" if name == "none" else "attacked"
        signal = [row["score"] for row in group if row.get("kind") == signal_kind]
        null = [row["score"] for row in group if row.get("kind") == "control"]
        per_attack[name] = compute_roc_metrics(
            signal,
            null,
            fpr_targets,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return {"condition": condition, "per_attack": per_attack}


def _parse_fpr_targets(raw: str) -> list[float]:
    """Parse a comma-separated --fpr-targets value into validated floats."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    targets = [float(part) for part in parts]
    if not targets:
        raise ValueError("--fpr-targets must contain at least one value")
    for target in targets:
        if not 0.0 < target <= 1.0:
            raise ValueError(f"FPR target {target} must be in (0, 1]")
    return targets


def _load_rows(path: str) -> list[dict[str, Any]]:
    """Load scores.jsonl into a list of dicts; raise on malformed lines."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            content = line.strip()
            if not content:
                continue
            try:
                rows.append(json.loads(content))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: compute per-attack ROC metrics and write JSON."""
    parser = argparse.ArgumentParser(
        description="ROC-based detection metrics with an empirical null (protocol §5.1-5.2)."
    )
    parser.add_argument("--scores", required=True, help="scores.jsonl input file")
    parser.add_argument("--out", required=True, help="JSON output file path")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="number of bootstrap resamples for the 95 percent CIs (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=1, help="bootstrap RNG seed")
    parser.add_argument(
        "--fpr-targets",
        default="0.001,0.01,0.1",
        help="comma-separated FPR targets in (0, 1] (default: 0.001,0.01,0.1)",
    )
    parser.add_argument(
        "--attack",
        default=None,
        help="restrict analysis to one attack name (default: all)",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        print("error: --seed must be >= 0", file=sys.stderr)
        return 1
    try:
        fpr_targets = _parse_fpr_targets(args.fpr_targets)
        rows = _load_rows(args.scores)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = analyze_scores(
        rows,
        fpr_targets,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        attack=args.attack,
    )
    try:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
        return 1

    print(f"condition: {result['condition']}")
    for name, metrics in result["per_attack"].items():
        auroc = "n/a" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
        print(
            f"  {name!r}: n_signal={metrics['n_signal']} n_null={metrics['n_null']} auroc={auroc}"
        )
        for fpr, tpr in metrics["tpr_at_fpr"].items():
            tpr_str = "n/a" if tpr is None else f"{tpr:.4f}"
            print(f"    TPR@{fpr} = {tpr_str}")
        for warning in metrics["warnings"]:
            print(f"    warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
