"""Tests for research/scripts/analyze_roc.py (gap 05-B1).

Covers the rank-based AUROC, TPR@FPR read off the empirical null, seeded
bootstrap CIs, degenerate inputs, and attack grouping in analyze_scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_roc import analyze_scores, compute_roc_metrics


def test_perfect_separation_auroc_is_one():
    signal = list(range(100, 200))
    null = list(range(100))
    res = compute_roc_metrics(signal, null, [0.1], n_bootstrap=10, seed=1)
    assert res["auroc"] == 1.0
    assert res["tpr_at_fpr"]["0.1"] == 1.0
    assert res["roc_points"][-1] == [1.0, 1.0]


def test_identical_distributions_auroc_half():
    rng = np.random.default_rng(7)
    scores = list(rng.normal(size=200))
    res = compute_roc_metrics(scores, scores, [0.1], n_bootstrap=10, seed=1)
    # The two groups are the exact same multiset: tie-averaged ranks put
    # half the rank mass on each side, so AUROC is exactly 0.5.
    assert res["auroc"] == 0.5


def test_tpr_at_fpr_read_off_empirical_null():
    # Hand-built tiny example: null = [0,1,2,3,4], signal = [1,2,3,4,5].
    # For target FPR alpha the threshold is the smallest null score t with
    # P(null >= t) <= alpha (>=, ties positive):
    #   alpha=0.2 -> t=4 (fpr 0.2) -> tpr = P(signal >= 4) = 2/5 = 0.4
    #   alpha=0.4 -> t=3 (fpr 0.4) -> tpr = P(signal >= 3) = 3/5 = 0.6
    #   alpha=0.6 -> t=2 (fpr 0.6) -> tpr = P(signal >= 2) = 4/5 = 0.8
    signal = [1, 2, 3, 4, 5]
    null = [0, 1, 2, 3, 4]
    res = compute_roc_metrics(signal, null, [0.2, 0.4, 0.6], n_bootstrap=50, seed=1)
    assert res["tpr_at_fpr"] == {"0.2": 0.4, "0.4": 0.6, "0.6": 0.8}
    # Rank-based AUROC for this example is 17/25 = 0.68.
    assert res["auroc"] == 0.68
    # ROC points are monotone and include both endpoints.
    fprs = [point[0] for point in res["roc_points"]]
    tprs = [point[1] for point in res["roc_points"]]
    assert fprs == sorted(fprs)
    assert tprs == sorted(tprs)
    assert res["roc_points"][0] == [0.0, 0.0]
    assert res["roc_points"][-1] == [1.0, 1.0]


def test_bootstrap_ci_bounds_and_containment():
    rng = np.random.default_rng(42)
    signal = list(rng.normal(loc=1.0, size=200))
    null = list(rng.normal(size=200))
    res = compute_roc_metrics(signal, null, [0.1, 0.5], n_bootstrap=500, seed=3)
    lo, hi = res["auroc_ci95"]
    assert lo <= hi
    assert lo <= res["auroc"] <= hi
    for fpr in ("0.1", "0.5"):
        flo, fhi = res["tpr_ci95"][fpr]
        assert flo <= fhi
        assert flo <= res["tpr_at_fpr"][fpr] <= fhi


def test_determinism_with_fixed_seed():
    rng = np.random.default_rng(11)
    signal = list(rng.normal(loc=0.5, size=150))
    null = list(rng.normal(size=150))
    kwargs = {"n_bootstrap": 300, "seed": 9}
    first = compute_roc_metrics(signal, null, [0.01, 0.1], **kwargs)
    second = compute_roc_metrics(signal, null, [0.01, 0.1], **kwargs)
    assert first["auroc"] == second["auroc"]
    assert first["tpr_at_fpr"] == second["tpr_at_fpr"]
    assert first["auroc_ci95"] == second["auroc_ci95"]
    assert first["tpr_ci95"] == second["tpr_ci95"]
    assert first["roc_points"] == second["roc_points"]


def test_degenerate_all_equal_scores_yields_none_and_warning():
    res = compute_roc_metrics([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.1], n_bootstrap=50, seed=1)
    assert res["auroc"] is None
    assert any("degenerate" in warning.lower() for warning in res["warnings"])


def test_missing_null_yields_none_and_warning():
    res = compute_roc_metrics([1.0, 2.0, 3.0], [], [0.1], n_bootstrap=50, seed=1)
    assert res["auroc"] is None
    assert res["tpr_at_fpr"]["0.1"] is None
    assert any("no null scores" in warning for warning in res["warnings"])


def test_small_null_warns():
    res = compute_roc_metrics([1.0, 2.0, 3.0], [0.5, 1.5], [0.1], n_bootstrap=50, seed=1)
    assert any("n_null=2 < 10" in warning for warning in res["warnings"])


def test_analyze_scores_groups_attacks():
    rows = [
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "watermarked",
            "score": 3.0,
            "is_watermarked": True,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "watermarked",
            "score": 4.0,
            "is_watermarked": True,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "control",
            "score": 0.0,
            "is_watermarked": False,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "control",
            "score": 1.0,
            "is_watermarked": False,
            "ok": True,
        },
        # kind "attacked" under attack "none" must NOT count as signal there.
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "attacked",
            "score": 9.0,
            "is_watermarked": True,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "paraphrase",
            "kind": "attacked",
            "score": 2.0,
            "is_watermarked": True,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "paraphrase",
            "kind": "attacked",
            "score": 2.5,
            "is_watermarked": True,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "paraphrase",
            "kind": "control",
            "score": 0.5,
            "is_watermarked": False,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "paraphrase",
            "kind": "control",
            "score": 1.5,
            "is_watermarked": False,
            "ok": True,
        },
        # kind "watermarked" under a non-none attack must NOT count as signal.
        {
            "condition": "en-core",
            "attack": "paraphrase",
            "kind": "watermarked",
            "score": 5.0,
            "is_watermarked": True,
            "ok": True,
        },
        # Invalid rows must be skipped: ok=false, missing score, bad score.
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "control",
            "score": 0.0,
            "is_watermarked": False,
            "ok": False,
        },
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "control",
            "is_watermarked": False,
            "ok": True,
        },
        {
            "condition": "en-core",
            "attack": "none",
            "kind": "control",
            "score": "oops",
            "is_watermarked": False,
            "ok": True,
        },
    ]
    res = analyze_scores(rows, [0.1], n_bootstrap=50, seed=1)
    assert res["condition"] == "en-core"
    assert set(res["per_attack"]) == {"none", "paraphrase"}
    none_metrics = res["per_attack"]["none"]
    assert none_metrics["n_signal"] == 2
    assert none_metrics["n_null"] == 2
    assert none_metrics["auroc"] == 1.0
    par_metrics = res["per_attack"]["paraphrase"]
    assert par_metrics["n_signal"] == 2
    assert par_metrics["n_null"] == 2
    # The --attack filter restricts per_attack to one key.
    filtered = analyze_scores(rows, [0.1], n_bootstrap=50, seed=1, attack="paraphrase")
    assert set(filtered["per_attack"]) == {"paraphrase"}


def test_analyze_scores_empty_and_condition_missing():
    res = analyze_scores([], [0.1], n_bootstrap=10, seed=1)
    assert res["condition"] is None
    assert res["per_attack"] == {}
