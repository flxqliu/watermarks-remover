#!/usr/bin/env python3
"""Paper figure generators (gap 05-B3): F1-F6 per research/02-paper-outline.md sec 6.

CLI:
    python3 research/scripts/make_figures.py --results-dir DIR --out-dir DIR \
        [--format png|pdf] [--dpi 200]

Reads the same results layout as make_tables.py (one directory per cell):

    DIR/<condition>/metrics.json    per-attack AUROC/TPR + optional
                                    "roc_points" per attack (F2)
    DIR/<condition>/quality.jsonl   quality metrics per attack (F3)
    DIR/<condition>/attacked.jsonl  before/after texts + cost (F5)
    DIR/<condition>/scores.jsonl    detector scores (F5, optional)

Writes out-dir/figures/f1.<fmt> ... f6.<fmt>. F1 (pipeline diagram) and F6
(policy timeline) are static schematics; F2-F5 read data and degrade to a
"no data" figure when their inputs are missing.

matplotlib is imported lazily inside main(): if it is not installed the
script prints a clear warning and exits 0 without writing any figures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEME_ORDER: tuple[str, ...] = (
    "kgw-d1",
    "kgw-d2",
    "kgw-d4",
    "synthid",
    "exp",
    "unigram",
    "sir",
)
SCHEME_LABELS: dict[str, str] = {
    "kgw-d1": "KGW (gamma=.25, delta=1)",
    "kgw-d2": "KGW (gamma=.5, delta=2)",
    "kgw-d4": "KGW (gamma=.5, delta=4)",
    "synthid": "SynthID-Text",
    "exp": "EXP (Gumbel)",
    "unigram": "Unigram",
    "sir": "SIR",
}
ATTACK_LABELS: dict[str, str] = {
    "none": "A0 none",
    "layerA": "A1 layerA",
    "paraphrase:1": "A2 paraphrase:1",
    "paraphrase:3": "A3 paraphrase:3",
    "backtranslate:de": "A4 backtranslate:de",
    "structural": "A5 structural",
    "humanize": "A6 humanize",
    "cheap": "A7 cheap",
    "layerA+paraphrase:3": "A8 layerA+paraphrase:3",
}
LAYERED_ATTACK = "layerA+paraphrase:3"

_RESERVED_TOP: frozenset[str] = frozenset(
    {
        "roc_points",
        "quality",
        "config",
        "meta",
        "cell",
        "condition",
        "manifest",
        "generated_at",
        "timestamp",
    }
)
_CONDITION_RE = re.compile(
    r"^(?P<scheme>.+?)-L(?P<length>\d+)-T(?P<temp>[0-9]+(?:\.[0-9]+)?)"
    r"-(?P<language>[a-z]{2})-s(?P<seed>\d+)-p(?P<prompt>\d+)$"
)


@dataclass(frozen=True)
class Condition:
    """One results cell with the artifacts the figures need."""

    dir: Path
    name: str
    scheme: str | None
    length: int | None
    temp: float | None
    language: str | None
    metrics: dict[str, dict[str, Any]]
    roc_points: dict[str, tuple[list[float], list[float]]]
    quality: list[dict[str, Any]]
    attacked: list[dict[str, Any]]
    scores: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Results loading (mirrors make_tables.py; also loads scores.jsonl)
# ---------------------------------------------------------------------------


def _scheme_from_name(name: str) -> str | None:
    for scheme in SCHEME_ORDER:
        if name == scheme or name.startswith(scheme + "-"):
            return scheme
    return None


def _num_in_name(name: str, prefix: str) -> int | None:
    m = re.search(rf"{re.escape(prefix)}(\d+)", name)
    return int(m.group(1)) if m else None


def _float_in_name(name: str, prefix: str) -> float | None:
    m = re.search(rf"{re.escape(prefix)}([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None


def _lang_in_name(name: str) -> str | None:
    m = re.search(r"-(en|de|fr|es)-", name)
    return m.group(1) if m else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        return []
    return rows


def _normalize_metrics(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("attack"):
                out[str(item["attack"])] = {k: v for k, v in item.items() if k != "attack"}
        return out
    if not isinstance(raw, dict):
        return {}
    for wrapper in ("attacks", "results", "per_attack"):
        if isinstance(raw.get(wrapper), dict):
            return _normalize_metrics(raw[wrapper])
    return {str(k): v for k, v in raw.items() if isinstance(v, dict) and k not in _RESERVED_TOP}


def _roc_pair(pts: Any) -> tuple[list[float], list[float]] | None:
    if isinstance(pts, dict):
        fpr, tpr = pts.get("fpr"), pts.get("tpr")
        if (
            isinstance(fpr, (list, tuple))
            and isinstance(tpr, (list, tuple))
            and len(fpr) == len(tpr)
            and len(fpr) > 1
        ):
            return [float(x) for x in fpr], [float(x) for x in tpr]
        return None
    if isinstance(pts, (list, tuple)) and len(pts) == 2:
        a, b = pts[0], pts[1]
        if (
            isinstance(a, (list, tuple))
            and isinstance(b, (list, tuple))
            and len(a) == len(b)
            and len(a) > 1
        ):
            return [float(x) for x in a], [float(x) for x in b]
    if isinstance(pts, (list, tuple)):
        try:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
        except (TypeError, ValueError, IndexError):
            return None
        if len(xs) == len(ys) and len(xs) > 1:
            return xs, ys
    return None


def _normalize_roc(raw: Any) -> dict[str, tuple[list[float], list[float]]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[list[float], list[float]]] = {}
    for attack, pts in raw.items():
        pair = _roc_pair(pts)
        if pair is not None:
            out[str(attack)] = pair
    return out


def _load_condition(path: Path) -> Condition | None:
    name = path.name
    m = _CONDITION_RE.match(name)
    if m:
        scheme = m.group("scheme")
        length = int(m.group("length"))
        temp = float(m.group("temp"))
        language = m.group("language")
    else:
        scheme = _scheme_from_name(name)
        length = _num_in_name(name, "L")
        temp = _float_in_name(name, "T")
        language = _lang_in_name(name)

    metrics: dict[str, dict[str, Any]] = {}
    roc: dict[str, tuple[list[float], list[float]]] = {}
    metrics_path = path / "metrics.json"
    if metrics_path.is_file():
        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict):
            roc.update(_normalize_roc(raw.get("roc_points")))
        metrics = _normalize_metrics(raw)
        for attack, attack_metrics in metrics.items():
            pair = _roc_pair(attack_metrics.get("roc_points"))
            if pair is not None:
                roc.setdefault(attack, pair)

    quality = _read_jsonl(path / "quality.jsonl")
    attacked = _read_jsonl(path / "attacked.jsonl")
    scores = _read_jsonl(path / "scores.jsonl")
    if not metrics and not quality and not attacked and not scores:
        return None
    return Condition(
        dir=path,
        name=name,
        scheme=scheme,
        length=length,
        temp=temp,
        language=language,
        metrics=metrics,
        roc_points=roc,
        quality=quality,
        attacked=attacked,
        scores=scores,
    )


def _load_conditions(results_dir: Path) -> list[Condition]:
    if not results_dir.is_dir():
        return []
    conditions: list[Condition] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        cond = _load_condition(child)
        if cond is not None:
            conditions.append(cond)
    return conditions


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _auroc(m: Mapping[str, Any]) -> float | None:
    for key in ("auroc", "auc", "roc_auc"):
        v = m.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    sub = m.get("metrics")
    if isinstance(sub, Mapping):
        return _auroc(sub)
    return None


def _fpr_in_key(key: str) -> float | None:
    flat = re.sub(r"[\s_@]+", "", key).lower()
    m = re.search(r"tprfpr([0-9.]+)", flat)
    if m:
        return float(m.group(1))
    m = re.search(r"tprat([0-9.]+)pct", flat)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _tpr_at_fpr(m: Mapping[str, Any], fpr: float = 0.01) -> float | None:
    t = m.get("tpr_at_fpr")
    if isinstance(t, dict):
        return _tpr_from_dict(t, fpr)
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    for key, val in m.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        want = _fpr_in_key(str(key))
        if want is not None and abs(want - fpr) < 1e-9:
            return float(val)
    sub = m.get("metrics")
    if isinstance(sub, Mapping):
        return _tpr_at_fpr(sub, fpr)
    return None


def _tpr_from_dict(d: Mapping[str, Any], fpr: float) -> float | None:
    for key, val in d.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        k = str(key).strip()
        pct = k.endswith("%")
        num = k[:-1] if pct else k
        try:
            x = float(num)
        except ValueError:
            continue
        if pct:
            x = x / 100.0
        if abs(x - fpr) < 1e-9:
            return float(val)
    return None


def _mean_metric(
    conds: Sequence[Condition], attack: str, metric: str, fpr: float = 0.01
) -> float | None:
    """Mean *metric* ('auroc' | 'tpr') for *attack* across *conds*."""
    values: list[float] = []
    for cond in conds:
        m = cond.metrics.get(attack)
        if not m:
            continue
        v = _auroc(m) if metric == "auroc" else _tpr_at_fpr(m, fpr)
        if v is not None:
            values.append(v)
    return sum(values) / len(values) if values else None


def _quality_num(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    sub = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else None
    for src in (row, sub):
        if not src:
            continue
        for key in aliases:
            v = src.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _no_data_ax(ax: Any) -> None:
    """Turn *ax* into a graceful 'no data' placeholder."""
    ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=14, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _mean_roc(np: Any, pairs: Sequence[tuple[list[float], list[float]]]) -> tuple[Any, Any] | None:
    """Interpolate ROC pairs onto a common FPR grid and average the TPRs."""
    if not pairs:
        return None
    grid = np.linspace(0.0, 1.0, 101)
    tprs = []
    for fpr, tpr in pairs:
        order = np.argsort(fpr)
        f = np.asarray(fpr, dtype=float)[order]
        t = np.asarray(tpr, dtype=float)[order]
        tprs.append(np.interp(grid, f, t))
    return grid, np.mean(tprs, axis=0)


def _truncate(text: str, limit: int = 600) -> str:
    """Collapse whitespace and truncate *text* to ~*limit* chars (word-safe)."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut + " ... [truncated at ~600 chars; no redaction beyond truncation]"


def _wrap(text: str, width: int = 74) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else "(empty)"


def _pick_case_row(cond: Condition) -> tuple[dict[str, Any], str]:
    for attack in (LAYERED_ATTACK, "paraphrase:3"):
        for row in cond.attacked:
            if row.get("attack") == attack:
                return row, attack
    row = cond.attacked[0]
    return row, str(row.get("attack", "unknown"))


def _detector_scores(cond: Condition, attack: str) -> tuple[str | None, str | None]:
    """Best-effort before/after detector scores for *attack*."""

    def scan(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
        before = after = None
        for key in ("score_before", "before_score", "original_score", "score_original"):
            v = row.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                before = f"{v:.3f}"
                break
        for key in ("score_after", "after_score", "candidate_score", "score_candidate"):
            v = row.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                after = f"{v:.3f}"
                break
        if after is None:
            for key in ("score", "detect_score"):
                v = row.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    after = f"{v:.3f}"
                    break
        return before, after

    for row in cond.attacked:
        if row.get("attack") == attack:
            before, after = scan(row)
            if before is not None or after is not None:
                return before, after
    for row in cond.scores:
        if row.get("attack") in (attack, None):
            before, after = scan(row)
            if before is not None or after is not None:
                return before, after
    return None, None


# ---------------------------------------------------------------------------
# Figure builders (each returns a matplotlib Figure)
# ---------------------------------------------------------------------------


def figure_f1(plt: Any) -> Any:
    """F1 (static): pipeline diagram -- Layer A + Layer B + feedback loop."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("pipeline stage (left to right)")
    ax.set_ylabel("")
    ax.set_title(
        "F1 - Layered removal pipeline: Layer A formatting cleanup + "
        "Layer B statistical rewrite with detection-feedback loop"
    )

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012", linewidth=1.5, edgecolor=ec, facecolor=fc
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(
        x1: float, y1: float, x2: float, y2: float, rad: float = 0.0, color: str = "black"
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", lw=1.5, color=color, connectionstyle=f"arc3,rad={rad}"
            ),
        )

    box(
        0.02,
        0.62,
        0.17,
        0.22,
        "Watermarked text\n(input; same-config\nscores known)",
        "#e0f2fe",
        "#0c4a6e",
    )
    box(
        0.25,
        0.62,
        0.20,
        0.22,
        "Layer A - formatting cleanup\nUnicode/invisible chars, bidi,\ntag removal (clean_text.py)",
        "#dbeafe",
        "#1e40af",
    )
    box(
        0.51,
        0.62,
        0.20,
        0.22,
        "Layer B - statistical rewrite\nparaphrase / back-translate /\nstructural / humanize",
        "#dbeafe",
        "#1e40af",
    )
    box(0.77, 0.62, 0.20, 0.22, "Attacked text (candidate)\noutput", "#f0fdf4", "#166534")
    box(
        0.40,
        0.10,
        0.32,
        0.22,
        "Detector (same-config MarkLLM)\nwatermark score on candidate",
        "#fef3c7",
        "#92400e",
    )
    arrow(0.19, 0.73, 0.25, 0.73)
    arrow(0.45, 0.73, 0.51, 0.73)
    arrow(0.71, 0.73, 0.77, 0.73)
    arrow(0.87, 0.62, 0.60, 0.32, rad=0.22)  # candidate -> detector
    arrow(0.40, 0.32, 0.56, 0.62, rad=0.22)  # detector -> Layer B feedback
    ax.text(
        0.44,
        0.015,
        "detection feedback: if still watermarked, rewrite again "
        "(adaptive; early stop when detection passes)",
        ha="center",
        fontsize=8,
        color="#92400e",
    )
    fig.tight_layout()
    return fig


def figure_f2(plt: Any, np: Any, conditions: Sequence[Condition]) -> Any:
    """F2: ROC curves pre/post per scheme (mean over cells with data)."""
    schemes = [s for s in SCHEME_ORDER if any(c.scheme == s for c in conditions)]
    post = next(
        (a for a in (LAYERED_ATTACK, "paraphrase:3") if any(a in c.roc_points for c in conditions)),
        None,
    )
    if not schemes:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("F2 - ROC curves pre/post per scheme")
        _no_data_ax(ax)
        fig.tight_layout()
        return fig
    cols = 2
    rows = (len(schemes) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= len(schemes):
            ax.set_visible(False)
            continue
        scheme = schemes[i]
        ax.set_title(SCHEME_LABELS[scheme], fontsize=10)
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8)
        pre_pairs = [
            c.roc_points["none"]
            for c in conditions
            if c.scheme == scheme and "none" in c.roc_points
        ]
        post_pairs = [
            c.roc_points[post]
            for c in conditions
            if c.scheme == scheme and post and post in c.roc_points
        ]
        mean_pre = _mean_roc(np, pre_pairs)
        mean_post = _mean_roc(np, post_pairs)
        if mean_pre is None and mean_post is None:
            _no_data_ax(ax)
        else:
            if mean_pre is not None:
                ax.plot(mean_pre[0], mean_pre[1], "-", color="tab:blue", lw=1.8, label="pre-attack")
            if mean_post is not None:
                ax.plot(
                    mean_post[0],
                    mean_post[1],
                    "-",
                    color="tab:red",
                    lw=1.8,
                    label="post-attack (A+B)",
                )
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        "F2 - ROC curves pre/post attack per scheme (mean over cells; dashed = chance)", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure_f3(plt: Any, conditions: Sequence[Condition]) -> Any:
    """F3: quality-detectability Pareto frontier (PPL vs AUROC per attack)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("F3 - Quality-detectability Pareto frontier (PPL vs AUROC)")
    ax.set_xlabel("PPL (gpt2-large; lower is better)")
    ax.set_ylabel("AUROC (higher is better)")
    ref = next((c for c in conditions if c.quality and c.metrics), None)
    if ref is None:
        _no_data_ax(ax)
        fig.tight_layout()
        return fig
    by_attack: dict[str, list[dict[str, Any]]] = {}
    for row in ref.quality:
        if isinstance(row, dict) and row.get("attack"):
            by_attack.setdefault(str(row["attack"]), []).append(row)
    points: list[tuple[str, float, float]] = []
    used_delta = False
    for attack, rows in by_attack.items():
        m = ref.metrics.get(attack)
        if not m:
            continue
        auroc = _auroc(m)
        ppl: float | None = None
        for r in rows:
            v = _quality_num(r, ("ppl", "perplexity", "candidate_ppl"))
            if v is None:
                v = _quality_num(r, ("ppl_delta", "perplexity_delta"))
                used_delta = v is not None
            if v is not None:
                ppl = v
                break
        if auroc is not None and ppl is not None:
            points.append((attack, ppl, auroc))
    if not points:
        _no_data_ax(ax)
    else:
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.scatter(xs, ys, s=60, color="tab:blue", zorder=3)
        for attack, x, y in points:
            ax.annotate(
                ATTACK_LABELS.get(attack, attack),
                (x, y),
                fontsize=8,
                xytext=(6, 6),
                textcoords="offset points",
            )
        ax.set_xlim(left=min(xs) * 0.9)
        if used_delta:
            ax.text(
                0.02,
                0.02,
                "x = PPL delta when absolute PPL is not recorded",
                transform=ax.transAxes,
                fontsize=8,
                color="0.4",
            )
    fig.tight_layout()
    return fig


def figure_f4(plt: Any, conditions: Sequence[Condition], attack: str) -> Any:
    """F4: TPR@1%FPR vs KGW strength (delta), with length as line style."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"F4 - TPR@1%FPR vs watermark strength (KGW), by length (attack: {attack})")
    ax.set_xlabel("KGW strength delta (gamma=0.25 for delta=1, else 0.5)")
    ax.set_ylabel("TPR@1%FPR (mean over cells)")
    kgw = [
        c for c in conditions if c.scheme in ("kgw-d1", "kgw-d2", "kgw-d4") and c.language == "en"
    ]
    deltas = {"kgw-d1": 1, "kgw-d2": 2, "kgw-d4": 4}
    styles = {100: ("-", "o"), 300: ("--", "s"), 500: (":", "^")}
    plotted = False
    for length in (100, 300, 500):
        xs: list[float] = []
        ys: list[float] = []
        for scheme in ("kgw-d1", "kgw-d2", "kgw-d4"):
            conds = [c for c in kgw if c.scheme == scheme and c.length == length and c.temp == 0.7]
            if not conds:
                conds = [c for c in kgw if c.scheme == scheme and c.length == length]
            v = _mean_metric(conds, attack, "tpr")
            if v is not None:
                xs.append(float(deltas[scheme]))
                ys.append(v)
        if xs:
            linestyle, marker = styles[length]
            ax.plot(xs, ys, linestyle=linestyle, marker=marker, lw=1.8, label=f"length {length}")
            plotted = True
    if not plotted:
        _no_data_ax(ax)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["delta=1\ngamma=.25", "delta=2\ngamma=.5", "delta=4\ngamma=.5"])
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def figure_f5(plt: Any, conditions: Sequence[Condition]) -> Any:
    """F5: case study -- before/after text with detector scores."""
    fig = plt.figure(figsize=(13, 6))
    cond = next((c for c in conditions if c.attacked), None)
    if cond is None:
        ax = fig.add_subplot(111)
        ax.set_title("F5 - Case study: before/after with detector scores")
        _no_data_ax(ax)
        fig.tight_layout()
        return fig
    row, attack = _pick_case_row(cond)
    before = _truncate(row.get("original", ""), 600)
    after = _truncate(row.get("candidate", ""), 600)
    score_before, score_after = _detector_scores(cond, attack)
    ax1, ax2 = fig.subplots(1, 2)
    for ax in (ax1, ax2):
        ax.axis("off")
    ax1.set_title(f"Before (original, watermarked) - detector score: {score_before or 'n/a'}")
    ax2.set_title(f"After ({attack}) - detector score: {score_after or 'n/a'}")
    ax1.text(
        0.02,
        0.98,
        _wrap(before),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        wrap=True,
    )
    ax2.text(
        0.02,
        0.98,
        _wrap(after),
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        wrap=True,
    )
    fig.suptitle(
        "F5 - Case study: before/after (600-char window; no redaction "
        "beyond truncation) with detector scores"
    )
    fig.tight_layout()
    return fig


def figure_f6(plt: Any) -> Any:
    """F6 (static): policy timeline -- EU AI Act Art. 50 vs measured collapse."""
    from datetime import date

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(11, 5))
    events = [
        (date(2023, 1, 1), "KGW watermark proposed\n(arXiv 2301.10226)", "context", 1),
        (date(2024, 5, 21), "EU AI Act adopted\n(Regulation (EU) 2024/1689)", "policy", 1),
        (date(2026, 8, 2), "Art. 50 in force:\ntransparency obligations", "policy", 1),
        (date(2026, 8, 2), "measured collapse:\nKGW-class under layered attack", "collapse", -1),
        (date(2026, 8, 15), "SynthID-Text API\nretired (Google)", "context", -1),
    ]
    colors = {"policy": "tab:red", "collapse": "darkred", "context": "tab:blue"}
    ax.axhline(0.0, color="0.25", lw=1.2)
    for d, label, kind, side in events:
        x = mdates.date2num(d)
        color = colors[kind]
        ax.plot([x, x], [0, side * 0.32], color=color, lw=1.8)
        ax.scatter([x], [0], color=color, s=40, zorder=5)
        ax.text(
            x,
            side * 0.36,
            label,
            ha="center",
            va="bottom" if side > 0 else "top",
            fontsize=8,
            color=color,
        )
    ax.axvspan(
        mdates.date2num(date(2026, 7, 15)),
        mdates.date2num(date(2026, 8, 20)),
        color="red",
        alpha=0.08,
        label="Art. 50 in force + measured collapse",
    )
    ax.set_xlim(mdates.date2num(date(2022, 9, 1)), mdates.date2num(date(2027, 6, 1)))
    ax.set_ylim(-0.75, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("date")
    ax.set_ylabel("policy / measurement")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "F6 - Policy timeline: EU AI Act Art. 50 (in force 2026-08-02) "
        "vs measured watermark collapse (this work)"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path, required=True, help="results layout root: DIR/<condition>/"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output root; figures are written to OUT/figures/",
    )
    p.add_argument("--format", choices=("png", "pdf"), default="png")
    p.add_argument(
        "--dpi", type=int, default=200, help="raster resolution for PNG output (default: 200)"
    )
    p.add_argument(
        "--attack", default=LAYERED_ATTACK, help="attack used by F4 (default: %(default)s)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import matplotlib
    except ImportError:
        print(
            "warning: matplotlib is not installed; skipping figure generation (F1-F6).",
            file=sys.stderr,
        )
        return 0
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    conditions = _load_conditions(args.results_dir)
    suffix = f".{args.format}"
    builders: list[tuple[str, Any]] = [
        ("f1", figure_f1(plt)),
        ("f2", figure_f2(plt, np, conditions)),
        ("f3", figure_f3(plt, conditions)),
        ("f4", figure_f4(plt, conditions, args.attack)),
        ("f5", figure_f5(plt, conditions)),
        ("f6", figure_f6(plt)),
    ]
    for name, fig in builders:
        fig.savefig(figures_dir / f"{name}{suffix}", dpi=args.dpi)
        plt.close(fig)
    print(f"wrote {len(builders)} figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
