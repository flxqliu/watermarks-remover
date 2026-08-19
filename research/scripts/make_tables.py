#!/usr/bin/env python3
"""Paper table generators (gap 05-B3): T1-T7 per research/02-paper-outline.md sec 6.

CLI:
    python3 research/scripts/make_tables.py --results-dir DIR --out-dir DIR

Reads the results layout emitted by research/scripts/run_experiments.py
(one directory per cell, named e.g. 'kgw-d2-L300-T0.7-en-s1-p0'):

    DIR/<condition>/metrics.json    per-attack {"auroc", "tpr_at_fpr"}, plus
                                    optional per-attack "roc_points"
    DIR/<condition>/scores.jsonl    detector scores per (doc, attack)
    DIR/<condition>/attacked.jsonl  per-attack rows with fields attack,
                                    original, candidate, stats, seconds, usd
    DIR/<condition>/quality.jsonl   quality metrics per attack (optional)

Writes out-dir/tables/t1.tex ... t7.tex and out-dir/tables/tables.md (every
table also in markdown). T1 (attack taxonomy) and T7 (published-baseline
template) are static; all data-dependent tables degrade to an explicit
"no data" row when their inputs are missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paper constants (locked v1 matrix, research/01-experiment-protocol.md)
# ---------------------------------------------------------------------------

SCHEME_ORDER: tuple[str, ...] = (
    "kgw-d1",
    "kgw-d2",
    "kgw-d4",
    "synthid",
    "exp",
    "unigram",
    "sir",
)

SCHEME_MD: dict[str, str] = {
    "kgw-d1": "KGW (gamma=0.25, delta=1)",
    "kgw-d2": "KGW (gamma=0.5, delta=2)",
    "kgw-d4": "KGW (gamma=0.5, delta=4)",
    "synthid": "SynthID-Text",
    "exp": "EXP (Gumbel)",
    "unigram": "Unigram",
    "sir": "SIR",
}

SCHEME_TEX: dict[str, str] = {
    "kgw-d1": "KGW ($\\gamma{=}0.25$, $\\delta{=}1$)",
    "kgw-d2": "KGW ($\\gamma{=}0.5$, $\\delta{=}2$)",
    "kgw-d4": "KGW ($\\gamma{=}0.5$, $\\delta{=}4$)",
    "synthid": "SynthID-Text",
    "exp": "EXP (Gumbel)",
    "unigram": "Unigram",
    "sir": "SIR",
}

ATTACKS: tuple[str, ...] = (
    "none",
    "layerA",
    "paraphrase:1",
    "paraphrase:3",
    "backtranslate:de",
    "structural",
    "humanize",
    "cheap",
    "layerA+paraphrase:3",
)

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

# Statistical-rewrite attacks map to the "B-only" pipeline column (T2).
B_FAMILY: frozenset[str] = frozenset(
    {"paraphrase:1", "paraphrase:3", "backtranslate:de", "structural", "humanize", "cheap"}
)
LAYERED_ATTACK = "layerA+paraphrase:3"

# T1: (id, attack, family, mechanism, implementation, prior-art anchor).
# Hardcoded from research/01-experiment-protocol.md section 4.1 -- no data
# needed; this is the reproducibility/novelty anchor of the attack design.
TAXONOMY: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("A0", "None (control)", "---", "No attack (control)", "---", "---"),
    (
        "A1",
        "Layer A only",
        "Formatting cleanup",
        "Unicode/invisible-char, bidi, tag cleanup",
        "clean_text.py (deterministic)",
        "Formatting-layer marks (zero-width/steganography class)",
    ),
    (
        "A2",
        "Paraphrase, single pass",
        "Statistical rewrite",
        "Paraphrase, single pass",
        ("rewrite_text.py --strength paraphrase --candidates 1 --max-loops 1"),
        "Paraphrase attacks in the watermark literature",
    ),
    (
        "A3",
        "Paraphrase, adaptive",
        "Statistical rewrite (adaptive)",
        "Paraphrase with detection-feedback early stop (up to 3 loops)",
        (
            "rewrite_text.py --strength paraphrase --candidates 3 "
            "--max-loops 3 --markllm-scheme <scheme>"
        ),
        "Same; our eval-loop is the 'oracle' version",
    ),
    (
        "A4",
        "Back-translation round trip",
        "Statistical rewrite",
        "Translation round trip EN -> DE -> EN",
        "rewrite_text.py --strength backtranslate --lang German",
        "Can Watermarks Survive Translation? (X-SIR, ACL 2024)",
    ),
    (
        "A5",
        "Structural: outline -> regenerate",
        "Statistical rewrite",
        "Outline -> regenerate",
        "rewrite_text.py --strength structural",
        "Summarization/outline attacks",
    ),
    (
        "A6",
        "Humanize",
        "Statistical rewrite",
        "Style transfer to human-like prose",
        "rewrite_text.py --strength humanize",
        "Style-transfer attacks",
    ),
    (
        "A7",
        "Cheap baselines",
        "Heuristic / cheap",
        ("Synonym substitution, random word deletion (5-10%), sentence reorder"),
        "research/scripts/attacks/cheap.py (gap 05-A3)",
        "Random Walk / impossibility (ICML 2024)",
    ),
    (
        "A8",
        "Full pipeline (layered)",
        "Layered (A + B)",
        "Layer A cleanup then adaptive paraphrase (A1 -> A2/A3)",
        "clean_text.py then rewrite_text.py",
        "Our layered contribution (this work)",
    ),
)

# T7: (baseline, reported metric/setting, default citation key, our closest cell).
T7_DEFAULT_CITES: dict[str, str] = {
    "kgw_robust": "kirchenbauer2023reliability",
    "x_sir": "he2024canwatermarks",
    "sand": "zhang2024watermarks",
    "synthid": "han2025synthid",
    "markllm": "pan2024markllm",
}
T7_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "KGW robustness (paraphrase)",
        "KGW; paraphrase; AUROC / TPR@FPR",
        "kgw_robust",
        "kgw-d2, L300, en, paraphrase:3",
    ),
    (
        "X-SIR: can watermarks survive translation?",
        "translation round trip; cross-lingual detection",
        "x_sir",
        "kgw-d2, L300, de, backtranslate:de",
    ),
    (
        "Watermarks in the Sand (impossibility)",
        "random-walk removal; impossibility theory",
        "sand",
        "kgw-d4, L300, en, cheap + adaptive rewrite",
    ),
    (
        "SynthID-Text robustness assessment",
        "SynthID under editing; detection rate",
        "synthid",
        "synthid, L300, en, layerA+paraphrase:3",
    ),
    (
        "MarkLLM toolkit (harness)",
        "same-config detection; open-source toolkit",
        "markllm",
        "all cells (harness)",
    ),
)

# T3 quality columns: (header, candidate keys in quality.jsonl rows).
QUALITY_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PPL delta", ("ppl_delta", "ppl_delta_pct", "perplexity_delta")),
    ("BERTScore", ("bertscore", "bert_score")),
    ("ROUGE-L", ("rouge_l", "rougeL", "rouge-l", "rouge")),
    ("SBERT", ("sbert", "sbert_cosine", "sbert_sim", "sbert_score")),
    ("length drift", ("length_drift", "length_drift_pct", "length_change")),
    ("num survival", ("num_survival", "number_survival", "numbers_survival")),
    ("URL survival", ("url_survival", "link_survival")),
)

# Keys at the top level of metrics.json that are not attack names.
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

_TEX_HEADER = (
    "% Generated by research/scripts/make_tables.py (gap 05-B3) -- do not "
    "edit by hand.\n"
    "% Requires the booktabs package; include with \\input{tables/<name>.tex}.\n"
)


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

_TEX_CHARS: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "\u2013": "--",  # en dash
    "\u2014": "---",  # em dash
}


def _tex(text: str) -> str:
    """Escape *text* for use inside a LaTeX table cell or caption.

    Single-pass character mapping: each character is escaped at most once,
    so the braces introduced by \textbackslash{} are never re-escaped.
    """
    return "".join(_TEX_CHARS.get(ch, ch) for ch in text)


def _fmt3(v: float | None) -> str:
    return "---" if v is None else f"{v:.3f}"


def _fmt_int(v: float | None) -> str:
    return "---" if v is None else f"{v:,.0f}"


def _fmt_sec(v: float | None) -> str:
    return "---" if v is None else f"{v:.1f}"


def _fmt_usd(v: float | None) -> str:
    return "---" if v is None else f"{v:.4f}"


def _render_tex(
    title: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str = "",
    align: str | None = None,
) -> str:
    """Render a complete LaTeX table snippet from raw (unescaped) cells."""
    n = len(headers)
    align = align or ("l" + "c" * (n - 1))
    lines = [
        _TEX_HEADER,
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_tex(title)}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{@{{{align}}}@{{}}}}",
        "\\toprule",
        " & ".join(_tex(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_tex(str(c)) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\vspace{{2pt}}\\footnotesize{{{_tex(note)}}}")
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def _render_md(
    heading: str, headers: Sequence[str], rows: Sequence[Sequence[str]], note: str = ""
) -> str:
    """Render a markdown table block (dash placeholders become em dashes)."""
    parts = [f"### {heading}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        cells = [str(c).replace("---", "\u2014") for c in row]
        parts.append("| " + " | ".join(cells) + " |")
    if note:
        parts += ["", f"*Note: {note}*"]
    parts.append("")
    return "\n".join(parts)


def _no_data_tex(title: str, label: str, headers: Sequence[str]) -> str:
    n = len(headers)
    return _render_tex(
        title,
        label,
        headers,
        [["no data" if i == 0 else "" for i in range(n)]],
        note="Inputs not found in --results-dir; regenerate after the run.",
    )


def _no_data_md(heading: str, headers: Sequence[str]) -> str:
    return _render_md(
        heading, headers, [["no data" if i == 0 else "" for i in range(len(headers))]]
    )


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One results cell: DIR/<condition>/ with its parsed artifacts."""

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
    """Return {attack: metrics} from any supported metrics.json shape."""
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
    """Normalize one attack's ROC points to (fpr, tpr) lists."""
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
    """Parse one DIR/<condition>/ directory (missing files are tolerated)."""
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
    if not metrics and not quality and not attacked:
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
    """TPR at the given FPR, tolerating dict/number and key-name variants."""
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


def _conds_for_scheme(
    conditions: Sequence[Condition], scheme: str, prefer: str | None = None
) -> list[Condition]:
    conds = [c for c in conditions if c.scheme == scheme]
    if prefer:
        preferred = [c for c in conds if prefer in c.name]
        if preferred:
            return preferred
    return conds


# ---------------------------------------------------------------------------
# Table builders (each returns (latex, markdown))
# ---------------------------------------------------------------------------


def table_t1() -> tuple[str, str]:
    """T1: static attack taxonomy (family x mechanism x implementation x anchor)."""
    headers = ("ID", "Attack", "Family", "Mechanism", "Implementation", "Prior-art anchor")
    rows = [list(row) for row in TAXONOMY]
    tex = _render_tex(
        "Attack taxonomy (T1): family x mechanism x implementation x prior-art "
        "anchor. Source: research/01-experiment-protocol.md section 4.1.",
        "tab:t1",
        headers,
        rows,
        align="clp{4.2cm}p{5.0cm}p{5.2cm}p{4.2cm}",
        note="A0 = control; A1 = Layer A (formatting cleanup); A2-A7 = Layer B "
        "(statistical rewrite); A8 = layered pipeline (our contribution).",
    )
    md = _render_md(
        "T1 -- Attack taxonomy (family x mechanism x implementation x prior-art anchor)",
        headers,
        rows,
        note="Hardcoded from research/01-experiment-protocol.md section 4.1.",
    )
    return tex, md


def _t2_scheme_rows(conditions: Sequence[Condition], scheme: str, metric: str) -> list[list[str]]:
    conds = _conds_for_scheme(conditions, scheme, prefer="-L300-T0.7-en-")
    rows: list[list[str]] = []
    for attack in ATTACKS[1:]:
        pre = _mean_metric(conds, "none", metric)
        aonly = _mean_metric(conds, "layerA", metric)
        bonly = _mean_metric(conds, attack, metric) if attack in B_FAMILY else None
        ab = _mean_metric(conds, LAYERED_ATTACK, metric) if attack == LAYERED_ATTACK else None
        rows.append([ATTACK_LABELS[attack], _fmt3(pre), _fmt3(aonly), _fmt3(bonly), _fmt3(ab)])
    return rows


def table_t2(conditions: Sequence[Condition]) -> tuple[str, str]:
    """T2: money table -- AUROC / TPR@1%FPR matrix (scheme x attack x stage)."""
    headers = ("attack", "pre-attack", "A-only", "B-only", "A+B")
    schemes = [s for s in SCHEME_ORDER if any(c.scheme == s for c in conditions)]
    if not schemes:
        return _no_data_tex("AUROC / TPR@1%FPR matrix (T2)", "tab:t2", headers), _no_data_md(
            "T2 -- AUROC / TPR@1%FPR matrix", headers
        )

    note = (
        "Rows: scheme-config x attack. 'pre-attack' = A0 (no attack); "
        "'A-only' = Layer A cleanup (A1); 'B-only' = the row's statistical "
        "rewrite (A2-A7); 'A+B' = layered pipeline (A8). Cells: mean over "
        "reference cells (L300, T0.7, en); '---' = not applicable."
    )
    md_parts = [
        "### T2 -- AUROC / TPR@1%FPR matrix (scheme-config x attack x pipeline stage)",
    ]
    tex_blocks: list[str] = []
    for metric, short in (("auroc", "AUROC"), ("tpr", "TPR@1%FPR")):
        ab = "a" if metric == "auroc" else "b"
        tex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{short} by attack and pipeline stage (T2{ab}). {_tex(note)}}}",
            f"\\label{{tab:t2{ab}}}",
            "\\small",
            "\\begin{tabular}{@{}lccccc@{}}",
            "\\toprule",
            " & ".join(headers) + " \\\\",
            "\\midrule",
        ]
        md_lines: list[str] = []
        for scheme in schemes:
            tex_lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{SCHEME_TEX[scheme]}}}}} \\\\")
            md_lines += [
                f"**{SCHEME_MD[scheme]}**",
                "",
                "| " + " | ".join(headers) + " |",
                "|" + "---|" * len(headers),
            ]
            for row in _t2_scheme_rows(conditions, scheme, metric):
                tex_lines.append(" & ".join(_tex(c) for c in row) + " \\\\")
                md_lines.append("| " + " | ".join(c.replace("---", "\u2014") for c in row) + " |")
        tex_lines += [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\vspace{{2pt}}\\footnotesize{{{_tex(note)}}}",
            "\\end{table}",
            "",
        ]
        tex_blocks.append("\n".join(tex_lines))
        md_parts += md_lines
        md_parts.append("")
    return _TEX_HEADER + "\n".join(tex_blocks), "\n".join(md_parts)


def _quality_value(row: Mapping[str, Any], aliases: tuple[str, ...]) -> float | None:
    sub = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else None
    for src in (row, sub):
        if not src:
            continue
        for alias in aliases:
            v = src.get(alias)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def table_t3(conditions: Sequence[Condition]) -> tuple[str, str]:
    """T3: quality per attack at the detection-collapse operating point."""
    headers = ["attack"] + [col for col, _ in QUALITY_COLUMNS]
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        for row in cond.quality:
            if isinstance(row, dict) and row.get("attack"):
                rows.append(row)
    if not rows:
        return _no_data_tex(
            "Quality per attack at the collapse point (T3)", "tab:t3", headers
        ), _no_data_md("T3 -- Quality per attack at collapse", headers)
    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_attack[str(row["attack"])].append(row)
    out_rows: list[list[str]] = []
    for attack in ATTACKS:
        bucket = by_attack.get(attack)
        if not bucket:
            continue
        cells = [ATTACK_LABELS[attack]]
        for col, aliases in QUALITY_COLUMNS:
            values: list[float] = []
            for r in bucket:
                v = _quality_value(r, aliases)
                if v is not None:
                    values.append(v)
            if not values:
                cells.append("---")
            else:
                mean = sum(values) / len(values)
                cells.append(f"{mean:.2f}" if col == "PPL delta" else f"{mean:.3f}")
        out_rows.append(cells)
    tex = _render_tex(
        "Quality per attack at the detection-collapse operating point (T3). "
        "PPL delta = perplexity increase vs. original (gpt2-large); BERTScore = "
        "deberta-xlarge-mnli; SBERT = all-MiniLM-L6-v2; survival = fraction of "
        "numbers/URLs preserved.",
        "tab:t3",
        headers,
        out_rows,
        note="Quality at the point where detection collapses (01 section 5.3); "
        "'---' = metric not recorded for this attack.",
    )
    md = _render_md(
        "T3 -- Quality per attack at collapse (PPL delta, BERTScore, "
        "ROUGE-L, SBERT, length drift, num/URL survival)",
        headers,
        out_rows,
    )
    return tex, md


def table_t4(conditions: Sequence[Condition], attack: str) -> tuple[str, str]:
    """T4: ablation -- TPR@1%FPR x (gamma, delta) x length (KGW, en)."""
    kgw = [
        c for c in conditions if c.scheme in ("kgw-d1", "kgw-d2", "kgw-d4") and c.language == "en"
    ]
    col_specs = ((100, 0.7), (300, 0.7), (500, 0.7), (300, 1.0))
    headers = ["config (gamma, delta)"] + [f"L{length} / T{temp:g}" for length, temp in col_specs]
    if not kgw:
        return _no_data_tex(
            "Ablation: TPR@1%FPR x (gamma, delta) x length (T4)", "tab:t4", headers
        ), _no_data_md("T4 -- Ablation: TPR@1%FPR x (gamma, delta) x length", headers)
    rows: list[list[str]] = []
    for scheme in ("kgw-d1", "kgw-d2", "kgw-d4"):
        conds = [c for c in kgw if c.scheme == scheme]
        cells: list[str] = []
        for length, temp in col_specs:
            sub = [c for c in conds if c.length == length and c.temp == temp]
            cells.append(_fmt3(_mean_metric(sub, attack, "tpr")))
        rows.append([SCHEME_MD[scheme], *cells])
    tex = _render_tex(
        f"Ablation (T4): TPR@1%FPR x (gamma, delta) x length, EN cells. Attack: {attack}.",
        "tab:t4",
        headers,
        rows,
        note="Mean across seeds/prompts for the matching cells; '---' = cell "
        "absent from the locked v1 matrix (01 section 2).",
    )
    md = _render_md(
        f"T4 -- Ablation: TPR@1%FPR x (gamma, delta) x length (attack: {attack})", headers, rows
    )
    return tex, md


def table_t5(conditions: Sequence[Condition], attack: str) -> tuple[str, str]:
    """T5: multilingual fragility -- EN/DE/FR/ES at L300, T0.7."""
    langs = ("en", "de", "fr", "es")
    schemes = ("kgw-d2", "synthid")
    headers = ["language"]
    for scheme in schemes:
        headers += [f"{SCHEME_MD[scheme]} AUROC", f"{SCHEME_MD[scheme]} TPR@1%FPR"]
    rows: list[list[str]] = []
    any_data = False
    for lang in langs:
        cells = [lang]
        for scheme in schemes:
            conds = [
                c
                for c in conditions
                if c.scheme == scheme and c.language == lang and c.length == 300 and c.temp == 0.7
            ]
            auroc = _mean_metric(conds, attack, "auroc")
            tpr = _mean_metric(conds, attack, "tpr")
            if auroc is not None or tpr is not None:
                any_data = True
            cells += [_fmt3(auroc), _fmt3(tpr)]
        rows.append(cells)
    if not any_data:
        return _no_data_tex(
            "Multilingual: AUROC / TPR@1%FPR by language (T5)", "tab:t5", headers
        ), _no_data_md("T5 -- Multilingual (EN/DE/FR/ES)", headers)
    tex = _render_tex(
        f"Multilingual fragility (T5): AUROC / TPR@1%FPR by language at L300, "
        f"T0.7. Attack: {attack}. DE/FR/ES use the Qwen2.5-1.5B holdout.",
        "tab:t5",
        headers,
        rows,
        note="Restricted multilingual grid (01 section 2): kgw-d2 + synthid, "
        "L300, T0.7; '---' = cell absent.",
    )
    md = _render_md(f"T5 -- Multilingual EN/DE/FR/ES (L300, T0.7; attack: {attack})", headers, rows)
    return tex, md


def _first_num(*sources: Mapping[str, Any] | None, keys: Sequence[str]) -> float | None:
    for src in sources:
        if not src:
            continue
        for key in keys:
            v = src.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _word_count(text: Any) -> int | None:
    if isinstance(text, str) and text.strip():
        return len(text.split())
    return None


def _attacked_cost(
    row: Mapping[str, Any],
) -> tuple[str, float | None, float | None, float | None, float | None] | None:
    """Normalize one attacked.jsonl row to per-1k-words cost figures."""
    attack = str(row.get("attack", "")).strip()
    if not attack:
        return None
    stats = row.get("stats") if isinstance(row.get("stats"), Mapping) else {}
    words = _first_num(stats, keys=("words", "words_in", "original_words"))
    if words is None:
        words = _word_count(row.get("original"))
    if not words or words <= 0:
        return None
    tok_in = _first_num(stats, row, keys=("tokens_in", "input_tokens", "prompt_tokens"))
    tok_out = _first_num(stats, row, keys=("tokens_out", "output_tokens", "completion_tokens"))
    sec = _first_num(row, stats, keys=("seconds", "wall_time", "elapsed_seconds"))
    usd = _first_num(row, stats, keys=("usd", "cost_usd", "cost"))
    scale = 1000.0 / words
    scaled = [None if v is None else v * scale for v in (tok_in, tok_out, sec, usd)]
    return (attack, scaled[0], scaled[1], scaled[2], scaled[3])


def table_t6(conditions: Sequence[Condition]) -> tuple[str, str]:
    """T6: attack cost -- tokens in/out, wall time, USD per 1k words."""
    headers = (
        "attack",
        "tokens in / 1k words",
        "tokens out / 1k words",
        "seconds / 1k words",
        "USD / 1k words",
    )
    per_attack: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"tin": [], "tout": [], "sec": [], "usd": []}
    )
    for cond in conditions:
        for row in cond.attacked:
            parsed = _attacked_cost(row)
            if parsed is None:
                continue
            attack, tin, tout, sec, usd = parsed
            if attack == "none":
                continue
            bucket = per_attack[attack]
            for name, val in (("tin", tin), ("tout", tout), ("sec", sec), ("usd", usd)):
                if val is not None:
                    bucket[name].append(val)
    if not per_attack:
        return _no_data_tex(
            "Attack cost per 1k words of original text (T6)", "tab:t6", headers
        ), _no_data_md("T6 -- Attack cost per 1k words", headers)
    rows: list[list[str]] = []
    for attack in ATTACKS[1:]:
        bucket = per_attack.get(attack)
        if not bucket:
            continue
        mean = {k: (sum(v) / len(v) if v else None) for k, v in bucket.items()}
        rows.append(
            [
                ATTACK_LABELS[attack],
                _fmt_int(mean["tin"]),
                _fmt_int(mean["tout"]),
                _fmt_sec(mean["sec"]),
                _fmt_usd(mean["usd"]),
            ]
        )
    tex = _render_tex(
        "Attack cost (T6): tokens in/out, wall time, and USD per 1k words of "
        "the original document, from attacked.jsonl (mean across cells).",
        "tab:t6",
        headers,
        rows,
        note="Rows normalized per 1,000 words of the original text; '---' = field not recorded.",
    )
    md = _render_md(
        "T6 -- Attack cost per 1k words (tokens in/out, wall time, USD)",
        headers,
        rows,
        note="Mean across cells; normalized per 1k words of the original text.",
    )
    return tex, md


def _bib_keys(results_dir: Path) -> set[str]:
    """Citation keys from a refs.bib, searched in likely locations."""
    here = Path(__file__).resolve().parents[1]  # research/
    candidates = [
        results_dir / "refs.bib",
        results_dir.parent / "refs.bib",
        results_dir / "paper" / "refs.bib",
        here / "paper" / "refs.bib",
        here / "refs.bib",
    ]
    keys: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))
    return keys


def table_t7(results_dir: Path) -> tuple[str, str]:
    """T7: static comparison with published baselines (placeholders + cites)."""
    bib_keys = _bib_keys(results_dir)
    headers = (
        "Baseline",
        "Reported metric / setting",
        "Reported",
        "Our closest cell",
        "Ours",
        "Status",
    )
    tex_rows: list[list[str]] = []
    md_rows: list[list[str]] = []
    for name, setting, key_id, our_cell in T7_ROWS:
        key = T7_DEFAULT_CITES[key_id]
        placeholder = "TBD (re-verify at submission)"
        tex_rows.append(
            [f"{name} \\cite{{{key}}}", setting, placeholder, our_cell, placeholder, placeholder]
        )
        md_rows.append(
            [f"{name} [cite: {key}]", setting, placeholder, our_cell, placeholder, placeholder]
        )
    note = (
        "All placeholder cells are 'TBD (re-verify at submission)': fill "
        "from the original papers during the submission pass."
    )
    if bib_keys:
        note += f" Citation keys found in refs.bib: {', '.join(sorted(bib_keys))}."
    else:
        note += " No refs.bib found yet (gap C2); default keys listed."
    tex = _render_tex(
        "Comparison with published baselines (T7)", "tab:t7", headers, tex_rows, note=note
    )
    # _render_tex escapes backslashes/braces; restore the \cite commands.
    for key in T7_DEFAULT_CITES.values():
        escaped = r"\textbackslash{}cite\{" + key + r"\}"
        tex = tex.replace(escaped, f"\\cite{{{key}}}")
    md = _render_md("T7 -- Comparison with published baselines", headers, md_rows, note=note)
    return tex, md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="results layout root: DIR/<condition>/{metrics,quality,attacked}.json[l]",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True, help="output root; tables are written to OUT/tables/"
    )
    p.add_argument(
        "--attack", default=LAYERED_ATTACK, help="attack used by T4/T5 (default: %(default)s)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.results_dir.is_dir():
        print(
            f"warning: results dir not found ({args.results_dir}); tables will contain 'no data'",
            file=sys.stderr,
        )
    conditions = _load_conditions(args.results_dir)
    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables: list[tuple[str, tuple[str, str]]] = [
        ("t1", table_t1()),
        ("t2", table_t2(conditions)),
        ("t3", table_t3(conditions)),
        ("t4", table_t4(conditions, args.attack)),
        ("t5", table_t5(conditions, args.attack)),
        ("t6", table_t6(conditions)),
        ("t7", table_t7(args.results_dir)),
    ]

    md_parts = [
        "# Research tables (gap 05-B3)",
        "",
        "Auto-generated by research/scripts/make_tables.py -- do not edit by hand.",
        "",
    ]
    for name, (tex, md) in tables:
        (tables_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
        md_parts += [md, ""]
    (tables_dir / "tables.md").write_text("\n".join(md_parts), encoding="utf-8")
    print(f"wrote {len(tables)} tables to {tables_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
