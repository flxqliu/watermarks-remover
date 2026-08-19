#!/usr/bin/env python3
"""Experiment orchestrator for the watermark-removal study.

Plans and drives the locked v1 factorial described in
research/01-experiment-protocol.md (3,500 cells, 7 schemes x lengths
100/300/500 x temps 0.7/1.0 x langs en/de/fr/es with the restricted
subsets encoded in cell_allowed()). It deliberately reuses the repo's
existing machinery (detect_text_watermark.py, rewrite_text.py,
clean_text.py, multilingual_gen.py, cheap.py) instead of reimplementing
anything, and shells out to it so the audit trail is the command history.

Modes:
  --plan       print cell counts + budget, exit (fully implemented)
  --dry-run    print the exact stage commands for a sample of cells
  --stage N    run a single stage (generate|attack|detect|evaluate|report)
  (no --stage) run every stage in order

Results layout (release-ready, condition-grouped; every row carries the
full cell identity scheme/length/temp/lang/seed/prompt):

  results/
    manifest.json          # pins, configs, seeds, timestamps (gap 05-A7)
    <condition>/generated.jsonl # watermarked + control texts per (seed, prompt)
    <condition>/attacked.jsonl  # per-attack outputs + cost fields
    <condition>/scores.jsonl    # detector scores per (doc, attack)
    <condition>/metrics.json    # AUROC, TPR@FPR, quality metrics
    <condition>/quality.jsonl   # quality metrics on a stratified subset
    report.md              # human-readable summary

Condition = (scheme, length, temp, language); there are 28 conditions
(14 EN core + 4 temp axis + 4 length axis + 6 multilingual). Resume is
per (condition, stage) via <stage>.done markers: a finished stage is
skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Design definition (mirrors 01-experiment-protocol.md, locked v1 matrix)
# --------------------------------------------------------------------------

SCHEMES = {
    # orchestrator name: (display, markllm alg, research/configs JSON)
    "kgw-d1": ("KGW", "KGW", "KGW-d1.json"),  # gamma=.25, delta=1
    "kgw-d2": ("KGW", "KGW", "KGW-d2.json"),  # gamma=.5, delta=2
    "kgw-d4": ("KGW", "KGW", "KGW-d4.json"),  # gamma=.5, delta=4
    "synthid": ("SynthID", "SynthID", "SynthID.json"),
    "exp": ("EXP", "EXP", "EXP.json"),
    "unigram": ("Unigram", "Unigram", "Unigram.json"),
    "sir": ("SIR", "SIR", "SIR.json"),
}

# orchestrator scheme -> detector CLI scheme (detect_text_watermark.SCHEMES key)
SCHEME_CLI = {
    "kgw-d1": "kgw",
    "kgw-d2": "kgw",
    "kgw-d4": "kgw",
    "synthid": "synthid",
    "exp": "exp",
    "unigram": "unigram",
    "sir": "sir",
}

# The 4 core schemes used by the temp / length / multilingual axes (01 §2).
CORE4 = ("kgw-d1", "kgw-d2", "kgw-d4", "synthid")
# The 2 schemes used by the multilingual grid (model holdout, 01 §3).
MULTILINGUAL2 = ("kgw-d2", "synthid")

ATTACKS = [
    "none",
    "layerA",  # clean_text.py (deterministic)
    "paraphrase:1",  # rewrite_text.py single pass
    "paraphrase:3",  # adaptive, early-stop on detection
    "backtranslate:de",  # --strength backtranslate --lang German
    "structural",
    "humanize",
    "cheap",  # expands to cheap:synonym / cheap:delete / cheap:reorder
    "layerA+paraphrase:3",  # full layered pipeline (our contribution)
]

# cheap.py sub-attacks (01 §4.1 A7).
CHEAP_SUBATTACKS = ("synonym", "delete", "reorder")

LENGTHS = [100, 300, 500]
TEMPS = [0.7, 1.0]
LANGUAGES = ["en", "de", "fr", "es"]
SEEDS = [1, 2, 3, 4, 5]
PROMPTS = 25

EN_MODEL = "facebook/opt-1.3b"
MULTILINGUAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MULTILINGUAL_MODEL_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_P = 0.95  # decoding protocol, 01 §3
REWRITE_TEMPERATURE = 0.9  # rewrite backend temperature, 01 §4.1
REWRITE_TIMEOUT = 300.0
MARKLLM_TIMEOUT = 900.0  # CPU generation can take minutes per text

MAX_CONFIG_BYTES = 1 << 20


@dataclass(frozen=True)
class Cell:
    scheme: str
    length: int
    temp: float
    language: str
    seed: int
    prompt_idx: int

    @property
    def id(self) -> str:
        return (
            f"{self.scheme}-L{self.length}-T{self.temp}"
            f"-{self.language}-s{self.seed}-p{self.prompt_idx}"
        )


@dataclass(frozen=True)
class Condition:
    scheme: str
    length: int
    temp: float
    language: str

    @property
    def id(self) -> str:
        return f"{self.scheme}-L{self.length}-T{self.temp}-{self.language}"


@dataclass
class Budget:
    cells: int
    generations: int
    attacks: int
    detections: int
    rewrite_tokens: int
    est_usd: float


def cell_allowed(scheme: str, length: int, temp: float, language: str) -> bool:
    """Locked v1 matrix restrictions (research/01-experiment-protocol.md §2)."""
    if language == "en" and temp == 0.7 and length in (100, 300):
        return True
    if language == "en" and temp == 1.0 and length == 300:
        return scheme in CORE4
    if language == "en" and length == 500 and temp == 0.7:
        return scheme in CORE4
    if language in ("de", "fr", "es") and length == 300 and temp == 0.7:
        return scheme in MULTILINGUAL2
    return False


def iter_cells(
    schemes: list[str],
    lengths: list[int],
    temps: list[float],
    languages: list[str],
    seeds: list[int],
    prompts: int,
) -> Iterator[Cell]:
    for scheme in schemes:
        for length in lengths:
            for temp in temps:
                for language in languages:
                    if not cell_allowed(scheme, length, temp, language):
                        continue
                    for seed in seeds:
                        for pidx in range(prompts):
                            yield Cell(scheme, length, temp, language, seed, pidx)


def iter_conditions(
    schemes: list[str],
    lengths: list[int],
    temps: list[float],
    languages: list[str],
) -> Iterator[Condition]:
    seen: set[tuple[str, int, float, str]] = set()
    for cell in iter_cells(schemes, lengths, temps, languages, [1], 1):
        key = (cell.scheme, cell.length, cell.temp, cell.language)
        if key not in seen:
            seen.add(key)
            yield Condition(*key)


def compute_budget(
    schemes: list[str],
    lengths: list[int],
    temps: list[float],
    languages: list[str],
    seeds: list[int],
    prompts: int,
    attacks: list[str],
) -> Budget:
    cells = sum(1 for _ in iter_cells(schemes, lengths, temps, languages, seeds, prompts))
    generations = 2 * cells
    n_attacks = len(attacks) - 1 + len(CHEAP_SUBATTACKS)  # 'none' + others; cheap x3
    detections = 2 * cells * (1 + n_attacks)
    tok_factor = {
        "paraphrase:1": 1.3,
        "paraphrase:3": 2.6,
        "backtranslate:de": 4.0,
        "structural": 2.0,
        "humanize": 1.3,
        "cheap": 0.1,
        "layerA": 0.0,
        "layerA+paraphrase:3": 2.6,
        "none": 0.0,
    }
    avg_in = 250
    rewrite_tokens = int(
        2
        * cells
        * avg_in
        * sum(tok_factor[a] * (len(CHEAP_SUBATTACKS) if a == "cheap" else 1) for a in attacks)
    )
    est_usd = rewrite_tokens * (0.5 + 1.5) / 1e6 * 1.4
    return Budget(cells, generations, n_attacks, detections, rewrite_tokens, est_usd)


# --------------------------------------------------------------------------
# Runtime helpers (subprocess shell-out + JSON-lines workers)
# --------------------------------------------------------------------------


class RunContext:
    """Everything a stage needs beyond its cell: scripts, workers, options."""

    def __init__(self, args: argparse.Namespace, upstream: Path) -> None:
        self.args = args
        self.upstream = upstream
        self.python = str(_venv_python(upstream) or sys.executable)
        self.scripts = Path(__file__).resolve().parents[1]  # research/
        self.service = self.scripts.parent / "service" / "scripts"
        self.corpus_dir = Path(args.corpus_dir).resolve()
        self.workers: dict[tuple, ServeWorker] = {}
        self._stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def config_path(self, scheme: str) -> Path:
        cfg = self.scripts / "configs" / SCHEMES[scheme][2]
        if not cfg.is_file():
            raise SystemExit(f"error: missing config {cfg} (run from the repo root)")
        return cfg

    def prompt_text(self, language: str, prompt_idx: int) -> str:
        p = self.corpus_dir / language / f"{prompt_idx:02d}.txt"
        if not p.is_file():
            raise SystemExit(f"error: missing prompt {p}; build the corpus first (gap 05-A6)")
        return p.read_text(encoding="utf-8", errors="surrogateescape").strip()

    def worker_for(
        self,
        *,
        kind: str,
        scheme: str,
        config: Path,
        model: str,
        language: str = "en",
        temperature: float | None = None,
    ) -> ServeWorker:
        """Persistent serve worker keyed by (kind, scheme, config, model, lang)."""
        key = (kind, scheme, str(config), model, language)
        w = self.workers.get(key)
        if w is not None:
            return w
        cli = SCHEME_CLI[scheme]
        if kind == "gen-multilingual":
            script = self.scripts / "scripts" / "multilingual_gen.py"
            cmd = [
                self.python,
                str(script),
                "serve",
                "--markllm-dir",
                str(self.upstream),
                "--scheme",
                cli,
                "--config",
                str(config),
                "--model",
                model,
                "--lang",
                language,
            ]
        else:  # detect_text_watermark serve (EN gen + all detection)
            script = self.service / "detect_text_watermark.py"
            cmd = [
                self.python,
                str(script),
                "serve",
                "--scheme",
                cli,
                "--config",
                str(config),
                "--model",
                model,
                "--upstream-dir",
                str(self.upstream),
                "--port",
                "0",
            ]
            if temperature is not None:
                cmd += ["--temperature", str(temperature), "--top-p", str(TOP_P)]
        w = ServeWorker(cmd, timeout=MARKLLM_TIMEOUT, label=key[0] + " " + key[1])
        self.workers[key] = w
        return w

    def close_workers(self) -> None:
        for w in self.workers.values():
            with _suppress(Exception):
                w.close()
        self.workers.clear()


class ServeWorker:
    """JSON-lines stdin/stdout worker (detect_text_watermark serve protocol).

    Speaks the ready-handshake + watermark/detect/exit protocol of
    detect_text_watermark.py serve (and multilingual_gen.py serve).
    """

    def __init__(self, cmd: list[str], *, timeout: float, label: str) -> None:
        self._timeout = timeout
        self._label = label
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_tail: list[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        ready = self._read_line(timeout)
        if ready is None or not ready.get("ready"):
            self.close()
            raise RuntimeError(
                f"{label} worker did not become ready"
                + (f": {ready.get('error')}" if ready else "")
                + (f"; stderr: {' | '.join(self._stderr_tail[-2:])}" if self._stderr_tail else "")
            )
        self.info = ready
        self.port = ready.get("port")

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 200:
                self._stderr_tail.pop(0)

    def _read_line(self, timeout: float) -> dict[str, Any] | None:
        q: queue.Queue[str] = queue.Queue()

        def _reader() -> None:
            try:
                line = self._proc.stdout.readline() if self._proc.stdout else ""
                q.put(line)
            except Exception as e:
                q.put(f"__error__:{e}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise RuntimeError(f"{self._label} worker response timed out")
        line = q.get()
        if line.startswith("__error__:"):
            raise RuntimeError(line[len("__error__:") :])
        if not line:
            raise RuntimeError(f"{self._label} worker closed (EOF)")
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            raise RuntimeError(f"{self._label} worker non-JSON: {line[:120]!r}") from None
        return data if isinstance(data, dict) else None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc.stdin is None:
            raise RuntimeError(f"{self._label} worker has no stdin")
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            resp = self._read_line(self._timeout)
        except Exception as e:
            hint = "; ".join(self._stderr_tail[-3:])
            raise RuntimeError(f"{self._label} worker failed: {e} ({hint})") from None
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") or f"{self._label} request failed")
        return resp

    def watermark(
        self,
        prompt: str,
        seed: int,
        max_new_tokens: int,
        *,
        temperature: float | None,
        top_p: float | None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "op": "watermark",
                "id": seed,
                "prompt": prompt,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    def detect(self, text: str) -> dict[str, Any]:
        return self.request({"op": "detect", "id": 0, "text": text})

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(json.dumps({"op": "exit"}) + "\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                with _suppress(Exception):
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            with _suppress(Exception):
                if stream is not None:
                    stream.close()


def _venv_python(upstream: Path) -> Path | None:
    if os.name == "nt":
        candidate = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = upstream / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _suppress(exc: type[BaseException]):
    return contextlib.suppress(exc)


def _run_cmd(cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _json_from_stderr(stderr: str) -> dict[str, Any] | None:
    idx = stderr.find("{")
    if idx < 0:
        return None
    try:
        data = json.loads(stderr[idx:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _tokens(text: str) -> int:
    return max(1, int(len(text) / 4.0))


def _numbers_preserved(original: str, candidate: str) -> float:
    a = set(re.findall(r"\d+", original))
    if not a:
        return 1.0
    return len(a & set(re.findall(r"\d+", candidate))) / len(a)


def _urls_preserved(original: str, candidate: str) -> float:
    a = set(re.findall(r"https?://\S+", original))
    if not a:
        return 1.0
    return len(a & set(re.findall(r"https?://\S+", candidate))) / len(a)


def _cell_dir(out_dir: Path, condition: Condition) -> Path:
    return out_dir / condition.id


def _done(ctx: RunContext, cell_dir: Path, stage: str) -> bool:
    return not ctx.args.force and (cell_dir / f"{stage}.done").exists()


def _mark_done(cell_dir: Path, stage: str) -> None:
    (cell_dir / f"{stage}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# Stage: generate
# --------------------------------------------------------------------------


def stage_generate(
    condition: Condition, out_dir: Path, ctx: RunContext, dry_run: bool = False
) -> list[str]:
    """Generate watermarked + control text for one condition (25 prompts x 5 seeds)."""
    cell_dir = _cell_dir(out_dir, condition)
    out_file = cell_dir / "generated.jsonl"
    if dry_run:
        cfg = ctx.config_path(condition.scheme)
        model = EN_MODEL if condition.language == "en" else MULTILINGUAL_MODEL
        cli = SCHEME_CLI[condition.scheme]
        if condition.language == "en":
            script = ctx.service / "detect_text_watermark.py"
            cmd = [
                ctx.python,
                str(script),
                "serve",
                "--scheme",
                cli,
                "--config",
                str(cfg),
                "--model",
                model,
                "--upstream-dir",
                str(ctx.upstream),
                "--temperature",
                str(condition.temp),
                "--top-p",
                str(TOP_P),
                "--port",
                "0",
            ]
        else:
            script = ctx.scripts / "scripts" / "multilingual_gen.py"
            cmd = [
                ctx.python,
                str(script),
                "serve",
                "--markllm-dir",
                str(ctx.upstream),
                "--scheme",
                cli,
                "--config",
                str(cfg),
                "--model",
                model,
                "--lang",
                condition.language,
            ]
        return [
            " ".join(shlex.quote(c) for c in cmd),
            f"# {PROMPTS * len(SEEDS)} watermark requests over stdin -> {out_file}",
        ]
    if _done(ctx, cell_dir, "generate"):
        print(f"[skip] generate {condition.id} (done)")
        return []
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = ctx.config_path(condition.scheme)
    model = EN_MODEL if condition.language == "en" else MULTILINGUAL_MODEL
    worker = ctx.worker_for(
        kind="gen-multilingual" if condition.language != "en" else "gen-en",
        scheme=condition.scheme,
        config=cfg,
        model=model,
        language=condition.language,
    )
    rows: list[dict[str, Any]] = []
    for pidx in range(1, PROMPTS + 1):
        prompt = ctx.prompt_text(condition.language, pidx)
        for seed in SEEDS:
            row: dict[str, Any] = {
                "condition": condition.id,
                "scheme": condition.scheme,
                "length": condition.length,
                "temp": condition.temp,
                "language": condition.language,
                "seed": seed,
                "prompt_idx": pidx,
                "prompt": prompt,
                "model": model,
                "config": str(cfg),
                "ok": False,
                "error": None,
            }
            try:
                resp = worker.watermark(
                    prompt,
                    seed,
                    condition.length,
                    temperature=condition.temp,
                    top_p=TOP_P,
                )
                wm = resp.get("watermarked") or ""
                plain = resp.get("unwatermarked") or ""
                if len(wm.strip()) < 50:
                    raise RuntimeError("watermarked sample too short")
                row.update(
                    {
                        "watermarked": wm,
                        "unwatermarked": plain,
                        "ok": True,
                    }
                )
            except Exception as e:
                row["error"] = str(e)[:300]
            rows.append(row)
            print(f"[generate] {condition.id} s{seed} p{pidx}: " + ("ok" if row["ok"] else "FAIL"))
    _write_jsonl(out_file, rows)
    _mark_done(cell_dir, "generate")
    return []


# --------------------------------------------------------------------------
# Stage: attack
# --------------------------------------------------------------------------


def _attack_rows(attack: str) -> list[str]:
    if attack == "cheap":
        return [f"cheap:{a}" for a in CHEAP_SUBATTACKS]
    return [attack]


def stage_attack(
    condition: Condition, attack: str, out_dir: Path, ctx: RunContext, dry_run: bool = False
) -> list[str]:
    """Apply one attack condition to every generated (seed, prompt) row."""
    cell_dir = _cell_dir(out_dir, condition)
    gen_file = cell_dir / "generated.jsonl"
    out_file = cell_dir / "attacked.jsonl"
    if dry_run:
        return [
            "clean_text.py / rewrite_text.py / cheap.py invocation "
            f"({condition.id}, {attack}) -> {out_file}"
        ]
    if not gen_file.is_file():
        print(f"[warn] attack {condition.id} {attack}: no generated.jsonl; run generate first")
        return []
    if _done(ctx, cell_dir, f"attack:{attack}"):
        print(f"[skip] attack {condition.id} {attack} (done)")
        return []
    rows = _read_jsonl(out_file)
    existing = {(r.get("seed"), r.get("prompt_idx"), r.get("attack")) for r in rows}
    done_any = False
    for gen in _read_jsonl(gen_file):
        if not gen.get("ok"):
            continue
        seed, pidx = gen["seed"], gen["prompt_idx"]
        original = gen["watermarked"]
        for sub in _attack_rows(attack):
            if (seed, pidx, sub) in existing:
                continue
            candidate, stats, err, seconds = _run_one_attack(condition, sub, original, seed, ctx)
            row: dict[str, Any] = {
                "condition": condition.id,
                "scheme": condition.scheme,
                "seed": seed,
                "prompt_idx": pidx,
                "attack": sub,
                "original": original,
                "candidate": candidate,
                "ok": err is None,
                "error": err,
                "seconds": seconds,
                "tokens_in": _tokens(original),
                "tokens_out": _tokens(candidate) if candidate else None,
            }
            if stats:
                row["stats"] = stats
            row["usd"] = _usd_estimate(row)
            rows.append(row)
            done_any = True
            print(
                f"[attack] {condition.id} {sub} s{seed} p{pidx}: "
                + ("ok" if err is None else f"FAIL {err[:120]}")
            )
    if rows:
        _write_jsonl(out_file, rows)
    if done_any:
        _mark_done(cell_dir, f"attack:{attack}")
    return []


def _run_one_attack(
    condition: Condition,
    attack: str,
    original: str,
    seed: int,
    ctx: RunContext,
) -> tuple[str | None, dict[str, Any] | None, str | None, float]:
    """Run one attack; returns (candidate, stats, error, seconds)."""
    import tempfile

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wm-attack-") as raw_td:
        td = Path(raw_td)
        in_path = td / "in.txt"
        in_path.write_text(original, encoding="utf-8")

        if attack == "layerA":
            out_path = td / "out.txt"
            cmd = [
                ctx.python,
                str(ctx.service / "clean_text.py"),
                str(in_path),
                "-o",
                str(out_path),
            ]
            proc = _run_cmd(cmd, timeout=REWRITE_TIMEOUT)
            if proc.returncode != 0:
                return (
                    None,
                    None,
                    (proc.stderr or proc.stdout or "").strip()[:300],
                    time.monotonic() - started,
                )
            return (
                out_path.read_text(encoding="utf-8", errors="surrogateescape"),
                None,
                None,
                time.monotonic() - started,
            )

        if attack.startswith("cheap:"):
            sub = attack.split(":", 1)[1]
            out_path = td / "out.txt"
            cmd = [
                ctx.python,
                str(ctx.scripts / "scripts" / "attacks" / "cheap.py"),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--attack",
                sub,
                "--seed",
                str(seed),
            ]
            proc = _run_cmd(cmd, timeout=REWRITE_TIMEOUT)
            if proc.returncode != 0:
                return (
                    None,
                    None,
                    (proc.stderr or proc.stdout or "").strip()[:300],
                    time.monotonic() - started,
                )
            return (
                out_path.read_text(encoding="utf-8", errors="surrogateescape"),
                None,
                None,
                time.monotonic() - started,
            )

        if attack.startswith("paraphrase:") or attack in (
            "backtranslate:de",
            "structural",
            "humanize",
        ):
            strength, _, cand_s = attack.partition(":")
            candidates = int(cand_s) if cand_s else 1
            out_path = td / "out.txt"
            lang = "German" if attack == "backtranslate:de" else "French"
            cmd = [
                ctx.python,
                str(ctx.service / "rewrite_text.py"),
                str(in_path),
                "-o",
                str(out_path),
                "--backend",
                ctx.args.rewrite_backend,
                "--model",
                ctx.args.rewrite_model or "",
                "--base-url",
                ctx.args.rewrite_base_url,
                "--strength",
                strength,
                "--lang",
                lang,
                "--candidates",
                str(candidates),
                "--max-loops",
                str(candidates),
                "--temperature",
                str(REWRITE_TEMPERATURE),
                "--timeout",
                str(REWRITE_TIMEOUT),
                "--markllm-scheme",
                SCHEME_CLI[condition.scheme],
                "--markllm-dir",
                str(ctx.upstream),
                "--markllm-model",
                EN_MODEL if condition.language == "en" else MULTILINGUAL_MODEL,
                "--json-stats",
            ]
            env = dict(os.environ)
            if ctx.args.rewrite_api_key:
                env["WATERMARKS_REWRITE_API_KEY"] = ctx.args.rewrite_api_key
            proc = _run_cmd(
                cmd,
                timeout=REWRITE_TIMEOUT + MARKLLM_TIMEOUT + 60,
            )
            if proc.returncode != 0:
                return (
                    None,
                    None,
                    (proc.stderr or proc.stdout or "").strip()[:300],
                    time.monotonic() - started,
                )
            stats = _json_from_stderr(proc.stderr)
            out_text = out_path.read_text(encoding="utf-8", errors="surrogateescape")
            return out_text, stats, None, time.monotonic() - started

        if attack == "layerA+paraphrase:3":
            cleaned = td / "cleaned.txt"
            cmd = [
                ctx.python,
                str(ctx.service / "clean_text.py"),
                str(in_path),
                "-o",
                str(cleaned),
            ]
            proc = _run_cmd(cmd, timeout=REWRITE_TIMEOUT)
            if proc.returncode != 0:
                return None, None, (proc.stderr or "").strip()[:300], time.monotonic() - started
            out_path = td / "out.txt"
            cmd = [
                ctx.python,
                str(ctx.service / "rewrite_text.py"),
                str(cleaned),
                "-o",
                str(out_path),
                "--backend",
                ctx.args.rewrite_backend,
                "--model",
                ctx.args.rewrite_model or "",
                "--base-url",
                ctx.args.rewrite_base_url,
                "--strength",
                "paraphrase",
                "--candidates",
                "3",
                "--max-loops",
                "3",
                "--temperature",
                str(REWRITE_TEMPERATURE),
                "--timeout",
                str(REWRITE_TIMEOUT),
                "--markllm-scheme",
                SCHEME_CLI[condition.scheme],
                "--markllm-dir",
                str(ctx.upstream),
                "--markllm-model",
                EN_MODEL if condition.language == "en" else MULTILINGUAL_MODEL,
                "--json-stats",
            ]
            proc = _run_cmd(cmd, timeout=REWRITE_TIMEOUT + MARKLLM_TIMEOUT + 60)
            if proc.returncode != 0:
                return None, None, (proc.stderr or "").strip()[:300], time.monotonic() - started
            stats = _json_from_stderr(proc.stderr)
            out_text = out_path.read_text(encoding="utf-8", errors="surrogateescape")
            return out_text, stats, None, time.monotonic() - started

        return None, None, f"unknown attack {attack!r}", time.monotonic() - started


def _usd_estimate(row: dict[str, Any]) -> float:
    stats = row.get("stats") or {}
    in_tok = stats.get("tokens_in") or row.get("tokens_in") or 0
    out_tok = stats.get("tokens_out") or row.get("tokens_out") or 0
    return round(float(in_tok) / 1e6 * 0.5 + float(out_tok) / 1e6 * 1.5, 8)


# --------------------------------------------------------------------------
# Stage: detect
# --------------------------------------------------------------------------


def stage_detect(
    condition: Condition, attack: str, out_dir: Path, ctx: RunContext, dry_run: bool = False
) -> list[str]:
    """Same-config detection on originals + controls + attacked text."""
    cell_dir = _cell_dir(out_dir, condition)
    gen_file = cell_dir / "generated.jsonl"
    attack_file = cell_dir / "attacked.jsonl"
    out_file = cell_dir / "scores.jsonl"
    if dry_run:
        return [
            f"detect_text_watermark.py serve ({condition.id}) -> {out_file}",
        ]
    if not gen_file.is_file():
        print(f"[warn] detect {condition.id}: no generated.jsonl")
        return []
    if _done(ctx, cell_dir, f"detect:{attack}"):
        print(f"[skip] detect {condition.id} {attack} (done)")
        return []
    cfg = ctx.config_path(condition.scheme)
    model = EN_MODEL if condition.language == "en" else MULTILINGUAL_MODEL
    # Reuse the generation worker: the serve protocol supports detect, so the
    # resident model (opt-1.3b for EN, Qwen for multilingual) serves both.
    worker = ctx.worker_for(
        kind="gen-multilingual" if condition.language != "en" else "gen-en",
        scheme=condition.scheme,
        config=cfg,
        model=model,
        language=condition.language,
    )
    rows = _read_jsonl(out_file)
    existing = {(r.get("seed"), r.get("prompt_idx"), r.get("attack"), r.get("kind")) for r in rows}

    def add(seed: int, pidx: int, kind: str, text: str, attack: str) -> None:
        if (seed, pidx, attack, kind) in existing:
            return
        try:
            resp = worker.detect(text)
            row = {
                "condition": condition.id,
                "scheme": condition.scheme,
                "seed": seed,
                "prompt_idx": pidx,
                "attack": attack,
                "kind": kind,
                "score": resp.get("score"),
                "is_watermarked": resp.get("is_watermarked"),
                "threshold": resp.get("threshold"),
                "ok": True,
            }
        except Exception as e:
            row = {
                "condition": condition.id,
                "scheme": condition.scheme,
                "seed": seed,
                "prompt_idx": pidx,
                "attack": attack,
                "kind": kind,
                "ok": False,
                "error": str(e)[:300],
            }
        rows.append(row)

    for gen in _read_jsonl(gen_file):
        if not gen.get("ok"):
            continue
        seed, pidx = gen["seed"], gen["prompt_idx"]
        if attack == "none":
            add(seed, pidx, "watermarked", gen["watermarked"], "none")
            add(seed, pidx, "control", gen["unwatermarked"], "none")
        else:
            for arow in _read_jsonl(attack_file):
                if (
                    arow.get("seed") == seed
                    and arow.get("prompt_idx") == pidx
                    and arow.get("attack") == attack
                    and arow.get("ok")
                ):
                    add(seed, pidx, "attacked", arow["candidate"], attack)
    if rows:
        _write_jsonl(out_file, rows)
        _mark_done(cell_dir, f"detect:{attack}")
    return []


# --------------------------------------------------------------------------
# Stage: evaluate
# --------------------------------------------------------------------------


def stage_evaluate(
    condition: Condition, out_dir: Path, ctx: RunContext, dry_run: bool = False
) -> list[str]:
    """ROC metrics (B1) + quality metrics (B2) for one condition."""
    cell_dir = _cell_dir(out_dir, condition)
    scores_file = cell_dir / "scores.jsonl"
    attack_file = cell_dir / "attacked.jsonl"
    metrics_file = cell_dir / "metrics.json"
    quality_file = cell_dir / "quality.jsonl"
    if dry_run:
        return [
            f"analyze_roc.py --scores {scores_file} --out {metrics_file}",
            f"evaluate_quality.py --input <pairs> --out {quality_file}",
        ]
    if not scores_file.is_file():
        print(f"[warn] evaluate {condition.id}: no scores.jsonl")
        return []
    if _done(ctx, cell_dir, "evaluate"):
        print(f"[skip] evaluate {condition.id} (done)")
        return []
    roc = ctx.scripts / "scripts" / "analyze_roc.py"
    proc = _run_cmd(
        [
            ctx.python,
            str(roc),
            "--scores",
            str(scores_file),
            "--out",
            str(metrics_file),
            "--n-bootstrap",
            str(ctx.args.n_bootstrap),
            "--seed",
            "1",
        ],
        timeout=600,
    )
    if proc.returncode != 0:
        print(f"[warn] analyze_roc failed for {condition.id}: {(proc.stderr or '')[:300]}")
    else:
        _mark_done(cell_dir, "evaluate")

    # Quality on a stratified subset of attacked pairs (01 §5.3).
    pairs = _quality_pairs(condition, attack_file, ctx.args.quality_per_condition)
    if pairs:
        pairs_file = cell_dir / "pairs.jsonl"
        _write_jsonl(pairs_file, pairs)
        qual = ctx.scripts / "scripts" / "evaluate_quality.py"
        proc = _run_cmd(
            [
                ctx.python,
                str(qual),
                "--input",
                str(pairs_file),
                "--out",
                str(quality_file),
                "--device",
                "cpu",
                "--warnings-out",
                str(cell_dir / "quality-warnings.json"),
            ],
            timeout=3600,
        )
        if proc.returncode != 0:
            print(f"[warn] evaluate_quality failed for {condition.id}: {(proc.stderr or '')[:300]}")
    return []


def _quality_pairs(condition: Condition, attack_file: Path, cap: int) -> list[dict[str, Any]]:
    """Stratified subset: up to *cap* (seed, prompt, attack) rows per condition."""
    pairs: list[dict[str, Any]] = []
    for row in _read_jsonl(attack_file):
        if not row.get("ok"):
            continue
        if row.get("attack") in ("none",):
            continue
        pairs.append(
            {
                "condition": condition.id,
                "scheme": condition.scheme,
                "seed": row["seed"],
                "prompt_idx": row["prompt_idx"],
                "attack": row["attack"],
                "original": row["original"],
                "candidate": row["candidate"],
            }
        )
        if len(pairs) >= cap:
            break
    return pairs


# --------------------------------------------------------------------------
# Stage: report
# --------------------------------------------------------------------------


def stage_report(out_dir: Path, ctx: RunContext) -> list[str]:
    """Aggregate everything into manifest.json + report.md."""
    from pins import env_summary

    manifest_path = out_dir / "manifest.json"
    model_ids = sorted({EN_MODEL, MULTILINGUAL_MODEL})
    manifest: dict[str, Any] = {
        "study": "How Fragile Are Deployed Text Watermarks? (arXiv v1)",
        "created": ctx._stamp,
        "pins": env_summary(ctx.upstream, model_ids),
        "design": {
            "schemes": {
                k: {"display": v[0], "algorithm": v[1], "config": v[2]} for k, v in SCHEMES.items()
            },
            "attacks": ATTACKS,
            "lengths": LENGTHS,
            "temps": TEMPS,
            "languages": LANGUAGES,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "top_p": TOP_P,
            "rewrite_temperature": REWRITE_TEMPERATURE,
            "rewrite_backend": ctx.args.rewrite_backend,
            "rewrite_model": ctx.args.rewrite_model,
        },
        "command": " ".join(shlex.quote(a) for a in sys.argv),
    }
    conditions: list[dict[str, Any]] = []
    total: dict[str, int] = {}
    for cond_dir in sorted(out_dir.glob("*-L*-T*-*")):
        if not cond_dir.is_dir():
            continue
        gen = _read_jsonl(cond_dir / "generated.jsonl")
        att = _read_jsonl(cond_dir / "attacked.jsonl")
        sc = _read_jsonl(cond_dir / "scores.jsonl")
        metrics: dict[str, Any] = {}
        if (cond_dir / "metrics.json").is_file():
            try:
                metrics = json.loads((cond_dir / "metrics.json").read_text("utf-8"))
            except ValueError:
                metrics = {"error": "unparseable metrics.json"}
        ok_gen = sum(1 for r in gen if r.get("ok"))
        ok_att = sum(1 for r in att if r.get("ok"))
        ok_sc = sum(1 for r in sc if r.get("ok"))
        total["generated"] = total.get("generated", 0) + ok_gen
        total["attacked"] = total.get("attacked", 0) + ok_att
        total["scores"] = total.get("scores", 0) + ok_sc
        conditions.append(
            {
                "condition": cond_dir.name,
                "generated": ok_gen,
                "attacked": ok_att,
                "scores": ok_sc,
                "metrics": metrics.get("per_attack", {}) if isinstance(metrics, dict) else {},
                "done": {s: (cond_dir / f"{s}.done").exists() for s in ("generate", "evaluate")},
            }
        )
    manifest["conditions"] = conditions
    manifest["totals"] = total
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    L: list[str] = []
    L.append("# watermarks-remover study — run summary")
    L.append("")
    L.append(f"- created: {ctx._stamp}")
    L.append(f"- repo commit: {manifest['pins'].get('repo_commit') or 'unknown'}")
    L.append(f"- MarkLLM commit: {manifest['pins'].get('markllm_commit') or 'unknown'}")
    L.append(
        f"- generated ok: {total.get('generated', 0)}, attacked ok: {total.get('attacked', 0)}, scores ok: {total.get('scores', 0)}"
    )
    L.append("")
    L.append("| condition | gen | att | scores | AUROC (none) | AUROC (layerA+paraphrase:3) |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    def _auroc(pa: dict[str, Any], attack: str) -> str:
        m = pa.get(attack) or {}
        v = m.get("auroc")
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

    for c in conditions:
        pa = c["metrics"]
        L.append(
            f"| {c['condition']} | {c['generated']} | {c['attacked']} | {c['scores']} "
            f"| {_auroc(pa, 'none')} | {_auroc(pa, 'layerA+paraphrase:3')} |"
        )
    L.append("")
    L.append("Full per-condition data: JSONL per condition dir; manifest: manifest.json.")
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--schemes",
        default=",".join(SCHEMES),
        help="comma list; see SCHEMES (default: locked v1 set, 7 schemes)",
    )
    p.add_argument(
        "--lengths",
        default=",".join(str(x) for x in LENGTHS),
        help="comma list of token lengths (default: locked v1)",
    )
    p.add_argument(
        "--temps",
        default=",".join(str(x) for x in TEMPS),
        help="comma list of temperatures (default: locked v1)",
    )
    p.add_argument(
        "--languages",
        default=",".join(LANGUAGES),
        help="comma list (default: locked v1: en,de,fr,es)",
    )
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--prompts", type=int, default=PROMPTS)
    p.add_argument("--attacks", default=",".join(ATTACKS))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--markllm-dir", default=None, help="MarkLLM checkout (default: env MARKLLM_DIR)"
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("research/corpus"),
        help="Prompt corpus dir with <lang>/01.txt..25.txt (default: research/corpus)",
    )
    p.add_argument(
        "--rewrite-backend", choices=("ollama", "openai-compatible"), default="openai-compatible"
    )
    p.add_argument(
        "--rewrite-model", default=None, help="Rewrite backend model (required for attacks)"
    )
    p.add_argument("--rewrite-base-url", default="http://127.0.0.1:11434")
    p.add_argument("--rewrite-api-key", default=None, help="Rewrite API key (env-only in children)")
    p.add_argument(
        "--rewrite-allow-remote", action="store_true", help="Allow non-loopback rewrite endpoints"
    )
    p.add_argument("--n-bootstrap", type=int, default=10000, help="ROC bootstrap resamples (B1)")
    p.add_argument(
        "--quality-per-condition",
        type=int,
        default=8,
        help="Quality pairs per condition (01 §5.3 stratified subset)",
    )
    p.add_argument("--plan", action="store_true", help="print budget and exit")
    p.add_argument("--dry-run", action="store_true", help="print stage commands without running")
    p.add_argument(
        "--stage",
        choices=["generate", "attack", "detect", "evaluate", "report"],
        default=None,
        help="run a single stage",
    )
    p.add_argument("--force", action="store_true", help="re-run stages even if marked done")
    return p


def main() -> int:
    args = build_parser().parse_args()
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]

    for s in schemes:
        if s not in SCHEMES:
            print(f"error: unknown scheme {s!r}; known: {sorted(SCHEMES)}", file=sys.stderr)
            return 2

    budget = compute_budget(schemes, lengths, temps, languages, seeds, args.prompts, attacks)

    if args.plan:
        print("=== Plan (locked v1 matrix, 01-experiment-protocol.md §2) ===")
        print(f"schemes      : {schemes}")
        print(f"lengths      : {lengths}")
        print(f"temps        : {temps}")
        print(f"languages    : {languages}")
        print(f"seeds        : {seeds}")
        print(f"prompts      : {args.prompts}")
        print(f"attack cells : {attacks}")
        print(
            "restrictions : cell_allowed() - temp-1.0/length-500 on CORE4, multilingual on "
            + str(list(MULTILINGUAL2))
        )
        print("---")
        print(f"cells        : {budget.cells}")
        print(f"generations  : {budget.generations} (watermarked + control)")
        print(
            f"attack runs  : {budget.cells} docs x {budget.attacks} attacks = {budget.cells * budget.attacks}"
        )
        print(f"detections   : {budget.detections} (incl. controls + empirical null)")
        print(f"rewrite tokens (est): {budget.rewrite_tokens:,} (wm + control attacked)")
        print(f"rewrite cost (est)  : {budget.est_usd:.2f}")
        print("---")
        print("CPU estimate (8 cores): gen ~35-50h, detect ~25-90h, quality ~10-20h;")
        print("API rewrite budget ~$50-120 (record model + version per 01 §6)")
        print(
            "Conditions : "
            + str(sum(1 for _ in iter_conditions(schemes, lengths, temps, languages)))
        )
        return 0

    if not args.markllm_dir:
        args.markllm_dir = os.environ.get("MARKLLM_DIR")
    if not args.markllm_dir:
        print("error: --markllm-dir (or MARKLLM_DIR) is required", file=sys.stderr)
        return 2
    upstream = Path(args.markllm_dir).expanduser().resolve()
    if not (upstream / "watermark").is_dir():
        print(
            f"error: MarkLLM checkout incomplete (no watermark/ dir): {upstream}", file=sys.stderr
        )
        return 2

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(args, upstream)

    conditions = list(iter_conditions(schemes, lengths, temps, languages))
    if args.dry_run:
        print(f"# {len(conditions)} conditions; sample commands:")
        for cond in conditions[:2]:
            for line in stage_generate(cond, out_dir, ctx, dry_run=True):
                print(line)
        for cond in conditions[:1]:
            for a in attacks[:2]:
                for line in stage_attack(cond, a, out_dir, ctx, dry_run=True):
                    print(line)
            for line in stage_detect(cond, "none", out_dir, ctx, dry_run=True):
                print(line)
            for line in stage_evaluate(cond, out_dir, ctx, dry_run=True):
                print(line)
        return 0

    try:
        if args.stage == "generate":
            for cond in conditions:
                stage_generate(cond, out_dir, ctx)
        elif args.stage == "attack":
            for cond in conditions:
                for a in attacks:
                    stage_attack(cond, a, out_dir, ctx)
        elif args.stage == "detect":
            for cond in conditions:
                for a in attacks:
                    stage_detect(cond, a, out_dir, ctx)
        elif args.stage == "evaluate":
            for cond in conditions:
                stage_evaluate(cond, out_dir, ctx)
        elif args.stage == "report":
            stage_report(out_dir, ctx)
        else:
            print(f"=== generate ({len(conditions)} conditions) ===")
            for cond in conditions:
                stage_generate(cond, out_dir, ctx)
            print("=== attack ===")
            for cond in conditions:
                for a in attacks:
                    stage_attack(cond, a, out_dir, ctx)
            print("=== detect ===")
            for cond in conditions:
                for a in attacks:
                    stage_detect(cond, a, out_dir, ctx)
            print("=== evaluate ===")
            for cond in conditions:
                stage_evaluate(cond, out_dir, ctx)
            print("=== report ===")
            stage_report(out_dir, ctx)
    finally:
        ctx.close_workers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
