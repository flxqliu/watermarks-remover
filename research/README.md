# research/ — Watermark Removal: Measurement Study & Paper Kit

Working area for turning the `watermarks-remover` project into a
publishable research paper. As of the arXiv-v1 implementation PR
(https://github.com/guillaumemeyer/watermarks-remover/pull/174), the
full code/analysis gap list (05-A1..A7, B1..B3, C1/C2) is implemented
here; the multi-week experiment run (05-D) and publishing logistics
(05-E) remain. Generated run data stays gitignored (`research/results/`).

## What is this paper about (one line)

*How robust are deployed-class LLM text watermarking schemes (KGW,
SynthID-Text, EXP, Unigram, SIR) to realistic user-side editing, and
what does a layered (formatting + statistical) removal pipeline do to
detection-rate and text quality?*

## File map

| File | Contents |
| --- | --- |
| `01-experiment-protocol.md` | Full experimental protocol: locked v1 factorial, factors, attacks, detection & quality metrics, statistics, budget, smoke test, risk register |
| `02-paper-outline.md` | Locked title, abstract draft, venue plan (arXiv v1 active), section-by-section skeleton, exact tables/figures spec, writing phases |
| `03-related-work.md` | Verified citation list (arXiv IDs checked), grouped by theme, with "where we position vs each" notes |
| `04-ethics-and-legal.md` | Dual-use framing, EU AI Act Art. 50 analysis, ethics statement draft, data policy, disclosure notes |
| `05-arxiv-readiness.md` | **Gap analysis for arXiv v1**: every missing code/analysis/paper/logistics item, budget, submission checklist — what must be built before publishing |
| `scripts/run_experiments.py` | Orchestrator: locked factorial design, `--plan` budget mode, and fully wired generate/attack/detect/evaluate/report stages with resume markers (gap 05-A4) |
| `scripts/pins.py` | Reproducibility pins: MarkLLM commit, repo commit, HF revisions, pip freeze (gap 05-A7) |
| `scripts/multilingual_gen.py` | CPU Qwen2.5-1.5B-Instruct generator for DE/FR/ES cells (model holdout, gap 05-A5) |
| `scripts/attacks/cheap.py` | Deterministic cheap baselines: synonym / word-deletion / sentence-reorder (gap 05-A3) |
| `scripts/analyze_roc.py` | AUROC, TPR@FPR (empirical null), bootstrap CIs (gap 05-B1) |
| `scripts/evaluate_quality.py` | PPL/BERTScore/ROUGE-L/SBERT/Levenshtein + survival metrics (gap 05-B2) |
| `scripts/make_tables.py`, `scripts/make_figures.py` | Paper tables T1-T7 and figures F1-F6 (gap 05-B3) |
| `configs/` | Pinned scheme configs incl. KGW-d1/d2/d4 strength JSONs (same JSON for gen and detect) |
| `corpus/` | 25 EN + 3x25 DE/FR/ES factual prompts (gap 05-A6) |
| `tests/` | pytest suite for the research harness (`make research-check`) |
| `paper/` | arXiv v1 LaTeX skeleton + verified bibliography (gaps C1/C2) |

## What already exists in the repo (reuse, don't rebuild)

| Capability | Where | Notes |
| --- | --- | --- |
| Watermarked corpus generation (SynthID only) | `service/scripts/bench_synthid_text.py` | MarkLLM `facebook/opt-1.3b`, 300 tok default, `--seeds`, `--docs`, `--max-new-tokens`; multi-scheme support is gap 05-A1 |
| Layer A (Unicode/invisible chars) | `service/scripts/text_unicode.py` via `clean_text.py` | deterministic |
| Layer B rewrites | `service/scripts/rewrite_text.py` | strengths: `paraphrase`, `humanize`, `backtranslate`, `structural`, `code`; transports: `ollama`, `openai-compatible`; evaluation-loop w/ early stop |
| Detection (same-config, KGW/SynthID) | `service/scripts/detect_text_watermark.py`, `text_detectors.py` | EXP/Unigram/SIR detection is gap 05-A2 |
| Existing metrics | `bench_synthid_text.py` | clear rate, score suppression, lexical divergence, length drift, number/URL survival, token/USD cost |
| Corpus seeds | `benchmarks/corpus/` (**8 files** — README previously said 9) | factual, neutral, 50-90 words; 25 EN + 75 multilingual prompts needed (gap 05-A6) |
| How-to doc | `docs/synthid-text-benchmark.md` | |

## Gap to close (this is the paper work)

Everything missing between today and an arXiv v1 is itemized in
**`05-arxiv-readiness.md`** (verified code gaps, analysis gaps, paper
gaps, publishing logistics, submission checklist, budget). Nothing in
that list is implemented yet — it is the hand-off to the implementation
phase.

## Status checklist

- [x] Decide framing — **measurement study** (not "attack tool" paper); title locked in 02 §1
- [x] Scope locked for arXiv v1 — Tier 1 core + multilingual (DE/FR/ES) + EXP/Unigram/SIR + length 500 + temp 1.0; attacks A0–A8; policy §7 in; human eval & file/metadata mini-study **deferred** (decision log below)
- [ ] Core experiment run (01 §2 matrix: 3,500 cells) — generate → attack → detect → evaluate
- [ ] Tables 1-7 + Figures 1-6 (02 §6)
- [ ] arXiv preprint (target: ~2-4 weeks after core results)
- [ ] Post-v1: workshop submission (NeurIPS 2026 workshops) → journal (TIFS) or ACL 2027

## Deferred (v2, not in arXiv v1)

- Human eval (20 × 3 annotators) — journal requirement.
- File/metadata mini-study (C2PA/EXIF strip; v0.1 01 §4.3).

## Decision log

| Date | Decision |
| --- | --- |
| 2026-08-18 | Scoped paper as *empirical robustness measurement* with layered-attack contribution; primary metrics ROC-based; venue ladder = arXiv → NeurIPS 2026 workshop → TIFS. Research dir created, gitignored, not committed. |
| 2026-08-18 | **v1 scope locked** (Q&A): measurement framing; title 2 in 02 §1; matrix = 7 schemes (KGW×3, SynthID, EXP, Unigram, SIR) × lengths 100/300/500 × temps 0.7/1.0 × langs en/de/fr/es with restricted subsets = **3,500 cells**; attacks A0–A8; policy §7 in; **deferred**: human eval, file/metadata mini-study; rewrite backend = OpenAI-compatible API; solo author; full run first (~2-4 weeks). Multi-scheme bench/detector support, ROC + quality metrics, cheap.py, corpus expansion, and paper artifacts are tracked as gaps in 05-arxiv-readiness.md (not yet implemented). |

## Git note

The arXiv-v1 implementation PR adds `!/research/` allow rules to
`.gitignore`, so the paper kit is tracked from that point on. Only
`research/results/` (generated JSONL data, released via Zenodo) stays
ignored.
