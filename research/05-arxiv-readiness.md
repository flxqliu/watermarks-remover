# 05 — arXiv v1 Readiness: Gap Analysis & Submission Checklist

Version: v0.1 (2026-08-18). Scope: arXiv v1 of *"How Fragile Are
Deployed Text Watermarks? An Empirical Study of Layered Watermark
Removal under Realistic User-Side Editing"* (locked scope in
research/README.md decision log).

This file itemizes **everything missing** between the current repo and
a submittable arXiv v1. It is a hand-off to the implementation phase —
**as of PR #174 all code/analysis gaps (A1-A7, B1-B3) and the paper
skeleton/bibliography (C1/C2) are implemented**; the experiment run
(D) and publishing logistics (E) remain. Items are marked with the
stage of the pipeline they block (generate → attack → detect →
evaluate → paper → publish).

---

## A. Verified code gaps (experiment harness)

Each row: what exists today / what's missing / effort / where it lands.
(All "exists" claims verified 2026-08-18.)

| # | Gap | What exists today | What's missing | Effort | Blocks |
| --- | --- | --- | --- | --- | --- |
| A1 | Multi-scheme **generation** | `service/scripts/bench_synthid_text.py` hardcodes `SCHEME = "synthid"` (line 63); MarkLLM checkout already ships `config/{KGW,SynthID,EXP,Unigram,SIR}.json` | `--scheme kgw|synthid|exp|unigram|sir` + `--config` override on the bench; 3 KGW strength JSONs (`research/configs/KGW-d1/d2/d4.json`) mirroring the checkout's `KGW.json` (γ=.25 δ=1; γ=.5 δ=2; γ=.5 δ=4) | small refactor (per 01 §10) | generate |
| A2 | Multi-scheme **detection** | `service/scripts/detect_text_watermark.py` scheme map (lines 41–45) covers `kgw`, `synthid` only | Add `exp`, `unigram`, `sir` entries + config wiring; same-config detection guarantee (01 §3: identical JSON for gen and detect) | small | detect |
| A3 | A7 cheap baselines | nothing | `research/scripts/attacks/cheap.py`: synonym substitution, 5–10% random word deletion, sentence reorder; deterministic + seeded | small | attack |
| A4 | Orchestrator stage wiring | `research/scripts/run_experiments.py` stages `generate/attack/detect/evaluate/report` all `raise NotImplementedError` (design/constants already aligned to the locked matrix) | Wire each stage to the repo scripts; emit documented results layout (`results/manifest.json`, per-cell `generated/attacked/scores.jsonl`, `metrics.json`, `report.md`); resume-able checkpoints | medium | all |
| A5 | Multilingual generator | opt-1.3b is English-only; **no GPU available** (verified 2026-08-18) | CPU path for `Qwen/Qwen2.5-1.5B-Instruct` (fallback 0.5B) via MarkLLM `TransformersConfig`; 5-prompt × 1-seed pilot quality check per language (01 §9) before the 750-cell grid | medium | generate (de/fr/es) |
| A6 | Corpus | `benchmarks/corpus/` has **8 files** (research/README previously said 9); factual, neutral, 50–90 words | 17 new EN prompts + 3×25 translated DE/FR/ES sets → `research/corpus/` (self-written, checkable claims) | medium | generate |
| A7 | Config pinning & manifest | `requirements-markllm.txt` pinned; MarkLLM checkout at `/home/guillaume/MarkLLM` | Record MarkLLM commit (`git -C /home/guillaume/MarkLLM rev-parse HEAD`), HF revisions (opt-1.3b, Qwen2.5-1.5B, gpt2-large, deberta-xlarge-mnli, all-MiniLM-L6-v2), `pip freeze` for MarkLLM + quality envs, `manifest.json` writer | small | reproducibility |

## B. Analysis gaps (nothing exists — verified by grep, 2026-08-18)

| # | Gap | Spec (from 01 §5) | Effort |
| --- | --- | --- | --- |
| B1 | ROC module | `research/scripts/analyze_roc.py`: AUROC, TPR@FPR ∈ {0.1%, 1%, 10%}, full ROC data per cell, **empirical null** from unwatermarked controls (never assume normal z-scores), 95% bootstrap CIs (10k resamples) | medium |
| B2 | Quality metrics | `research/scripts/evaluate_quality.py`: PPL (`gpt2-large`, never the generator), BERTScore (`deberta-xlarge-mnli`), ROUGE-L, SBERT cosine (`all-MiniLM-L6-v2`), Levenshtein %, length drift, number/URL survival; `research/requirements-quality.txt` (bert-score, rouge-score, sentence-transformers); stratified subset ~2–4k texts incl. every collapse point | medium |
| B3 | Table/figure generators | T1–T7 + F1–F6 per 02 §6: F2 ROC curves, F3 Pareto (PPL vs AUROC), F4 strength×length, F1 pipeline TikZ, F5 case study, F6 policy timeline | medium |

## C. Paper gaps

| # | Gap | Notes |
| --- | --- | --- |
| C1 | LaTeX skeleton | `research/paper/`, ACL-style template (fits venue ladder); abstract draft (02 §2), ethics draft (04 §3), acknowledgments (MarkLLM/THU-BPM) exist as markdown |
| C2 | Bibliography | `.bib` from 03 A–E; **§F candidates dropped** (see 03 header); re-verify every ID/venue at submission |
| C3 | Tables 1–7, Figures 1–6 | Specs in 02 §6; **require real numbers from the run** — no placeholders |
| C4 | Full prose §1–§10 | Skeleton in 02 §5; writing phase W2 |

## D. The experiment run itself (the largest single item)

- 3,500-cell locked matrix (01 §2): 7,000 generations, 56,000 attack
  outputs, ~65,000 detections + ~2,000 null controls.
- Budget: ~$50–120 API rewrites (model + version recorded), ~2.5–4
  weeks wall on one 8-core CPU box (01 §6).
- Order: smoke test (01 §9) → EN core → temp/length axes → multilingual
  pilot → multilingual grid → analysis (B1–B3) → tables/figures (C3).

## E. Publishing logistics (user-owned)

| # | Item | Who | Notes |
| --- | --- | --- | --- |
| E1 | arXiv account + endorsement | user | Solo, no affiliation (allowed); new authors in cs.CL typically need endorsement from an existing author — **start in week 1**, it can take days |
| E2 | Categories | user | cs.CL primary; cs.CR + cs.LG secondary |
| E3 | Release package | agent+user | Repo URL (github.com/guillaumemeyer/watermarks-remover), MIT license, corpus/configs/results JSONL; Zenodo DOI optional |
| E4 | Final upload | user | After the checklist below passes; arXiv license selection at upload |

## F. arXiv v1 submission checklist

- [ ] Title/abstract/authors finalized (02 §1-2; decision log)
- [ ] All Tables 2–7 contain real numbers (no placeholders)
- [ ] All Figures 1–6 rendered from real data
- [ ] Ethics statement (04 §3) + disclosure note (04 §5) in the PDF
- [ ] Acknowledgments (MarkLLM / THU-BPM) present
- [ ] Citations re-verified (03; IDs + venue labels re-checked at submission)
- [ ] Data-availability + reproducibility statements (01 §7-8) written
- [ ] Release links live (corpus, configs, results JSONL, harness)
- [ ] E1/E2 done; E4 performed

## G. Deferred (v2, explicitly out of arXiv v1)

- Human eval (20 × 3 annotators) — journal/ACL requirement.
- File/metadata mini-study (C2PA/EXIF strip; v0.1 01 §4.3) — separate
  mini-experiment.
- Venue ladder beyond arXiv (NeurIPS 2026 workshop → ACL 2027 / TIFS).
