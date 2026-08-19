# 01 — Experimental Protocol

*Working title of the study: "How fragile are deployed text watermarks?
A measurement study of multi-layer watermark removal under realistic
user-side editing."*

Version: v0.2 (2026-08-18). Scope locked for arXiv v1 (see
research/README.md decision log). Owner: Guillaume.

---

## 1. Research questions

- **RQ1 (robustness).** How robust are deployed-class text watermarking
  schemes (KGW, SynthID-Text, EXP, Unigram, SIR) to realistic user-side
  editing (paraphrase, translation round-trip, structural rewrite,
  humanization, Unicode/formatting cleanup), measured with ROC-based
  detection metrics?
- **RQ2 (layering).** Does a *layered* removal pipeline (formatting-layer
  cleanup **+** statistical rewrite) outperform single-layer baselines at
  equal text-quality cost? Is the gain additive, or does one layer dominate?
- **RQ3 (frontier).** What is the quality–detectability Pareto frontier?
  I.e., what detection rate can a watermarker keep while text remains
  usable (PPL/BERTScore within tolerance), under each attack?
- **RQ4 (policy, secondary).** The EU AI Act Art. 50 transparency regime
  (in force 2026-08-02) leans on watermarking. Does the mechanism survive
  contact with real users? (Feeds §7 of the paper and the ethics/position
  angle; not required for the core experiment.)

**Primary claim to defend:** *Under realistic editing, quality-preserving
removal collapses TPR@1%FPR of KGW-class watermarks to near chance;
SynthID-Text resists token-level substitution but not paraphrase /
back-translation; a layered pipeline dominates single layers at equal
quality cost.*

**Anti-claim we must preempt:** "Your detector is misconfigured / your
attacks destroy the text / a stronger watermark config would survive."
Mitigations in §5.3, §6, §10.

---

## 2. Design overview

Factorial experiment, paired design: the same watermarked document is
attacked by every attack condition, and detection is run on every
(doc, attack) pair with the same watermark key/config. Unwatermarked
control texts run through the same attack+detect pipeline
(already supported via `--restamp-control`).

### v1 factorial (locked 2026-08-18)

| Grid | Schemes | Lengths | Temp | Langs | Prompts | Seeds | Cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EN core | KGW (γ=.25, δ=1); KGW (γ=.5, δ=2); KGW (γ=.5, δ=4); SynthID-Text (default MarkLLM config); EXP (Gumbel); Unigram; SIR | 100, 300 | 0.7 | en | 25 | 5 | 1,750 |
| EN temp axis | 4 core schemes (KGW×3, SynthID) | 300 | 1.0 | en | 25 | 5 | 500 |
| EN length axis | 4 core schemes | 500 | 0.7 | en | 25 | 5 | 500 |
| Multilingual | KGW (γ=.5, δ=2); SynthID-Text | 300 | 0.7 | de, fr, es | 25 | 5 | 750 |
| **Total** | | | | | | | **3,500 cells** |

→ 3,500 watermarked + 3,500 unwatermarked generations = **7,000
texts**; each passed through 8 attack cells (§4) = **56,000 attack
outputs**; ~65,000 detection runs (originals + attacks + controls)
plus ~2,000 unwatermarked texts for the empirical null.

Restrictions are deliberate (a naive full cartesian — 7 schemes × 3
lengths × 2 temps × 4 langs × 25 × 5 — would be 21,000 cells and
infeasible on CPU): the temp axis, length axis, and multilingual grid
each vary exactly one factor against the 4 core schemes so the effects
are attributable. See §3 for the multilingual generator (model holdout).

### Deferred (post-v1, not in arXiv v1)

- Human eval (20 samples × 3 annotators) — journal/ACL requirement.
- File/metadata mini-study (C2PA/EXIF strip on 20 self-made files;
  v0.1 §4.3) — separate mini-experiment, revisit for v2.

---

## 3. Generation protocol

- **EN core:** `facebook/opt-1.3b` via MarkLLM (matches existing
  harness; CPU-feasible).
- **Multilingual (DE/FR/ES):** opt-1.3b is English-only, so these cells
  use a CPU-feasible multilingual generator — **Qwen/Qwen2.5-1.5B-
  Instruct** (fallback: 0.5B if too slow) — reported explicitly as a
  model-holdout factor in the paper (scheme × language, not confounded
  with the EN core). If the multilingual pilot (§9) shows unacceptable
  output quality, shrink the grid to 10 prompts per language before
  committing the full run.
- Prompts: extend `benchmarks/corpus/` to **25 factual, neutral EN
  seeds** (50-90 words, varied domains; keep claims checkable) plus
  **3×25 translated DE/FR/ES sets**; record all prompts in
  `research/corpus/` (additive, not a fork).
- Decoding: **temperature 0.7, top-p 0.95** (realistic); record
  `do_sample=True`, fixed seeds 1..5 per (prompt, seed) pair.
- Watermark configs: fixed per scheme; copy the MarkLLM config JSONs
  (KGW: γ/δ/hash_key/f_scheme/window_scheme; SynthID; EXP; Unigram;
  SIR) into `research/configs/` and **use the same JSON for
  generation and detection**. Synthetic keys — fine to commit; note it.
- Control: same prompts, same seeds, no watermark → empirical null
  distribution for FPR calibration (§5.2).

## 4. Attack conditions

### 4.1 Text-level attacks

| # | Attack | Implementation | Prior-art anchor |
| --- | --- | --- | --- |
| A0 | None (control) | — | — |
| A1 | Layer A only: Unicode/invisible-char, bidi, tag cleanup | `clean_text.py` (deterministic) | formatting-layer marks (zero-width/steganography class) |
| A2 | Paraphrase, single pass | `rewrite_text.py --strength paraphrase --candidates 1 --max-loops 1` | paraphrase attacks in watermark literature |
| A3 | Paraphrase, adaptive (early stop on detection, up to 3 loops) | `rewrite_text.py --strength paraphrase --candidates 3 --max-loops 3 --markllm-scheme <scheme>` | same; our eval-loop is the "oracle" version |
| A4 | Back-translation round trip EN→DE→EN | `rewrite_text.py --strength backtranslate --lang German` | "Can Watermarks Survive Translation?" (X-SIR, ACL 2024) |
| A5 | Structural: outline → regenerate | `rewrite_text.py --strength structural` | summarization/outline attacks |
| A6 | Humanize | `rewrite_text.py --strength humanize` | style-transfer attacks |
| A7 | Cheap baselines: synonym substitution, random word deletion (5-10%), sentence reorder | `research/scripts/attacks/cheap.py` (to be written — gap 05-A3) | Random Walk / impossibility (ICML 2024) |
| A8 | **Full pipeline: A1 → A2/A3** | `clean_text.py` then `rewrite_text.py` | our layered contribution |

Notes:
- CLI names above match the real `rewrite_text.py` flags: `--strength`
  selects the rewrite type; `--backend` selects the transport
  (`ollama` / `openai-compatible`); `--candidates` × `--max-loops`
  control the detection-guided loop.
- The Layer B rewrite loop stops early when detection passes — that's an
  *adaptive* attack (stronger than one-shot). Report both: with
  early-stop (adaptive, "removal oracle") and forced single-pass.
- Record rewrite cost (tokens in/out, wall time, USD) per cell — this is
  the "practical attack cost" contribution (Table 6).

### 4.2 Detector-side conditions

- Detectors: MarkLLM same-config (KGW z-score; SynthID mean/weighted/
  bayesian; EXP/Unigram/SIR per their configs).
- Also run **best-effort detection**: threshold tuned on held-out data
  (favorable to the watermarker; preempts "weak threshold" criticism).
- Explicit **strawman preemption**: run the strongest config we have
  (KGW γ=0.5 δ=4, long text) as the "best case for watermarking" column.

## 5. Detection & evaluation protocol

### 5.1 Metrics (primary)

- **AUROC** over (watermarked, unwatermarked) score distributions.
- **TPR @ FPR ∈ {0.1%, 1%, 10%}** read off the ROC.
- Full ROC curves saved per cell (for Figure 2).
- Secondary: clear rate (existing bench metric, before-positive →
  after-negative), mean/median score suppression.
- **95% bootstrap CIs (10k resamples) on AUROC and TPR@FPR**; report
  effect sizes, not just p-values.

### 5.2 Empirical null

- ≥1,000 unwatermarked generations (same prompts/seeds/temps), scored by
  the *same* detector. Do **not** assume z-scores are standard normal for
  FPR calibration; SynthID tournament scores definitely aren't. Empirical
  null is mandatory — use all unwatermarked controls.

### 5.3 Quality metrics

| Metric | Model/tool | Why |
| --- | --- | --- |
| PPL | `gpt2-large` (independent of generator opt-1.3b — never score with the generator) | fluency |
| BERTScore | `deberta-xlarge-mnli` | semantic preservation |
| ROUGE-L | standard | content overlap |
| SBERT cosine | `all-MiniLM-L6-v2` | cheap semantic similarity |
| Levenshtein distance % | stdlib | edit magnitude |
| Length drift %, number/URL survival | already in bench | practical usability |

**Quality @ detection-collapse**: for each attack, report quality metrics
*at the operating point where detection collapses* (the Pareto framing).
This is the figure that makes the paper defensible: "removal without
destruction." To bound CPU cost, compute full quality on a stratified
subset (~2-4k texts covering every scheme × attack × collapse point).

### 5.4 Statistics

- 5 seeds/cell, paired design; 95% bootstrap CIs as above.
- Detection randomness: detection scores are deterministic given
  (text, key) for KGW; SynthID scoring may have randomness — seed it.
- Multiple-comparison discipline: pre-register the headline cells (the
  v1 matrix); treat nothing else as confirmatory.

## 6. Budget & compute (estimates)

CPU-only (opt-1.3b; Qwen2.5-1.5B for multilingual; MarkLLM). Rough
per-op figures from existing runs: gen ≈ 1-3 min/text (CPU), detect ≈
10-40 s/text.

| Stage | Volume | CPU-hours (1 core) | Parallel 8 cores |
| --- | --- | --- | --- |
| Generate (incl. multilingual) | 7,000 texts | ~250-400 | ~35-50 h |
| Detect (originals+attacks+controls) | ~65,000 | ~180-720 | ~25-90 h |
| Null corpus scoring | 2,000 | ~10-20 | ~2-3 h |
| Rewrites (API) | 56,000 outputs | — | ~1-2 days (rate-limited) |
| Quality metrics (stratified subset) | ~2-4k texts | ~60-150 | ~10-20 h |

API rewrite budget: **~$50-120** (paraphrase ≈ 1.3× input tokens per
pass; back-translate ≈ 2×2 passes; structural ≈ 2 passes; cheap
instruct model, temperature 0.9 — record model + version).

→ **v1 ≈ 2.5-4 weeks of part-time work on one 8-core box + API
budget.** That is the arXiv-v1 core; the extended scope adds ~1-2 weeks
over the original Tier-1-only estimate.

## 7. Reproducibility

- Pin: MarkLLM checkout commit (`git -C /home/guillaume/MarkLLM
  rev-parse HEAD`), HF model revisions (opt-1.3b, Qwen2.5-1.5B,
  gpt2-large, deberta-xlarge-mnli, all-MiniLM-L6-v2), all scheme
  configs (copied into `research/configs/`).
- Seed policy: every random op seeded; seeds recorded per cell in
  `results/manifest.json`.
- Environment: record `pip freeze` for the MarkLLM env (it is separate
  from the service stdlib env) and for the quality-metric env.
- Release plan: `research/results/` (JSONL per cell), analysis
  notebooks, corpus, and a `Makefile`-style runner
  (`run_experiments.py --plan`).

## 8. Artifacts & release

1. `research/corpus/` — 25 EN + 75 multilingual prompts.
2. `research/configs/` — scheme configs + keys + model pins.
3. `research/results/` — scores, ROC data, quality metrics, manifest.
4. `research/scripts/` — attack additions (cheap.py, gap 05-A3),
   analysis notebooks.
5. Paper artifacts: tables 1-7, figures 1-6 (spec in 02 §6).

## 9. Smoke test (do this first, before the full run)

Purpose: verify the pipeline end-to-end (generation → attack → detect
→ evaluate) on one scheme and a handful of prompts before committing
the 3,500-cell run.

```bash
# Requires the multi-scheme bench refactor (gap 05-A1) to have landed.
python3 service/scripts/bench_synthid_text.py \
  --corpus research/corpus --docs 5 --seeds 2 --max-new-tokens 300 \
  --variants paraphrase:3 --restamp-control --tag smoke \
  --markllm-model facebook/opt-1.3b
```

Then, once the ROC/quality analysis modules (gaps 05-B1/B2) exist:
compute AUROC/TPR@FPR + quality on the smoke output, and sanity-check
the multilingual pilot (5 prompts × 1 seed per language, DE/FR/ES, via
Qwen2.5-1.5B) for output quality before the full multilingual grid.

## 10. Risk register

| Risk | Mitigation |
| --- | --- |
| "Detector is misconfigured" | Same-config detection is the field standard; add best-effort tuned detector + strongest-config column; identical config JSON for gen and detect (§3) |
| "Attacks destroy text" | Quality metrics at collapse point; show PPL/BERTScore within tolerance |
| "Not real vendors" | Honest limitation; vendor APIs are black-box/retired (Google retired SynthID API Aug 2026); optional Claude API probe study if ToS allows |
| "Already known" (novelty) | Novelty = layered attack + systematic ROC measurement across schemes + quality Pareto + policy measurement; check 03-related-work for overlap |
| Ethics rejection | 04-ethics-and-legal.md; frame as robustness evaluation of deployed mechanisms |
| Compute blowup | Locked restricted matrix (3,500 cells, not 21,000); `--plan` budget mode in run_experiments.py; stage checkpoints |
| **Multilingual generation quality** | opt-1.3b is English-only → Qwen2.5-1.5B holdout; 5-prompt pilot check before the grid; fallback: shrink to 10 prompts/lang or defer to v1.1 |
| Bench hardcodes SynthID | Add `--scheme kgw|synthid|exp|unigram|sir` to bench (gap 05-A1; small refactor) |
