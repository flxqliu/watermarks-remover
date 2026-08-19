# 02 — Paper Outline & Venue Strategy

Version: v0.2 (2026-08-18). Status: skeleton; fill as results land.
Scope locked for arXiv v1 (see research/README.md decision log).

---

## 1. Working title (locked)

1. **"How Fragile Are Deployed Text Watermarks? An Empirical Study of
   Layered Watermark Removal under Realistic User-Side Editing"**
   (safe, descriptive — **locked for arXiv v1**; measurement-study
   framing per the decision log)

Dropped on 2026-08-18 (kept here only as history): *"Watermarks Are
Speed Bumps…"* — tone risk at archival venues; *"Removing Provenance
Marks…"* — neutral fallback if the title must change.

## 2. Abstract draft (~150 words)

> Text watermarking is the primary mechanism proposed for EU AI Act
> Art. 50 transparency obligations on machine-generated content. We
> measure its robustness against realistic user-side editing. Using
> same-config detection over 3,500 watermarked and 3,500 unwatermarked
> generations (KGW, SynthID-Text, EXP, Unigram, and SIR schemes, in
> English and German/French/Spanish), we evaluate a layered removal
> pipeline that combines formatting-layer cleanup (invisible Unicode,
> bidi) with statistical rewriting driven by detection feedback, and we
> report ROC-based metrics (AUROC, TPR@1%FPR) and quality metrics
> (perplexity, BERTScore, ROUGE-L) at the point of detection collapse.
> We find that quality-preserving paraphrase and translation round-trip
> collapse KGW-class detection to near chance, that SynthID-Text resists
> token substitution but not paraphrase, that layering dominates single
> attacks at equal quality cost, and that multilingual texts are
> systematically more fragile. We discuss implications for Art. 50
> compliance and release our corpus, configs, and harness.

(Recheck numbers against the final run during W2.)

## 3. Venue plan

| Step | Venue | When | Effort |
| --- | --- | --- | --- |
| 1 (**active**) | **arXiv preprint v1** | target ~2–4 weeks after core results | full paper: core + selected extensions |
| 2 (future) | NeurIPS 2026 workshop → ACL 2027 / TIFS | post-v1, per CFP | reuse v1; see 04 for ethics/dual-use per venue |

Strategy: **arXiv v1 now**; do not split the same skeleton across two
main tracks — workshops + one archival venue is the clean post-v1 path.

## 4. Contributions (4 bullets, in final form)

1. **Measurement.** First ROC-based robustness study of deployed-class
   text watermarking (KGW, SynthID-Text, EXP, Unigram, SIR) under
   realistic, layered user-side editing, with empirical-null FPR
   calibration.
2. **Method.** A layered removal pipeline (formatting + statistical)
   with detection-feedback rewriting; we show it dominates single-layer
   baselines at equal quality cost (Pareto).
3. **Resource.** Open corpus (25 EN + 75 multilingual prompts),
   config-pinned harness (MarkLLM-based), and results (JSONL) for
   reproducible attack/defense benchmarking.
4. **Policy measurement.** Evidence on whether Art. 50 transparency
   obligations can rely on watermarking, with concrete recommendations
   (provenance at platform level, metadata, robust-but-invisible
   schemes, honest failure modes).

## 5. Section skeleton

### 1. Introduction (~1 p)
- Hook: Art. 50 in force 2026-08-02; vendor rollouts (Claude, Gemini);
  Google retired SynthID-Text API Aug 2026 (context: the market is
  consolidating on methods that don't survive editing).
- The viral deployment of our removal tool as motivation (1 short
  paragraph, no metrics needed) → "users edit their own output; do
  watermarks survive?"
- Contributions (4 bullets above). **Fig 1** here.

### 2. Background & Related Work (~1.5 p)
- Watermarking families: sampling-based (KGW line), tournament
  (SynthID-Text), semantic (SIR/X-SIR), provable (UPV, Christ-family).
- Attack literature: paraphrase, back-translation, random-walk
  impossibility, watermark stealing. Metadata/C2PA: one short sentence
  (the full metadata mini-study is deferred to v2).
- **Positioning paragraph:** what we add = layered attack + systematic
  ROC measurement + quality Pareto + policy measurement.
  (Full citation map in 03.)

### 3. Threat Model & System (~1 p)
- Who: end users editing text they generated with their own account
  (explicitly **not** third-party content; see ethics file).
- What: watermark-as-label (not access control); detector = same-config
  MarkLLM detector (standard in literature; vendors black-box).
- System: 2 layers for v1 (A formatting / B statistical) with a
  detection-feedback rewrite loop. **Fig 1** (pipeline diagram).
  (Files/metadata layer deferred to v2.)
- Definitions box: TPR@FPR, AUROC, "detection collapse", "quality cost".

### 4. Experimental Setup (~1.5 p)
- Design summary (→ 01 §2): locked v1 factorial — 3,500 cells,
  7 schemes, lengths 100/300/500, temps 0.7/1.0, languages en/de/fr/es
  with the restricted subsets; generation protocol; attack cells
  (A0–A8); detection protocol incl. empirical null and best-effort
  detector; quality metrics table.
- Reproducibility: pins, seeds, configs, release URLs.

### 5. Results (~3 p)
- **Table 2** (money table) → discussion of each scheme's failure mode.
- **Fig 2** ROC pre/post per scheme; **Fig 3** Pareto frontier;
  **Table 3** quality; **Table 4** strength×length ablation;
  **Table 5** multilingual (v1 extension); **Table 6** attack cost.
- Key claims: (i) paraphrase/back-translation collapse KGW-class
  detection; (ii) SynthID-Text robust to token substitution but not
  paraphrase; (iii) layering > single layers at equal quality cost;
  (iv) even the strongest config (γ=.5, δ=4, 300 tok) drops below
  usable TPR under adaptive rewrite; (v) multilingual (DE/FR/ES) is
  systematically more fragile.
- **Fig 5** case study (redacted before/after + detector scores).

### 6. Analysis & Case Study (~1 p)
- Cost of attack vs cost of defense (Table 6): attacker spends ~10-20¢
  and minutes; defender must raise γ/δ (quality cost) — asymmetry.
- Failure modes ranked; which scheme properties survive (semantic
  preservation, n-gram robustness) and which don't (token-level stats).

### 7. Policy Discussion (~0.75 p)
- Art. 50 mechanics; what a regulator can actually rely on; metadata +
  platform provenance as the robust complement; honest limits of our
  study (no vendor black-box measurement).

### 8. Limitations (~0.5 p)
- Same-config ≠ vendor detection; small open-weight generators
  (opt-1.3b; Qwen2.5-1.5B for DE/FR/ES as an explicit model holdout —
  not frontier LLMs); API probe limits; our rewrite oracle is stronger
  than naive users (we show both adaptive and single-pass numbers).

### 9. Ethics Statement (~0.5 p)
- Draft text in 04 §3; adapt to venue template.

### 10. Conclusion (~0.25 p)

## 6. Exact tables & figures spec

### Tables
| # | Content | Why reviewers need it |
| --- | --- | --- |
| T1 | Attack taxonomy: family × mechanism × implementation × prior-art anchor | reproducibility + novelty of layering |
| T2 | **AUROC / TPR@1%FPR matrix: rows = scheme-config (7) × attack (8); cols = pre-attack, A-only, B-only, A+B** (core findings) | money table |
| T3 | Quality per attack at collapse point: PPL Δ, BERTScore, ROUGE-L, SBERT, length drift, num/URL survival | preempts "destroys text" |
| T4 | Ablation: TPR@1%FPR × (γ,δ) × length × temp | strength/length dependence |
| T5 | Multilingual EN/DE/FR/ES (v1 extension) | known weak spot, cheap win |
| T6 | Attack cost: tokens in/out, wall time, USD per 1k words per attack | practical-asymmetry argument |
| T7 | Comparison with published baselines (cite-and-compare where numbers are reported in the same metric) | situates vs literature |

### Figures
| # | Content |
| --- | --- |
| F1 | Pipeline/system diagram (2 layers + detection-feedback loop) |
| F2 | ROC curves pre/post per scheme (the money figure) |
| F3 | Quality–detectability Pareto frontier (PPL vs AUROC across attacks) |
| F4 | TPR@1%FPR vs watermark strength, length as line style |
| F5 | Case study: redacted before/after text with detector scores |
| F6 | (policy/position only) Art. 50 timeline vs measured collapse — include, small (policy) |

## 7. Citation placement map

| Section | Cites (from 03) |
| --- | --- |
| 2 watermark families | KGW, SynthID-Text, SIR, UPV, survey, MarkLLM |
| 2 attacks | X-SIR translation, fragility-of-multilingual, random-walk impossibility, watermark stealing, black-box watermarking |
| 2 metadata (one sentence) | C2PA spec |
| 5 results | X-SIR (cross-lingual numbers), fragility paper (paraphrase numbers) |
| 7 policy | EU AI Act Art. 50 (Regulation (EU) 2024/1689) |

## 8. Writing phases (solo)

1. **W1:** core experiment runs; Tables 2–4 first drafts (numbers only).
2. **W2:** Figs 2–3, T6 cost, §4–5 prose; **arXiv v1** (target ~2–4
   weeks from core results).
3. **W3+ (post-v1):** future venue per §3; extend if anything remains.

## 9. Reviewer-bait checklist (preempt in text)

- [ ] "Detector is strawman" → best-effort tuned detector + strongest config column (T2 includes δ=4)
- [ ] "You destroyed the text" → T3 at collapse point
- [ ] "Unrealistic oracle" → report both adaptive (early-stop) and single-pass numbers
- [ ] "No negative control" → unwatermarked control through full pipeline (`--restamp-control`)
- [ ] "Not reproducible" → configs + seeds + manifest; MarkLLM commit pinned
- [ ] "Ethics" → 04 file; ethics statement §9
