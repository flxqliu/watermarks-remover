# SynthID-text removal benchmark

bench_synthid_text.py measures how well the Layer B rewrite
(rewrite_text.py) removes SynthID-text-class watermarks, and at what
cost. It generates a controlled corpus with the MarkLLM SynthID scheme, runs
removal variants, and emits a shareable report.

## What it measures

| Metric | Meaning |
| --- | --- |
| Clear rate | % of watermarked samples that flip to not-watermarked after removal (MarkLLM same-config detection) |
| Score suppression | mean/median drop in detector score (before - after) |
| Quality | lexical divergence (bigram Jaccard distance), length drift, number/URL survival |
| Cost | estimated tokens in/out, wall time per document, optional USD at your prices |
| Efficiency | clears per million output tokens - removal rate per unit of rewrite cost |
| Controls | Layer A only (expect ~0% - Unicode scrub must not clear a statistical mark), sanity-gate exclusions, optional re-stamp check |

## How to run

Prerequisites (all external, matching the repo's optional-harness model):

1. A MarkLLM checkout: run service/scripts/setup_markllm.sh (clones
   THU-BPM/MarkLLM at a pinned commit and creates ~/MarkLLM/.venv).
2. A rewrite backend: Ollama (default, loopback) or any
   OpenAI-compatible endpoint. The rewrite model must be a real model.

    # minimal: 3 docs, 1 seed, paraphrase with 1 and 3 candidates (Ollama)
    MARKLLM_DIR=~/MarkLLM \
    python3 service/scripts/bench_synthid_text.py \
      --markllm-dir ~/MarkLLM \
      --rewrite-backend ollama --rewrite-model llama3.2 \
      --out-dir out/bench-2026-06-01

    # recommended full run: more docs/seeds, backtranslate variant, re-stamp control
    python3 service/scripts/bench_synthid_text.py \
      --markllm-dir ~/MarkLLM \
      --docs 10 --seeds 3 \
      --variants "paraphrase:1,paraphrase:3,backtranslate:1" \
      --restamp-control \
      --rewrite-backend openai-compatible \
      --rewrite-model deepseek-v4-flash \
      --rewrite-base-url https://api.deepseek.com \
      --rewrite-allow-remote \
      --out-dir out/bench-deepseek \
      --tag deepseek-v4-flash

API keys are read from the environment only (WATERMARKS_REWRITE_API_KEY),
never argv. Non-loopback rewrite endpoints require --rewrite-allow-remote.

No vendor tier: Google retired SynthID text watermarking on its API in
Aug 2026 (DETECT_TEXT_WATERMARK is rejected on current models), so detection
here is MarkLLM same-config only. A vendor tier can be re-added if Google
exposes detection again (e.g. via Vertex AI).

Cost modeling: --cost-per-mtok-in 0.30 --cost-per-mtok-out 1.20 (example
prices) attaches an estimated USD figure per row; token counts are
chars / --chars-per-token estimates (default 4.0).

## Outputs (in --out-dir)

- report.md - self-contained Markdown you can paste anywhere: methodology,
  config, results table, controls, caveats, exact reproduction command.
- results.json - full per-sample/per-row data + aggregates.
- results.csv - one row per (doc, seed, variant) for plotting.
- work/ - generated watermarked/unwatermarked samples (kept for inspection).

## Running from Docker (compose)

The `wr-markllm` service in compose.yaml can run the benchmark end-to-end
(image: pinned MarkLLM checkout at /opt/markllm + all scripts). The image
installs CPU torch by design, so use it for portability/CI, not for GPU
throughput on this machine — for GPU runs use the host `setup_markllm.sh`
venv instead (see README).

```bash
docker compose --profile harness build wr-markllm
docker compose run --rm wr-markllm \
  /app/bench_synthid_text.py --markllm-dir /opt/markllm \
  --corpus /bench-corpus --out-dir /data --tag docker-run \
  --docs 10 --seeds 3 --variants "paraphrase:1,paraphrase:3,backtranslate:1" \
  --restamp-control
```

Env (rewrite backend) is wired from your .env via compose interpolation;
results land in the `bench-out` volume (/data); the bundled
corpus is mounted read-only at /bench-corpus. The image runs the
persistent MarkLLM serve worker by default, so the ~2-4h one-shot runs
are not a constraint inside the container either.

## What it can and cannot claim

- Can claim: under the MarkLLM SynthID scheme config the benchmark
  controls, at these seeds/docs, with this rewrite backend, this clear rate and
  cost were observed. Same-config-only detection is deterministic and
  reproducible (fixed seeds, pinned MarkLLM commit, recorded commands).
- Cannot claim: that Google's production SynthID-Text detector will fail.
  MarkLLM's SynthID is a research reimplementation with a different keying,
  and Google retired text watermark detection on its API (Aug 2026), so no
  vendor tier exists to verify against. Rewriting with a watermarked model
  can also re-stamp the text - run --restamp-control to check.

## Sharing a run

Share the --out-dir directory. report.md embeds the reproduction command,
the MarkLLM commit, the watermarks-remover commit, and the caveats, so a reader
can (a) trust what was measured and (b) rerun it. Keep work/ out of archives
unless you want the raw samples.

## Notes on statistical power

- A single document tells you nothing - the watermark is probabilistic. Use
  several documents (--docs 10+) and several seeds per document
  (--seeds 3+) so clear-rate differences are distinguishable.
- Longer text carries more watermark signal: default --max-new-tokens 300.
  Very short samples are excluded by the sanity gate automatically.
- Compare variants (strength x candidates) within one run, not across runs
  with different backends - the rewrite model dominates the outcome.
