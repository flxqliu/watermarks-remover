# Panoptes adapter

This repository ships a native adapter for the
[Panoptes](https://github.com/marketstandard/Panoptes) workbench, so anyone
can evaluate watermark removal here directly and get a signed before/after
card — no injected adapter, no glue code:

```bash
python -m bench evaluate-repo \
  https://github.com/guillaumemeyer/watermarks-remover \
  --kind watermark-remover
```

Panoptes discovers `panoptes.adapter.json` at the repo root, runs
`panoptes_adapter.transform` over its watermarked generations and
Unicode-embedded controls in a scrubbed subprocess, and measures detection
before vs. after.

## What the card means

The removal-retention evaluation covers two watermark families:

| Family | Before | After | Reading |
| --- | --- | --- | --- |
| Unicode (zero-width carriers) | 100% present | 0% present | Layer A scrubs the full family |
| KGW statistical (z-score) | 100% detected | 100% detected | Layer A does not touch statistical marks |

That asymmetry is the point: Unicode hygiene is complete for the family it
targets and a no-op for statistical watermarks. Removal claims should always
name the family they cover.

## Layer B opt-in

The adapter defaults to the deterministic Layer A scrub
(`service/scripts/text_unicode.clean_text`). To evaluate the statistical
rewrite as well, opt in:

```bash
WATERMARKS_REMOVER_ADAPTER_LAYER_B=1 \
WATERMARKS_REWRITE_BACKEND=ollama \
WATERMARKS_REWRITE_MODEL=llama3.2 \
python -m bench evaluate-repo ... --kind watermark-remover
```

Layer B reads the same `WATERMARKS_REWRITE_*` variables as
`rewrite_text.py` (see `.env.example`). The CLI's `print-prompt` backend is
rejected in the adapter — it returns a prompt, not cleaned text.

**Sandbox caveat:** Layer B defaults to a loopback backend (a local
Ollama). `evaluate-repo` scrubs env vars containing `API_KEY`/`TOKEN`/etc.
before running an adapter, so `WATERMARKS_REWRITE_API_KEY` never reaches the
adapter subprocess and keyed remote backends cannot authenticate there.
However, `WATERMARKS_REWRITE_ALLOW_REMOTE=1` is not a credential variable
and survives scrubbing — a *keyless* non-loopback backend can still receive
evaluated text under `evaluate-repo`. Networking is only disabled entirely
when the caller passes `--docker`. For keyed remote backends, run the
SynthID-text benchmark (`docs/synthid-text-benchmark.md`) instead.

## Security

Evaluating a repository clones and executes its code. Panoptes runs the
adapter in a subprocess with a scrubbed environment and a wall-clock limit;
pass `--docker` for a network-disabled container. See Panoptes'
`docs/testing-external-repos.md` for the full contract and threat model.
