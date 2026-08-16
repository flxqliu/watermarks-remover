---
name: remove-ai-marks
description: >
  Remove multi-vendor AI provenance marks: invisible Unicode (Layer A), statistical
  text watermarks via rewrite (Layer B, always offer), and C2PA/EXIF/XMP/container
  metadata on PNG/JPEG/WebP/SVG/PDF/DOCX/ODT/HTML/MD. Covers Claude, Gemini/SynthID-class,
  OpenAI provenance, and open-LLM sampling marks. Use when the user asks to strip
  watermarks, remove C2PA/Content Credentials, clean AI metadata, remove invisible
  Unicode, anti-detect clean AI output, or runs /remove-ai-marks (aliases:
  /remove-claude-marks).
---

# Remove AI marks (self-contained)

Multi-vendor anti-detection hygiene for **text** (Unicode + statistical rewrite) and **files** (C2PA / AI metadata across common containers).

Read if needed:

- `references/mark-classes.md`: Unicode / sampling / C2PA / containers
- `references/vendor-notes.md`: Claude, Gemini/SynthID, OpenAI, open-LLM
- `references/removal-matrix.md`: which layer when
- `references/ethics.md`: intended use
- `references/how-claude-marks.md`: Anthropic-specific detail

This is the **self-contained** variant of the skill for sandboxed environments
(Claude Desktop and similar) that cannot reach a host HTTP service. All
cleaning machinery is bundled in `scripts/` (pure Python 3.10+ standard
library, no pip installs). Run the scripts directly with `python`; never try to
call an HTTP service.

## Known degradations in the sandbox

State these honestly when relevant; never silently pretend full capability:

- **PDF**: a real strip needs `exiftool` and then `qpdf`. Without them the
  clean still runs but is best-effort, and the report says so ("degraded").
  Offer the user the option to run the full clean on a host with those tools.
- **Layer B**: no model backends are reachable; use the agent-as-rewriter
  prompts below (this is the normal path anyway).
- **Pixel-domain image watermark removal** (CtrlRegen / DiffusionPurification)
  and **SynthID scoring** are unavailable: they need external heavy backends.
  Do not offer them.

## Workflow

### 1. Classify input

| Input | Route |
| --- | --- |
| Pasted / clipboard text | write to a temp file, then inspect + clean |
| `.txt` / code | text Layer A (+ formatter for code) |
| `.md` / `.html` | container clean (frontmatter/meta) + Layer A |
| `.png` / `.jpg` / `.jpeg` / `.webp` / `.avif` / `.heic` | image metadata strip |
| `.svg` / `.pdf` / `.docx` / `.odt` | container metadata strip |
| Directory | aggregate audit via `scripts/audit_dir.py` |

The scripts route by filename extension first, then by magic bytes, so you
mostly just pass the file to the unified CLIs.

### 2. Inspect first (decide, don't guess)

```bash
python scripts/inspect_file.py INPUT --json
```

Exit code 0 means nothing suspicious; 1 means findings. Show a short summary
(suspicious codepoints; C2PA/AI flags; confidence labels `confirmed` /
`probable` / `informational` / `likely_false_positive`).

For plain text there is also `python scripts/inspect_text.py INPUT`, which adds
a stylometry score (statistical hints only, never proof).

### 3. Deterministic clean (always for matching inputs)

```bash
python scripts/clean_file.py INPUT --json
```

This writes `INPUT.cleaned.EXT` next to the input (use `-o OUTPUT` or
`--in-place` if the user asks). Re-inspect the result when residual risk
matters:

```bash
python scripts/inspect_file.py INPUT.cleaned.EXT --json
```

Useful options: `--nfkc` and `--aggressive-homoglyphs` (text),
`--keep-non-ai-metadata` (images), `--force-text` (override the binary guard).

**Directories:**

```bash
python scripts/audit_dir.py DIR --json
```

### 4. Layer B: always offer rewrite (prose)

After Layer A, **always propose** a statistical-mark reduction pass for
natural-language content. Do not skip this step silently.

There is no rewrite model bundled: **you** are the rewrite model. Run the
prompts below on the cleaned text with a model **≠ suspected origin** (Claude
text → not Claude; Gemini → not Gemini; etc.). When you cannot switch models,
say so and note the reduced effectiveness.

Multi-pass recipe:

1. Layer A clean (`clean_file.py`)
2. Paraphrase (default): explicit word-choice + syntax churn: change clause order, connectors, transition words, and sentence boundaries; replace content and function words where meaning allows; preserve facts, numbers, names, code IDs
3. Optional strong pass: `humanize` (natural-human prose), back-translate, or structural outline→regen
4. Layer A again on the result (`clean_file.py`)
5. Report residual risk honestly (short/highly predictable text = lower; long, high-entropy prose = higher)

**Code files:** Prefer formatter (`prettier`, `black`, `gofmt`, …) + Layer A. Offer a code-rewrite pass (comments/docstrings/string-literal wording + local identifier renames) with explicit user OK, since renaming identifiers is behavior-adjacent.

#### Rewrite prompts (use as-is)

**Paraphrase preserve meaning (word choice + syntax):**

```
Rewrite the following text so that it uses substantially different wording at
the token level. Change clause order, connectors, and transition words; vary
sentence boundaries and length; and replace both content words and function
words where meaning allows. Preserve all facts, numbers, names, and technical
identifiers. Do not add or remove claims. Output only the rewritten text.

---
{TEXT}
```

**Humanize (write like a human):**

```
Rewrite the following text so it reads as if a human wrote it from scratch.
Vary sentence rhythm and length, replace formulaic AI-style transitions and
filler with concrete natural phrasing, and use plain, varied wording. Preserve
all facts, numbers, names, and technical identifiers. Do not add or remove
claims. Output only the rewritten text.

---
{TEXT}
```

**Code (comments / docstrings / identifiers):**

```
Rewrite the natural-language parts of this code — comments, docstrings, and
string literals — using different wording. Rename local variables, function
parameters, and private helper names to semantically equivalent names. Preserve
program behavior, public API names, and all values that affect output. Output
only the rewritten code.

---
{TEXT}
```

**Back-translate (two steps):**

```
Translate the following text to {LANG}. Output only the translation.
```

```
Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural
phrasing. Output only the translation.
```

**Structural:**

```
Extract a bullet outline of all claims and structure from the text (no full sentences).
```

Then:

```
Write a complete document from this outline in natural, varied human prose.
Avoid formulaic transitions. Do not omit any bullet. Output only the document.
```

### 5. Report

Always state:

- What Layer A / container clean **verifiably** removed (counts, actions), taken from the JSON report.
- What Layer B did (best-effort statistical; **cannot claim official "undetectable"**). Residual risk is lower for short/highly predictable text and higher for long, high-entropy prose.
- Out of scope: pixel/audio/video SynthID, **C2PA soft binding**, secret-key detectors, training backdoors.
- Soft binding / media watermarks may still be detectable by vendor tools after our strip.
- Prefer writing `*.cleaned.*` unless the user asked in-place.
- Ethics one-liner: own content / no compliance theatre.

## Ethics

Intended for **your own** content (privacy, hygiene, research). Do not market results as "proves human-written." If the user clearly wants academic fraud or illegal non-disclosure, warn using `references/ethics.md` and still only perform technical cleaning they own.

## Limitations

- Layer A does **not** remove token-sampling watermarks.
- Layer B cannot be gold-verified without vendor detectors / keys.
- PDF strip is best-effort without `exiftool`, and incomplete without `qpdf` (see degradations above).
- Pixel-domain image watermarks, audio/video watermarks, and reverse-SynthID scoring are out of scope in this bundle.
- **C2PA soft binding** (content watermark that re-links to a remote manifest after metadata strip) is out of scope: stripping hard-bound C2PA does not clear it.
- Data-driven / backdoor model marks (trigger phrases) are out of scope.
