# Watermarks Remover for macOS

A native SwiftUI front end for the tools in this repo: drop files in, see what
AI provenance marks they carry, strip them, keep the report.

The app is a **thin client**, exactly like the agent skill. It does not
re-implement any cleaning logic: it starts `service/scripts/server.py` on a
private loopback port and speaks the documented HTTP API to it. Whatever the
Python pipeline supports, the app supports — the day a new format lands in
`format_dispatch.py`, the app routes it too.

```
┌──────────────────────┐   HTTP (127.0.0.1, bearer token)   ┌───────────────────┐
│ Watermarks Remover   │ ─────────────────────────────────► │ server.py         │
│ SwiftUI · macOS 13+  │ ◄───────────────────────────────── │ stdlib Python     │
└──────────────────────┘   /inspect /clean /detect          └───────────────────┘
```

## Build

```bash
make mac-app          # -> app/macos/build/Watermarks Remover.app
make mac-app-run      # build and launch
```

Requirements: macOS 13 Ventura or newer, Swift 5.9+ (Xcode 15 or the Command
Line Tools), and a Python 3.10+ interpreter on the machine that runs the app.

The build script compiles the SwiftPM executable, wraps it in an app bundle,
copies `service/scripts/*.py` into `Contents/Resources`, renders the icon
(`Scripts/make_icon.py`, stdlib-only) and ad-hoc signs the result.

## What it does

| Panel | What it gives you |
| --- | --- |
| **File queue** | Drag in files or folders. Each row shows the routed kind, a status dot, and a one-line verdict. |
| **Inspection** | Layer A hidden-character hits with codepoints and confidence, metadata findings, C2PA / AI-metadata flags, stylometry gauge, detector results, and the raw JSON. |
| **Clean** | Removed/replaced counts, per-action list, residue warnings, and a reveal button for the output. |
| **Text scratchpad** | Paste a draft, inspect or clean it in memory, copy the result back. Nothing touches disk. |
| **Capabilities** | Which of `c2patool`, `exiftool`, `qpdf`, pixel backends and text detectors the service actually found. |

Cleaned files are written next to the original as `name.cleaned.ext` by
default; a chosen output folder and in-place replacement (with a confirmation
step) are the other two modes.

## How the service is started

- Port: the kernel hands out a free loopback port per launch.
- Auth: a random bearer token per launch, passed through
  `WATERMARKS_SERVER_API_KEY` in the child environment — never through `argv`,
  so it does not show up in `ps`.
- `PATH`: extended with `/opt/homebrew/bin` and `/usr/local/bin`, because a GUI
  app inherits a bare `PATH` and would otherwise report Homebrew's `exiftool`
  and `qpdf` as missing.
- Shutdown: the child is terminated when the app quits (`ProcessRegistry`), so
  no interpreter is left listening.
- Settings → Service can point the app at a service you already run instead
  (Docker, a remote box) with its own URL and API key.

Nothing leaves the machine unless you explicitly turn on a detector that calls
a vendor API — those toggles are off by default and labelled.

## Layout

```
app/macos/
├── Package.swift                  SwiftPM executable, macOS 13+
├── Info.plist                     bundle metadata
├── Scripts/build-app.sh           compile + assemble + icon + ad-hoc sign
├── Scripts/make_icon.py           icon renderer (stdlib PNG writer)
└── Sources/WatermarksRemover/
    ├── WatermarksRemoverApp.swift entry point, menus, settings scene
    ├── AppModel.swift             queue, actions, output paths, report export
    ├── Model/                     JSON view, report summaries, options, prefs
    ├── Service/                   interpreter discovery, process lifecycle, client
    └── Views/                     sidebar, detail cards, scratchpad, footer
```

## Limits

- The app is not sandboxed: it reads and writes the files you point it at and
  spawns a local interpreter. Do not ship this build to the App Store as-is.
- Files are sent to the service one at a time (the batch endpoints exist, but
  per-file progress and bounded memory matter more here); the service's 256 MB
  input cap applies.
- Layer B rewriting is not wired into the UI. Stylometry scores are shown, and
  rewriting stays a deliberate step you take with `rewrite_text.py` or an agent.

## Responsible use

Same as the rest of the repo: this is for content **you own**. Removing
provenance from someone else's work, or to pass AI output off as human where
that matters (school, journalism, legal filings), is not what it is for. See
[`skills/remove-ai-marks/references/ethics.md`](../../skills/remove-ai-marks/references/ethics.md).
