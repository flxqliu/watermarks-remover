#!/usr/bin/env python3
"""Package the self-contained remove-ai-marks skill for Claude Desktop.

Claude Desktop runs skills inside a sandboxed VM that cannot reach servers on
the host, so the HTTP-client skill (skills/remove-ai-marks) does not work
there. This script assembles a zip whose scripts are copied fresh from
service/scripts at build time: one source of truth, no vendored duplicates.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL_SOURCE = ROOT / "skills" / "remove-ai-marks-desktop"
REFERENCES_SOURCE = ROOT / "skills" / "remove-ai-marks" / "references"
SCRIPTS_SOURCE = ROOT / "service" / "scripts"

# Folder name inside the zip; matches the frontmatter name so Desktop imports
# it as a drop-in replacement for the HTTP-client skill.
SKILL_DIR_IN_ZIP = "remove-ai-marks"
DEFAULT_OUT = ROOT / "dist" / "remove-ai-marks-desktop.zip"

# Stdlib-only modules reachable from the unified CLIs, plus the offline-capable
# extras (audit_dir for folders, rewrite_text's print-prompt backend for
# Layer B). server.py, audit_website.py and every optional heavy backend stay
# out: they need the host network or external checkouts the sandbox lacks.
BUNDLED_SCRIPTS = [
    "audit_dir.py",
    "audit_lib.py",
    "clean_file.py",
    "clean_text.py",
    "common.py",
    "container_meta.py",
    "format_dispatch.py",
    "image_meta.py",
    "inspect_file.py",
    "inspect_text.py",
    "rewrite_text.py",
    "score_stylometry.py",
    "text_unicode.py",
]


def build_zip(out: Path) -> Path:
    skill_md = SKILL_SOURCE / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"missing {skill_md}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(skill_md, f"{SKILL_DIR_IN_ZIP}/SKILL.md")
        for ref in sorted(REFERENCES_SOURCE.glob("*.md")):
            zf.write(ref, f"{SKILL_DIR_IN_ZIP}/references/{ref.name}")
        for name in BUNDLED_SCRIPTS:
            src = SCRIPTS_SOURCE / name
            if not src.is_file():
                raise FileNotFoundError(f"missing bundled script: {src}")
            zf.write(src, f"{SKILL_DIR_IN_ZIP}/scripts/{name}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output zip path (default: {DEFAULT_OUT.relative_to(ROOT)})",
    )
    args = p.parse_args()
    path = build_zip(args.out)
    print(f"wrote {path}")
    print("Import it in Claude Desktop: Settings > Capabilities > Skills > Upload skill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
