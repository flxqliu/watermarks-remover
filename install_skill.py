#!/usr/bin/env python3
"""Install a bundled skill for Codex, Claude, or Cursor."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SKILL_NAME = "clean-user-facing-text"
SKILL_NAMES = (DEFAULT_SKILL_NAME, "remove-ai-marks")
TARGETS = ("codex", "claude", "cursor")
TARGET_SETTINGS = {
    "codex": ("CODEX_HOME", ".codex"),
    "claude": ("CLAUDE_HOME", ".claude"),
    "cursor": ("CURSOR_HOME", ".cursor"),
}


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _stage(skills_dir: Path, source: Path, skill_name: str) -> tuple[Path, Path]:
    skills_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{skill_name}.staging.", dir=skills_dir)
    )
    staged_skill = staging_root / skill_name
    try:
        shutil.copytree(source, staged_skill)
        if not (staged_skill / "SKILL.md").is_file():
            raise RuntimeError("staged skill is missing SKILL.md")
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root, staged_skill


def _install(
    destination: Path, source: Path, force: bool
) -> tuple[bool, Path | None]:
    if _present(destination) and not force:
        print(f"already exists: {destination}", file=sys.stderr)
        print(
            "No changes made. Re-run with --force to back up and replace.",
            file=sys.stderr,
        )
        return False, None

    staging_root, staged_skill = _stage(destination.parent, source, destination.name)
    backup: Path | None = None
    try:
        if _present(destination):
            backup = destination.with_name(
                f"{destination.name}.backup.{uuid.uuid4().hex[:12]}"
            )
            os.replace(destination, backup)
        try:
            os.replace(staged_skill, destination)
        except BaseException:
            if backup is not None and not _present(destination):
                os.replace(backup, destination)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return True, backup


def _agent_home(target: str, override: str | None) -> Path:
    environment_name, default_directory = TARGET_SETTINGS[target]
    configured = override or os.environ.get(environment_name)
    return Path(configured or Path.home() / default_directory).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=(*TARGETS, "all"),
        default="cursor",
        help="Agent to install for (default: cursor)",
    )
    parser.add_argument(
        "--skill",
        choices=SKILL_NAMES,
        default=DEFAULT_SKILL_NAME,
        help=f"Bundled skill to install (default: {DEFAULT_SKILL_NAME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Back up and replace existing installations",
    )
    parser.add_argument("--cursor-home", help="Override Cursor home (default: ~/.cursor)")
    parser.add_argument("--codex-home", help="Override Codex home (default: ~/.codex)")
    parser.add_argument("--claude-home", help="Override Claude home (default: ~/.claude)")
    args = parser.parse_args()

    selected_targets = TARGETS if args.target == "all" else (args.target,)
    overrides = {
        "codex": args.codex_home,
        "claude": args.claude_home,
        "cursor": args.cursor_home,
    }
    source = ROOT / "skills" / args.skill
    destinations = {
        target: _agent_home(target, overrides[target]) / "skills" / args.skill
        for target in selected_targets
    }

    if not args.force:
        existing = [path for path in destinations.values() if _present(path)]
        if existing:
            for path in existing:
                print(f"already exists: {path}", file=sys.stderr)
            print(
                "No changes made. Re-run with --force to back up and replace.",
                file=sys.stderr,
            )
            return 1

    for target, destination in destinations.items():
        installed, backup = _install(destination, source, args.force)
        if not installed:
            return 1
        label = target.capitalize()
        if backup is not None:
            print(f"{label}: backed up existing skill to {backup}")
        print(f"{label}: installed {destination}")

    print("Start a new agent session if the skill does not appear automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
