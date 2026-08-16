"""Tests for the bundled Claude Desktop skill package.

The Desktop variant is assembled at build time by package_desktop_skill.py:
the engine scripts are copied fresh from service/scripts, so there are no
vendored duplicates to keep in sync (unlike the Cursor skill).
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from package_desktop_skill import BUNDLED_SCRIPTS, SKILL_DIR_IN_ZIP, build_zip  # noqa: E402

REFERENCES = sorted(
    p.name for p in (ROOT / "skills" / "remove-ai-marks" / "references").glob("*.md")
)


@pytest.fixture(scope="module")
def built_zip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("desktop-skill") / "remove-ai-marks-desktop.zip"
    return build_zip(out)


def test_zip_member_list_is_exact(built_zip):
    with zipfile.ZipFile(built_zip) as zf:
        members = sorted(n for n in zf.namelist() if not n.endswith("/"))
    expected = sorted(
        [f"{SKILL_DIR_IN_ZIP}/SKILL.md"]
        + [f"{SKILL_DIR_IN_ZIP}/references/{name}" for name in REFERENCES]
        + [f"{SKILL_DIR_IN_ZIP}/scripts/{name}" for name in BUNDLED_SCRIPTS]
    )
    assert members == expected


def test_zip_excludes_server_and_backend_files(built_zip):
    with zipfile.ZipFile(built_zip) as zf:
        names = zf.namelist()
    for banned in ("server.py", "audit_website.py", "setup_", "requirements-"):
        assert not any(banned in n for n in names), f"{banned} leaked into the bundle"


def test_skill_md_frontmatter(built_zip):
    with zipfile.ZipFile(built_zip) as zf:
        raw = zf.read(f"{SKILL_DIR_IN_ZIP}/SKILL.md").decode("utf-8")
    # Checkout may materialise the file with either line ending.
    text = raw.replace("\r\n", "\n")
    assert text.startswith("---\n")
    frontmatter = text.split("---", 2)[1]
    assert "name: remove-ai-marks" in frontmatter
    assert "description:" in frontmatter
    # The Desktop variant must not instruct HTTP use.
    body = text.split("---", 2)[2]
    assert "curl" not in body
    assert "scripts/inspect_file.py" in body
    assert "scripts/clean_file.py" in body


def test_bundled_scripts_match_service_copies(built_zip):
    with zipfile.ZipFile(built_zip) as zf:
        for name in BUNDLED_SCRIPTS:
            bundled = zf.read(f"{SKILL_DIR_IN_ZIP}/scripts/{name}")
            service = (ROOT / "service" / "scripts" / name).read_bytes()
            assert bundled == service, f"{name} differs from service copy"


def test_extracted_bundle_is_self_contained(built_zip, tmp_path):
    with zipfile.ZipFile(built_zip) as zf:
        zf.extractall(tmp_path)
    scripts = tmp_path / SKILL_DIR_IN_ZIP / "scripts"

    sample = tmp_path / "sample.md"
    sample.write_text("# Title\n\nbody​here\n", encoding="utf-8")

    inspect = subprocess.run(
        [sys.executable, str(scripts / "inspect_file.py"), str(sample), "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert inspect.returncode == 1  # ZWSP present -> suspicious
    report = json.loads(inspect.stdout)
    assert report["kind"] == "container"

    clean = subprocess.run(
        [sys.executable, str(scripts / "clean_file.py"), str(sample), "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert clean.returncode == 0, clean.stderr
    cleaned = tmp_path / "sample.cleaned.md"
    assert cleaned.is_file()
    assert "​" not in cleaned.read_text(encoding="utf-8")


def test_cli_writes_zip(tmp_path):
    out = tmp_path / "bundle.zip"
    result = subprocess.run(
        [sys.executable, str(ROOT / "package_desktop_skill.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        assert f"{SKILL_DIR_IN_ZIP}/SKILL.md" in zf.namelist()
