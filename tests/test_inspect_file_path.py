"""inspect_file.py must print the filename so batched runs (find -exec ... >results.txt)
can be traced back to the file that produced each block. See issue #31."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_file.py"), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_text_kind_prints_path_as_first_line(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("plain prose, nothing suspicious\n", encoding="utf-8")
    r = run(str(target))
    lines = r.stdout.splitlines()
    assert lines[0] == f"Path: {target}"
    assert lines[1] == "Kind: text"


def test_text_kind_json_includes_path(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("plain prose\n", encoding="utf-8")
    r = run(str(target), "--json")
    data = json.loads(r.stdout)
    assert data["path"] == str(target)
    assert data["kind"] == "text"


def test_container_kind_prints_path_as_first_line():
    target = FIXTURES / "sample_meta.svg"
    r = run(str(target))
    lines = r.stdout.splitlines()
    assert lines[0] == f"Path: {target}"
    assert lines[1] == "Kind: container"
