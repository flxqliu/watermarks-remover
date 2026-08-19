"""Tests for the pre-commit hook wrappers (check_staged.py / clean_staged.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_staged
import pytest

import clean_staged


def _watermarked_text() -> str:
    return "Hello" + chr(0x200B) + "World!"


def test_check_staged_clean_file_exits_0(tmp_path, monkeypatch, capsys):
    f = tmp_path / "clean.txt"
    f.write_text("Nothing to see here.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 0


def test_check_staged_marked_file_exits_1(tmp_path, monkeypatch, capsys):
    f = tmp_path / "marked.txt"
    f.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 1
    err = capsys.readouterr().err
    assert str(f) in err
    assert "layer-a" in err


def test_check_staged_multiple_files_one_marked(tmp_path, monkeypatch, capsys):
    clean = tmp_path / "clean.txt"
    clean.write_text("plain text", encoding="utf-8")
    marked = tmp_path / "marked.txt"
    marked.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(clean), str(marked)])
    assert check_staged.main() == 1
    err = capsys.readouterr().err
    assert str(marked) in err
    assert str(clean) not in err


def test_check_staged_unknown_format_skipped(tmp_path, monkeypatch):
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe no known magic bytes here")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 0


def test_check_staged_missing_path_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(tmp_path / "nope.txt")])
    assert check_staged.main() == 2


def test_clean_staged_marked_file_cleans_and_exits_1(tmp_path, monkeypatch, capsys):
    f = tmp_path / "marked.txt"
    f.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    assert f.read_text(encoding="utf-8") == "HelloWorld!"
    err = capsys.readouterr().err
    assert str(f) in err


def test_clean_staged_already_clean_file_exits_0_unchanged(tmp_path, monkeypatch):
    f = tmp_path / "clean.txt"
    original = "Nothing to see here."
    f.write_text(original, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_text(encoding="utf-8") == original


def test_clean_staged_unknown_format_skipped(tmp_path, monkeypatch):
    f = tmp_path / "data.bin"
    original = b"\x00\x01\x02\xff\xfe no known magic bytes here"
    f.write_bytes(original)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_bytes() == original


def test_pre_commit_hooks_manifest_defines_both_hooks():
    # No PyYAML in this project's stdlib-only test deps (requirements-dev.txt) —
    # check the manifest's shape textually rather than adding a parser dependency.
    text = (ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")
    assert "id: watermarks-remover-check" in text
    assert "id: watermarks-remover-clean" in text
    assert text.count("entry: python3 service/scripts/") == 2
    assert text.count("language: system") == 2


def test_clean_staged_crashed_cleaner_blocks_commit(tmp_path, monkeypatch, capsys):
    """A crashed clean_file.py must stop the commit, not read as already-clean
    (#159). The issue's real trigger is a staged symlink (safe_write refuses,
    traceback, empty stdout, exit != 0|2)."""
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.txt"
    target.write_text(_watermarked_text(), encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(link)])
    assert clean_staged.main() == 1
    assert "failed" in capsys.readouterr().err.lower()


def test_clean_staged_crash_exit_code_blocks_commit(tmp_path, monkeypatch, capsys):
    """Directly pin the exit-code contract: any nonzero, non-2 cleaner exit
    (crash, kill) yields exit 1 from the hook."""
    from types import SimpleNamespace

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="Traceback (most recent call last): ...")

    monkeypatch.setattr(clean_staged.subprocess, "run", fake_run)
    f = tmp_path / "x.txt"
    f.write_text("plain", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    err = capsys.readouterr().err
    assert "clean_file.py failed (exit 1)" in err


def test_clean_staged_garbage_json_blocks_commit(tmp_path, monkeypatch, capsys):
    """Unparseable cleaner output is a failure, not a skip (#159)."""
    from types import SimpleNamespace

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not json at all", stderr="")

    monkeypatch.setattr(clean_staged.subprocess, "run", fake_run)
    f = tmp_path / "y.txt"
    f.write_text("plain", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    assert "could not parse" in capsys.readouterr().err
