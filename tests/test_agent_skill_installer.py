from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_NAME = "remove-ai-marks"


def _run_installer(home: Path, *args: str, check: bool = True):
    env = os.environ.copy()
    for variable in ("CLAUDE_HOME", "CODEX_HOME", "CURSOR_HOME"):
        env.pop(variable, None)
    env.update({"HOME": str(home), "USERPROFILE": str(home)})
    return subprocess.run(
        [sys.executable, str(ROOT / "install_skill.py"), *args],
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def test_installs_remove_ai_marks_for_codex(tmp_path):
    codex_home = tmp_path / "codex-home"

    _run_installer(
        tmp_path,
        "--target",
        "codex",
        "--skill",
        SKILL_NAME,
        "--codex-home",
        str(codex_home),
    )

    installed = codex_home / "skills" / SKILL_NAME
    assert (installed / "SKILL.md").is_file()
    assert (installed / "agents" / "openai.yaml").is_file()


def test_installs_remove_ai_marks_for_claude(tmp_path):
    claude_home = tmp_path / "claude-home"

    _run_installer(
        tmp_path,
        "--target",
        "claude",
        "--skill",
        SKILL_NAME,
        "--claude-home",
        str(claude_home),
    )

    installed = claude_home / "skills" / SKILL_NAME
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "ethics.md").is_file()


def test_target_all_installs_every_agent_skill(tmp_path):
    _run_installer(tmp_path, "--target", "all", "--skill", SKILL_NAME)

    for agent_home in (".codex", ".claude", ".cursor"):
        installed = tmp_path / agent_home / "skills" / SKILL_NAME
        assert (installed / "SKILL.md").is_file()


def test_codex_metadata_matches_remove_ai_marks_skill():
    metadata = ROOT / "skills" / SKILL_NAME / "agents" / "openai.yaml"

    assert metadata.is_file()
    text = metadata.read_text(encoding="utf-8")
    assert 'display_name: "Remove AI Marks"' in text
    assert "$remove-ai-marks" in text


def test_remove_ai_marks_skill_uses_portable_base64_commands():
    skill_text = (ROOT / "skills" / SKILL_NAME / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "base64 -w0" not in skill_text
    assert "tr -d '\\r\\n'" in skill_text
