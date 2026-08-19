#!/usr/bin/env python3
"""Reproducibility pin helpers (gap 05-A7).

Records, at run time, the exact versions every result depends on so the
released JSONL can be reproduced (research/01-experiment-protocol.md §7):

  - the MarkLLM checkout commit (git rev-parse HEAD of --markllm-dir)
  - the watermarks-remover repo commit
  - HF hub revisions for every model used (opt-1.3b, Qwen2.5-1.5B-Instruct,
    gpt2-large, deberta-xlarge-mnli, all-MiniLM-L6-v2)
  - pip freeze of the active interpreter's environment

Every helper fails soft: a missing git repo, hub, or pip returns None and
the caller records the gap rather than aborting the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_commit(dir_path: Path | str | None, *, short: bool = True) -> str | None:
    if not dir_path:
        return None
    try:
        # S607: "git" is resolved from PATH by design (repo/MarkLLM checkouts
        # are not guaranteed to be absolute); args are a fixed literal list.
        r = subprocess.run(
            ["git", "-C", str(dir_path), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode != 0:
            return None
        head = r.stdout.strip()
        return head[:12] if short and head else head or None
    except (OSError, subprocess.SubprocessError):
        return None


def repo_commit(*, short: bool = True) -> str | None:
    """Commit of the watermarks-remover repo containing this file."""
    here = Path(__file__).resolve()
    # research/scripts/pins.py -> repo root is three levels up.
    root = here.parents[2]
    return _git_commit(root, short=short)


def markllm_commit(upstream: str | Path | None, *, short: bool = True) -> str | None:
    """Commit of the MarkLLM checkout (--markllm-dir)."""
    return _git_commit(upstream, short=short)


def hf_revision(model_id: str) -> str | None:
    """HF hub revision (commit sha) for *model_id*, or None if unavailable.

    Uses huggingface_hub when installed (it is, in the MarkLLM env); never
    raises -- offline runs simply record None.
    """
    if os.environ.get("HF_HUB_OFFLINE"):
        return None
    try:
        from huggingface_hub import model_info  # type: ignore[import-not-found]

        info = model_info(model_id)
        return info.sha or None
    except Exception:
        return None


def hf_revisions(model_ids: list[str]) -> dict[str, str | None]:
    return {mid: hf_revision(mid) for mid in model_ids}


def pip_freeze() -> list[str] | None:
    """pip freeze of the active environment (MarkLLM or quality env)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if r.returncode != 0:
            return None
        return sorted(line for line in r.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def env_summary(
    upstream: str | Path | None = None,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    """One dict of every pin the manifest needs (gap 05-A7)."""
    model_ids = model_ids or []
    return {
        "repo_commit": repo_commit(),
        "markllm_commit": markllm_commit(upstream),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hf_revisions": hf_revisions(model_ids),
        "pip_freeze": pip_freeze(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(env_summary(), indent=2))
