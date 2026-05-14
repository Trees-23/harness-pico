"""Lightweight git-backed helpers for Dream memory maintenance."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LineAge:
    age_days: int


class GitStore:
    """Small subset of Nanobot's GitStore used by Pico Dream."""

    def __init__(self, workspace: Path, tracked_files: list[str] | None = None):
        self._workspace = Path(workspace)
        self._tracked_files = list(tracked_files or [])

    def is_initialized(self) -> bool:
        return (self._workspace / ".git").exists()

    def line_ages(self, file_path: str) -> list[LineAge]:
        if not self.is_initialized():
            return []
        target = self._workspace / file_path
        if not target.exists() or target.stat().st_size == 0:
            return []

        try:
            blame = subprocess.run(
                ["git", "blame", "--line-porcelain", "--", file_path],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return []

        now = datetime.now(tz=timezone.utc).date()
        ages: list[LineAge] = []
        current_author_time: int | None = None
        for raw_line in blame.stdout.splitlines():
            line = raw_line.rstrip("\n")
            if line.startswith("author-time "):
                try:
                    current_author_time = int(line.split(" ", 1)[1].strip())
                except Exception:
                    current_author_time = None
                continue
            if not line.startswith("\t"):
                continue
            if current_author_time is None:
                continue
            committed = datetime.fromtimestamp(current_author_time, tz=timezone.utc).date()
            ages.append(LineAge(age_days=(now - committed).days))
        return ages
