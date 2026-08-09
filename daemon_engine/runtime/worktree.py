"""Worktree isolation: git worktree management for parallel agent work.

Integrates learn-claude-code s18 worktree isolation pattern:
- WorktreeManager: manages git worktrees for isolated agent work
- validate_worktree_name: reject path traversal and illegal chars
- create_worktree: git worktree add with dedicated branch
- bind_task_to_worktree: associate task with worktree
- remove_worktree: safety check before removal (uncommitted changes)
- keep_worktree: preserve for manual review
- Event logging: lifecycle events to JSONL file

This enables multiple agents to work in isolated worktrees, each with
its own branch, preventing conflicts during parallel development.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_WT_NAME = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")


def validate_worktree_name(name: str) -> str | None:
    """Return error message if invalid, None if valid."""
    if not name:
        return "Worktree name cannot be empty"
    if name == "." or name == "..":
        return f"'{name}' is not a valid worktree name"
    if "/" in name or "\\" in name:
        return f"Invalid worktree name: path separators not allowed"
    if not VALID_WT_NAME.match(name):
        return (
            f"Invalid worktree name '{name}': "
            "only letters, digits, dots, underscores, dashes (1-64 chars)"
        )
    return None


@dataclass
class WorktreeInfo:
    """Information about a git worktree."""
    name: str
    path: str
    branch: str
    task_id: str | None = None
    created_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "branch": self.branch,
            "task_id": self.task_id,
            "created_at": self.created_at,
        }


class WorktreeManager:
    """Manages git worktrees for isolated agent work.

    Each worktree gets a dedicated branch (wt/<name>) and can be
    optionally bound to a task. Safety checks prevent removing
    worktrees with uncommitted changes unless forced.
    """

    def __init__(
        self,
        repo_root: str | Path,
        worktrees_dir: str | Path | None = None,
    ) -> None:
        self._repo_root = Path(repo_root)
        self._worktrees_dir = Path(worktrees_dir) if worktrees_dir else self._repo_root / ".worktrees"
        self._worktrees_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = self._worktrees_dir / "events.jsonl"
        self._stats = {
            "total_created": 0,
            "total_removed": 0,
            "total_kept": 0,
            "total_failed": 0,
        }

    def _run_git(self, args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
        """Run git command. Return (ok, output)."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd or self._repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = (result.stdout + result.stderr).strip()
            out = out[:5000] if out else "(no output)"
            return result.returncode == 0, out
        except subprocess.TimeoutExpired:
            return False, "Error: git timeout"
        except FileNotFoundError:
            return False, "Error: git not found"
        except Exception as exc:
            return False, f"Error: {exc}"

    def _log_event(
        self,
        event_type: str,
        worktree_name: str,
        task_id: str = "",
    ) -> None:
        """Append a lifecycle event to events.jsonl."""
        event = {
            "type": event_type,
            "worktree": worktree_name,
            "task_id": task_id,
            "timestamp": time.time(),
        }
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            logger.error("Failed to log event: %s", exc)

    def _get_events(self, worktree_name: str) -> list[dict[str, Any]]:
        """Get events for a worktree."""
        if not self._events_file.exists():
            return []
        events = []
        try:
            for line in self._events_file.read_text().splitlines():
                if line.strip():
                    event = json.loads(line)
                    if event.get("worktree") == worktree_name:
                        events.append(event)
        except Exception:
            pass
        return events

    def create_worktree(
        self,
        name: str,
        task_id: str = "",
        base_branch: str = "HEAD",
    ) -> str:
        """Create a git worktree with a dedicated branch.

        Optionally bind to a task. Returns success message or error.
        """
        err = validate_worktree_name(name)
        if err:
            return f"Error: {err}"

        path = self._worktrees_dir / name
        if path.exists():
            return f"Worktree '{name}' already exists at {path}"

        branch = f"wt/{name}"
        ok, result = self._run_git(
            ["worktree", "add", str(path), "-b", branch, base_branch]
        )
        if not ok:
            self._stats["total_failed"] += 1
            return f"Git error: {result}"

        self._stats["total_created"] += 1
        self._log_event("create", name, task_id)
        logger.info("Worktree created: %s at %s", name, path)
        return f"Worktree '{name}' created at {path}"

    def remove_worktree(
        self,
        name: str,
        discard_changes: bool = False,
    ) -> str:
        """Remove worktree. Refuses if uncommitted changes unless discard."""
        err = validate_worktree_name(name)
        if err:
            return err

        path = self._worktrees_dir / name
        if not path.exists():
            return f"Worktree '{name}' not found"

        if not discard_changes:
            files, commits = self._count_changes(path)
            if files < 0:
                return (
                    f"Cannot verify worktree '{name}' status. "
                    "Use discard_changes=true to force removal."
                )
            if files > 0 or commits > 0:
                return (
                    f"Worktree '{name}' has {files} uncommitted file(s) "
                    f"and {commits} unpushed commit(s). "
                    "Use discard_changes=true to force removal, "
                    "or keep_worktree to preserve for review."
                )

        ok1, _ = self._run_git(["worktree", "remove", str(path), "--force"])
        if not ok1:
            return f"Failed to remove worktree directory for '{name}'"

        self._run_git(["branch", "-D", f"wt/{name}"])
        self._stats["total_removed"] += 1
        self._log_event("remove", name)
        logger.info("Worktree removed: %s", name)
        return f"Worktree '{name}' removed"

    def keep_worktree(self, name: str) -> str:
        """Keep worktree for manual review. Branch preserved."""
        err = validate_worktree_name(name)
        if err:
            return err

        path = self._worktrees_dir / name
        if not path.exists():
            return f"Worktree '{name}' not found"

        self._stats["total_kept"] += 1
        self._log_event("keep", name)
        logger.info("Worktree kept for review: %s", name)
        return f"Worktree '{name}' kept for review. Branch 'wt/{name}' preserved."

    def _count_changes(self, path: Path) -> tuple[int, int]:
        """Count uncommitted files and unpushed commits in a worktree."""
        try:
            r1 = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            files = len([l for l in r1.stdout.strip().splitlines() if l.strip()])
            r2 = subprocess.run(
                ["git", "log", "@{push}..HEAD", "--oneline"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            commits = len([l for l in r2.stdout.strip().splitlines() if l.strip()])
            return files, commits
        except Exception:
            return -1, -1

    def list_worktrees(self) -> list[WorktreeInfo]:
        """List all managed worktrees."""
        result: list[WorktreeInfo] = []
        if not self._worktrees_dir.exists():
            return result
        for path in sorted(self._worktrees_dir.iterdir()):
            if path.is_dir() and path.name != "events.jsonl":
                info = WorktreeInfo(
                    name=path.name,
                    path=str(path),
                    branch=f"wt/{path.name}",
                    events=self._get_events(path.name),
                )
                result.append(info)
        return result

    def get_worktree(self, name: str) -> WorktreeInfo | None:
        """Get info for a specific worktree."""
        path = self._worktrees_dir / name
        if not path.exists():
            return None
        return WorktreeInfo(
            name=name,
            path=str(path),
            branch=f"wt/{name}",
            events=self._get_events(name),
        )

    def get_worktree_path(self, name: str) -> Path | None:
        """Get the path for a worktree."""
        path = self._worktrees_dir / name
        return path if path.exists() else None

    def get_events(self) -> list[dict[str, Any]]:
        """Get all events from the events log."""
        if not self._events_file.exists():
            return []
        events = []
        try:
            for line in self._events_file.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))
        except Exception:
            pass
        return events

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "active_worktrees": len(self.list_worktrees()),
            "worktrees_dir": str(self._worktrees_dir),
        }

    def cleanup_all(self, discard_changes: bool = True) -> list[str]:
        """Remove all worktrees. Returns list of results."""
        results = []
        for wt in self.list_worktrees():
            result = self.remove_worktree(wt.name, discard_changes=discard_changes)
            results.append(result)
        return results
