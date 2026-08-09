"""Tests for worktree isolation."""

import json
import subprocess
from pathlib import Path

import pytest

from daemon_engine.runtime.worktree import (
    WorktreeManager,
    WorktreeInfo,
    validate_worktree_name,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo for testing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    (repo / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True)
    return repo


class TestValidateWorktreeName:
    def test_valid_name(self):
        assert validate_worktree_name("feature-auth") is None
        assert validate_worktree_name("ui_work") is None
        assert validate_worktree_name("task.123") is None

    def test_empty(self):
        assert validate_worktree_name("") is not None

    def test_dot(self):
        assert validate_worktree_name(".") is not None

    def test_double_dot(self):
        assert validate_worktree_name("..") is not None

    def test_path_traversal(self):
        assert validate_worktree_name("../etc/passwd") is not None
        assert validate_worktree_name("foo/bar") is not None
        assert validate_worktree_name("foo\\bar") is not None

    def test_invalid_chars(self):
        assert validate_worktree_name("test name") is not None
        assert validate_worktree_name("test:name") is not None
        assert validate_worktree_name("test*name") is not None

    def test_too_long(self):
        assert validate_worktree_name("a" * 65) is not None

    def test_max_length(self):
        assert validate_worktree_name("a" * 64) is None


class TestWorktreeManager:
    def test_creation(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        assert len(manager.list_worktrees()) == 0

    def test_create_worktree(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        result = manager.create_worktree("feature-auth")
        assert "created" in result.lower()
        assert len(manager.list_worktrees()) == 1

    def test_create_invalid_name(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        result = manager.create_worktree("../bad")
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_create_duplicate(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("feature")
        result = manager.create_worktree("feature")
        assert "already exists" in result

    def test_remove_worktree(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("temp")
        result = manager.remove_worktree("temp")
        assert "removed" in result.lower()
        assert len(manager.list_worktrees()) == 0

    def test_remove_nonexistent(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        result = manager.remove_worktree("nonexistent")
        assert "not found" in result

    def test_remove_with_uncommitted_changes(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("dirty")
        wt_path = manager.get_worktree_path("dirty")
        (wt_path / "new_file.txt").write_text("uncommitted")
        result = manager.remove_worktree("dirty")
        assert "uncommitted" in result.lower() or "force" in result.lower()

    def test_remove_with_discard(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("dirty")
        wt_path = manager.get_worktree_path("dirty")
        (wt_path / "new_file.txt").write_text("uncommitted")
        result = manager.remove_worktree("dirty", discard_changes=True)
        assert "removed" in result.lower()

    def test_keep_worktree(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("keep")
        result = manager.keep_worktree("keep")
        assert "kept" in result.lower()

    def test_keep_nonexistent(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        result = manager.keep_worktree("nonexistent")
        assert "not found" in result

    def test_list_worktrees(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("wt1")
        manager.create_worktree("wt2")
        wts = manager.list_worktrees()
        assert len(wts) == 2
        names = [w.name for w in wts]
        assert "wt1" in names
        assert "wt2" in names

    def test_get_worktree(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("feature")
        info = manager.get_worktree("feature")
        assert info is not None
        assert info.name == "feature"
        assert info.branch == "wt/feature"

    def test_get_worktree_nonexistent(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        assert manager.get_worktree("nonexistent") is None

    def test_get_worktree_path(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("feature")
        path = manager.get_worktree_path("feature")
        assert path is not None
        assert path.exists()

    def test_get_worktree_path_nonexistent(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        assert manager.get_worktree_path("nonexistent") is None

    def test_events_logged(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("logged")
        events = manager.get_events()
        assert len(events) >= 1
        assert events[0]["type"] == "create"
        assert events[0]["worktree"] == "logged"

    def test_remove_event_logged(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("temp")
        manager.remove_worktree("temp")
        events = manager.get_events()
        types = [e["type"] for e in events]
        assert "create" in types
        assert "remove" in types

    def test_stats(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("wt1")
        manager.create_worktree("wt2")
        stats = manager.stats()
        assert stats["total_created"] == 2
        assert stats["active_worktrees"] == 2

    def test_cleanup_all(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("wt1")
        manager.create_worktree("wt2")
        results = manager.cleanup_all()
        assert len(results) == 2
        assert len(manager.list_worktrees()) == 0

    def test_worktree_info_to_dict(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        manager.create_worktree("feature")
        info = manager.get_worktree("feature")
        d = info.to_dict()
        assert d["name"] == "feature"
        assert d["branch"] == "wt/feature"

    def test_with_task_binding(self, git_repo):
        manager = WorktreeManager(repo_root=git_repo)
        result = manager.create_worktree("task-wt", task_id="task_123")
        assert "created" in result.lower()
        events = manager.get_events()
        assert any(e.get("task_id") == "task_123" for e in events)

    def test_custom_worktrees_dir(self, git_repo, tmp_path):
        custom_dir = tmp_path / "custom_wts"
        manager = WorktreeManager(
            repo_root=git_repo,
            worktrees_dir=custom_dir,
        )
        manager.create_worktree("custom")
        assert (custom_dir / "custom").exists()
