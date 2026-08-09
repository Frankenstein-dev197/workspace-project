"""Tests for the tool system."""

import pytest

from daemon_engine.tools.automation_tools import AutomationTools
from daemon_engine.tools.browser_tools import BrowserTools
from daemon_engine.tools.devops_tools import DevOpsTools
from daemon_engine.tools.research_tools import ResearchTools
from daemon_engine.tools.tool_registry import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()
    BrowserTools().register_all(reg)
    ResearchTools().register_all(reg)
    DevOpsTools().register_all(reg)
    AutomationTools().register_all(reg)
    return reg


class TestToolRegistry:
    def test_list_tools(self, registry):
        tools = registry.list_tools()
        assert "web_search" in tools
        assert "bash" in tools
        assert "web_scraper" in tools

    def test_list_by_category(self, registry):
        browser_tools = registry.list_by_category("browser")
        assert "web_search" in browser_tools
        assert "web_fetch" in browser_tools

    def test_get_descriptions(self, registry):
        descs = registry.get_descriptions()
        assert "web_search" in descs
        assert isinstance(descs["web_search"], str)

    def test_execute_unknown_tool(self, registry):
        result = registry.execute("nonexistent_tool")
        assert result.success is False
        assert "not found" in result.error

    def test_dangerous_pattern_blocked(self, registry):
        result = registry.execute("bash", command="rm -rf /")
        assert result.success is False
        assert "dangerous" in result.error.lower()

    def test_unregister(self, registry):
        assert registry.unregister("web_search") is True
        assert "web_search" not in registry.list_tools()


class TestBrowserTools:
    def test_web_search(self, registry):
        result = registry.execute("web_search", query="python tutorials")
        assert result.success is True
        assert "python" in result.output.lower()

    def test_extract_links(self, registry):
        result = registry.execute("extract_links", url="https://example.com")
        # May fail if network unavailable, but should handle gracefully
        assert isinstance(result.success, bool)

    def test_browser_navigate(self, registry):
        result = registry.execute("browser_navigate", url="https://example.com")
        assert result.success is True


class TestResearchTools:
    def test_osint_lookup(self, registry):
        result = registry.execute("osint_lookup", username="testuser")
        assert result.success is True
        assert "testuser" in result.output

    def test_data_extract(self, registry):
        result = registry.execute("data_extract", text="Contact: test@example.com or visit https://example.com")
        assert result.success is True
        assert "test@example.com" in result.data["emails"]
        assert "https://example.com" in result.data["urls"]

    def test_fact_check(self, registry):
        result = registry.execute("fact_check", claim="The sky is blue")
        assert result.success is True
        assert result.data["claim"] == "The sky is blue"


class TestDevOpsTools:
    def test_execute_command(self, registry):
        result = registry.execute("bash", command="echo 'hello world'")
        assert result.success is True
        assert "hello world" in result.output

    def test_file_write_read(self, registry):
        registry.execute("file_write", path="test_file.txt", content="test content")
        result = registry.execute("file_read", path="test_file.txt")
        assert result.success is True
        assert "test content" in result.output

    def test_file_list(self, registry):
        registry.execute("file_write", path="list_test.txt", content="x")
        result = registry.execute("file_list", path=".")
        assert result.success is True

    def test_dangerous_command_blocked(self, registry):
        result = registry.execute("bash", command="sudo rm -rf /")
        assert result.success is False


class TestAutomationTools:
    def test_schedule_task(self, registry):
        result = registry.execute("schedule_task", name="daily_report", command="echo report")
        assert result.success is True
        assert "daily_report" in result.output

    def test_run_pipeline(self, registry):
        steps = [{"name": "step1", "command": "echo hello"}, {"name": "step2", "command": "echo world"}]
        result = registry.execute("run_pipeline", steps=steps)
        assert result.success is True
        assert result.data["steps_completed"] == 2

    def test_deploy_application(self, registry):
        result = registry.execute("deploy_application", project_path=".")
        assert result.success is True

    def test_git_operations(self, registry):
        result = registry.execute("git_operations", operation="status", path=".")
        assert isinstance(result.success, bool)
