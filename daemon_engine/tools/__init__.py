"""Tool system: external tools for agents to act on the world.

Integrates tool patterns from Browser-Use (browser automation), Scrapy/
Scrapling (web scraping), Sherlock (OSINT), and Ansible (DevOps automation).
Provides a unified tool registry with browser, research, scraping, devops, and
automation tool categories.
"""

from daemon_engine.tools.tool_registry import ToolRegistry, ToolResult, ToolDefinition
from daemon_engine.tools.browser_tools import BrowserTools
from daemon_engine.tools.research_tools import ResearchTools
from daemon_engine.tools.scraping_tools import ScrapingTools, Spider, Selector, ScrapedItem
from daemon_engine.tools.devops_tools import DevOpsTools
from daemon_engine.tools.automation_tools import AutomationTools
from daemon_engine.tools.auto_throttle import AutoThrottle, parse_retry_after, DomainStats

__all__ = [
    "ToolRegistry",
    "ToolResult",
    "ToolDefinition",
    "BrowserTools",
    "ResearchTools",
    "ScrapingTools",
    "Spider",
    "Selector",
    "ScrapedItem",
    "DevOpsTools",
    "AutomationTools",
    "AutoThrottle",
    "parse_retry_after",
    "DomainStats",
]
