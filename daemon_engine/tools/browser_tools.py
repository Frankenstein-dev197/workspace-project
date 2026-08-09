"""Browser Tools: web browsing and automation capabilities.

Integrates patterns from Browser-Use (AI-driven browser automation),
Puppeteer (headless browser control), and Scrapling (adaptive parsing).
Provides page fetching, content extraction, and form interaction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from daemon_engine.tools.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class BrowserTools:
    """Browser automation tools inspired by Browser-Use and Puppeteer."""

    def __init__(self) -> None:
        self._session_pages: dict[str, str] = {}

    def register_all(self, registry: ToolRegistry) -> None:
        registry.register(
            "web_fetch",
            "Fetch the content of a web page URL",
            self.web_fetch,
            category="browser",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "web_search",
            "Search the web for a query and return results",
            self.web_search,
            category="browser",
            parameters={"query": {"type": "string", "required": True}},
        )
        registry.register(
            "extract_links",
            "Extract all hyperlinks from a web page",
            self.extract_links,
            category="browser",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "extract_text",
            "Extract clean text content from a web page",
            self.extract_text,
            category="browser",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "browser_navigate",
            "Navigate to a URL in a browser session (simulated)",
            self.browser_navigate,
            category="browser",
            parameters={"url": {"type": "string", "required": True}},
            is_safe=False,
        )

    def web_fetch(self, url: str, **kwargs: Any) -> ToolResult:
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "DaemonEngine/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            return ToolResult(
                tool_name="web_fetch",
                success=True,
                output=content[:10000],
                data={"url": url, "status": resp.status, "length": len(content)},
            )
        except Exception as exc:
            return ToolResult(tool_name="web_fetch", success=False, error=str(exc))

    def web_search(self, query: str, **kwargs: Any) -> ToolResult:
        try:
            results = self._mock_search(query)
            return ToolResult(
                tool_name="web_search",
                success=True,
                output=results,
                data={"query": query, "num_results": len(results.split("\n"))},
            )
        except Exception as exc:
            return ToolResult(tool_name="web_search", success=False, error=str(exc))

    def _mock_search(self, query: str) -> str:
        return (
            f"Search results for '{query}':\n"
            f"1. {query} - Overview and Documentation\n"
            f"2. {query} - Tutorial and Examples\n"
            f"3. {query} - Best Practices\n"
            f"4. {query} - Community Discussion\n"
            f"5. {query} - Latest News\n"
        )

    def extract_links(self, url: str, **kwargs: Any) -> ToolResult:
        fetch_result = self.web_fetch(url)
        if not fetch_result.success:
            return fetch_result
        links = re.findall(r'href=["\']([^"\']+)["\']', fetch_result.output)
        unique_links = list(dict.fromkeys(links))
        return ToolResult(
            tool_name="extract_links",
            success=True,
            output="\n".join(unique_links[:50]),
            data={"url": url, "num_links": len(unique_links)},
        )

    def extract_text(self, url: str, **kwargs: Any) -> ToolResult:
        fetch_result = self.web_fetch(url)
        if not fetch_result.success:
            return fetch_result
        text = re.sub(r"<script[^>]*>.*?</script>", "", fetch_result.output, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return ToolResult(
            tool_name="extract_text",
            success=True,
            output=text[:5000],
            data={"url": url, "text_length": len(text)},
        )

    def browser_navigate(self, url: str, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "default")
        self._session_pages[session_id] = url
        return ToolResult(
            tool_name="browser_navigate",
            success=True,
            output=f"Navigated to {url} in session {session_id}",
            data={"url": url, "session_id": session_id},
        )
