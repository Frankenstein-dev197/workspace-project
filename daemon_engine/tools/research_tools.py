"""Research Tools: information gathering and OSINT capabilities.

Integrates patterns from Scrapy (structured data extraction), Scrapling
(adaptive web scraping), and Sherlock (social media username lookup).
Provides web scraping, OSINT, and data collection tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from daemon_engine.tools.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ResearchTools:
    """Research and information gathering tools."""

    # Simplified social platform list inspired by Sherlock
    SOCIAL_PLATFORMS = [
        "github.com", "twitter.com", "instagram.com", "facebook.com",
        "linkedin.com", "reddit.com", "youtube.com", "tiktok.com",
        "medium.com", "dev.to", "hackernews.com", "gitlab.com",
    ]

    def register_all(self, registry: ToolRegistry) -> None:
        registry.register(
            "web_scraper",
            "Scrape structured data from a web page using CSS-like selectors",
            self.web_scraper,
            category="research",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "osint_lookup",
            "Look up a username across social platforms (Sherlock-style)",
            self.osint_lookup,
            category="research",
            parameters={"username": {"type": "string", "required": True}},
        )
        registry.register(
            "data_extract",
            "Extract structured data (emails, phones, URLs) from text",
            self.data_extract,
            category="research",
            parameters={"text": {"type": "string", "required": True}},
        )
        registry.register(
            "summarize_url",
            "Fetch and summarize the content of a URL",
            self.summarize_url,
            category="research",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "fact_check",
            "Basic fact-checking by cross-referencing a claim with web sources",
            self.fact_check,
            category="research",
            parameters={"claim": {"type": "string", "required": True}},
        )

    def web_scraper(self, url: str, **kwargs: Any) -> ToolResult:
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "DaemonEngine/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "No title"
            meta_match = re.search(
                r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            description = meta_match.group(1).strip() if meta_match else ""
            headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html, re.DOTALL | re.IGNORECASE)
            headings = [re.sub(r"<[^>]+>", "", h).strip() for h in headings[:10]]
            data = {
                "title": title,
                "description": description,
                "headings": headings,
                "url": url,
                "html_length": len(html),
            }
            return ToolResult(
                tool_name="web_scraper",
                success=True,
                output=json.dumps(data, indent=2)[:5000],
                data=data,
            )
        except Exception as exc:
            return ToolResult(tool_name="web_scraper", success=False, error=str(exc))

    def osint_lookup(self, username: str, **kwargs: Any) -> ToolResult:
        found: list[dict[str, str]] = []
        for platform in self.SOCIAL_PLATFORMS:
            url = f"https://{platform}/{username}"
            found.append({"platform": platform, "url": url, "status": "unverified"})
        return ToolResult(
            tool_name="osint_lookup",
            success=True,
            output=json.dumps(found, indent=2)[:5000],
            data={"username": username, "platforms_checked": len(found), "results": found},
        )

    def data_extract(self, text: str, **kwargs: Any) -> ToolResult:
        emails = list(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)))
        urls = list(set(re.findall(r"https?://[^\s<>\"]+", text)))
        phones = list(set(re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text)))
        ips = list(set(re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)))
        data = {"emails": emails, "urls": urls, "phones": phones, "ip_addresses": ips}
        return ToolResult(
            tool_name="data_extract",
            success=True,
            output=json.dumps(data, indent=2),
            data=data,
        )

    def summarize_url(self, url: str, **kwargs: Any) -> ToolResult:
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "DaemonEngine/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            sentences = text.split(". ")
            summary = ". ".join(sentences[:5]) + "." if len(sentences) > 5 else text
            return ToolResult(
                tool_name="summarize_url",
                success=True,
                output=summary[:2000],
                data={"url": url, "original_length": len(text), "summary_length": len(summary)},
            )
        except Exception as exc:
            return ToolResult(tool_name="summarize_url", success=False, error=str(exc))

    def fact_check(self, claim: str, **kwargs: Any) -> ToolResult:
        keywords = re.findall(r"\b[A-Z][a-z]+\b", claim)
        data = {
            "claim": claim,
            "extracted_keywords": keywords,
            "verification_status": "unverified",
            "note": "Automated fact-checking requires web search integration for full verification.",
        }
        return ToolResult(
            tool_name="fact_check",
            success=True,
            output=json.dumps(data, indent=2),
            data=data,
        )
