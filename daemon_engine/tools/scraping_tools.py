"""Scraping tools: advanced web scraping inspired by Scrapy and Scrapling.

Integrates:
- Scrapy's Spider pattern (start_urls, parse, callback pipeline)
- Scrapling's Selector (CSS/XPath extraction with lxml-style API)
- Scrapling's adaptive HTML parsing
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

from daemon_engine.tools.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class Selector:
    """HTML element selector inspired by Scrapling's Selector class.

    Provides CSS-like and XPath-like element extraction using regex patterns
    (lightweight, no lxml dependency). For production use, this can be
    upgraded to use lxml directly.
    """

    def __init__(self, html: str, url: str = "") -> None:
        self.html = html
        self.url = url
        self._text = self._strip_tags(html)

    def _strip_tags(self, html: str) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def css(self, selector: str) -> list[str]:
        if selector.startswith("a"):
            return self._extract_tag_content("a")
        if selector.startswith("p"):
            return self._extract_tag_content("p")
        if selector.startswith("h"):
            for level in range(1, 7):
                if selector.startswith(f"h{level}"):
                    return self._extract_tag_content(f"h{level}")
        if selector.startswith("div"):
            return self._extract_tag_content("div")
        if selector.startswith("img"):
            return self._extract_attr("img", "src")
        if selector.startswith("meta"):
            return self._extract_attr("meta", "content")
        if selector.startswith("title"):
            matches = re.findall(r"<title[^>]*>(.*?)</title>", self.html, re.DOTALL | re.IGNORECASE)
            return [m.strip() for m in matches]
        return []

    def xpath(self, expr: str) -> list[str]:
        if "//text()" in expr:
            return [self._text]
        if "//a" in expr or "//@href" in expr:
            return self._extract_attr("a", "href")
        if "//img" in expr or "//@src" in expr:
            return self._extract_attr("img", "src")
        tag_match = re.search(r"//(\w+)", expr)
        if tag_match:
            return self._extract_tag_content(tag_match.group(1))
        return []

    def text(self) -> str:
        return self._text

    def title(self) -> str:
        matches = re.findall(r"<title[^>]*>(.*?)</title>", self.html, re.DOTALL | re.IGNORECASE)
        return matches[0].strip() if matches else ""

    def links(self) -> list[str]:
        return self._extract_attr("a", "href")

    def images(self) -> list[str]:
        return self._extract_attr("img", "src")

    def meta_description(self) -> str:
        metas = re.findall(
            r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            self.html, re.IGNORECASE,
        )
        if not metas:
            metas = re.findall(
                r'<meta\s+[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']',
                self.html, re.IGNORECASE,
            )
        return metas[0].strip() if metas else ""

    def headings(self, max_level: int = 3) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for level in range(1, max_level + 1):
            result[f"h{level}"] = self._extract_tag_content(f"h{level}")
        return result

    def tables(self) -> list[list[list[str]]]:
        tables: list[list[list[str]]] = []
        table_matches = re.findall(r"<table[^>]*>(.*?)</table>", self.html, re.DOTALL | re.IGNORECASE)
        for table_html in table_matches:
            rows: list[list[str]] = []
            row_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
            for row_html in row_matches:
                cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL | re.IGNORECASE)
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables

    def _extract_tag_content(self, tag: str) -> list[str]:
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        matches = re.findall(pattern, self.html, re.DOTALL | re.IGNORECASE)
        return [re.sub(r"<[^>]+>", "", m).strip() for m in matches if m.strip()]

    def _extract_attr(self, tag: str, attr: str) -> list[str]:
        pattern = rf"<{tag}[^>]*{attr}=[\"']([^\"']+)[\"'][^>]*>"
        matches = re.findall(pattern, self.html, re.IGNORECASE)
        if self.url:
            return [urljoin(self.url, m) for m in matches]
        return matches


@dataclass
class ScrapedItem:
    """A single scraped data item (inspired by Scrapy's Item)."""
    url: str
    title: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    raw_html: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "data": self.data}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class Spider:
    """Web spider inspired by Scrapy's Spider class.

    Subclass and override parse() to extract structured data from pages.
    """

    name: str = "base-spider"
    start_urls: list[str] = []
    allowed_domains: list[str] = []

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        if name:
            self.name = name
        self.__dict__.update(kwargs)
        if not hasattr(self, "start_urls"):
            self.start_urls = []

    def start_requests(self) -> Iterator[tuple[str, str]]:
        for url in self.start_urls:
            yield url, "parse"

    def parse(self, selector: Selector, url: str) -> Iterator[ScrapedItem | tuple[str, str]]:
        raise NotImplementedError

    def follow(self, url: str, callback: str = "parse") -> tuple[str, str]:
        return url, callback

    def is_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        domain = urlparse(url).netloc
        return any(d in domain for d in self.allowed_domains)


class GenericSpider(Spider):
    """A generic spider that extracts common page data."""

    name = "generic"
    start_urls: list[str] = []

    def parse(self, selector: Selector, url: str) -> Iterator[ScrapedItem]:
        item = ScrapedItem(
            url=url,
            title=selector.title(),
            raw_html=selector.html[:5000],
            data={
                "description": selector.meta_description(),
                "headings": selector.headings(),
                "links": selector.links()[:20],
                "images": selector.images()[:10],
                "text": selector.text()[:2000],
                "tables": selector.tables(),
            },
        )
        yield item


class StructuredDataSpider(Spider):
    """Spider that extracts structured data (JSON-LD, microdata)."""

    name = "structured-data"

    def parse(self, selector: Selector, url: str) -> Iterator[ScrapedItem]:
        jsonld = self._extract_jsonld(selector.html)
        item = ScrapedItem(
            url=url,
            title=selector.title(),
            data={"json_ld": jsonld, "text": selector.text()[:1000]},
        )
        yield item

    def _extract_jsonld(self, html: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        blocks = re.findall(
            r'<script\s+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE,
        )
        for block in blocks:
            try:
                data = json.loads(block.strip())
                results.append(data)
            except json.JSONDecodeError:
                continue
        return results


class ScrapingTools:
    """Scraping tools registered with the tool registry.

    Combines Scrapy's spider pattern with Scrapling's adaptive parsing.
    """

    def __init__(self) -> None:
        self._spiders: dict[str, type[Spider]] = {
            "generic": GenericSpider,
            "structured": StructuredDataSpider,
        }

    def register_all(self, registry: ToolRegistry) -> None:
        registry.register(
            "scrape_page",
            "Scrape a web page and extract structured data (title, headings, links, text, tables)",
            self.scrape_page,
            category="scraping",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "scrape_with_selector",
            "Scrape a web page using CSS/XPath selectors to extract specific elements",
            self.scrape_with_selector,
            category="scraping",
            parameters={"url": {"type": "string", "required": True}, "selector": {"type": "string", "required": True}},
        )
        registry.register(
            "scrape_multiple",
            "Scrape multiple URLs and return aggregated results",
            self.scrape_multiple,
            category="scraping",
            parameters={"urls": {"type": "array", "required": True}},
        )
        registry.register(
            "extract_structured_data",
            "Extract JSON-LD structured data from a web page",
            self.extract_structured_data,
            category="scraping",
            parameters={"url": {"type": "string", "required": True}},
        )
        registry.register(
            "crawl_links",
            "Crawl a page and follow links to discover related pages",
            self.crawl_links,
            category="scraping",
            parameters={"url": {"type": "string", "required": True}, "max_depth": {"type": "int", "required": False}},
        )

    def _fetch_html(self, url: str) -> str:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "DaemonEngine/1.0 Scraper"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def scrape_page(self, url: str, **kwargs: Any) -> ToolResult:
        try:
            html = self._fetch_html(url)
            selector = Selector(html, url=url)
            spider = GenericSpider()
            items = list(spider.parse(selector, url))
            if items:
                item = items[0]
                return ToolResult(
                    tool_name="scrape_page",
                    success=True,
                    output=item.to_json()[:8000],
                    data=item.to_dict(),
                )
            return ToolResult(tool_name="scrape_page", success=False, error="No data extracted")
        except Exception as exc:
            return ToolResult(tool_name="scrape_page", success=False, error=str(exc))

    def scrape_with_selector(self, url: str, selector_expr: str = "", **kwargs: Any) -> ToolResult:
        try:
            html = self._fetch_html(url)
            selector = Selector(html, url=url)
            if selector_expr.startswith("//"):
                results = selector.xpath(selector_expr)
            else:
                results = selector.css(selector_expr)
            return ToolResult(
                tool_name="scrape_with_selector",
                success=True,
                output=json.dumps(results[:50], indent=2, ensure_ascii=False),
                data={"url": url, "selector": selector_expr, "results": results[:50]},
            )
        except Exception as exc:
            return ToolResult(tool_name="scrape_with_selector", success=False, error=str(exc))

    def scrape_multiple(self, urls: list[str] | None = None, **kwargs: Any) -> ToolResult:
        if not urls:
            return ToolResult(tool_name="scrape_multiple", success=False, error="No URLs provided")
        results: list[dict[str, Any]] = []
        for url in urls[:10]:
            try:
                html = self._fetch_html(url)
                selector = Selector(html, url=url)
                results.append({
                    "url": url,
                    "title": selector.title(),
                    "text": selector.text()[:500],
                    "success": True,
                })
            except Exception as exc:
                results.append({"url": url, "error": str(exc), "success": False})
        return ToolResult(
            tool_name="scrape_multiple",
            success=True,
            output=json.dumps(results, indent=2, ensure_ascii=False)[:8000],
            data={"urls_scraped": len(results), "results": results},
        )

    def extract_structured_data(self, url: str, **kwargs: Any) -> ToolResult:
        try:
            html = self._fetch_html(url)
            spider = StructuredDataSpider()
            selector = Selector(html, url=url)
            items = list(spider.parse(selector, url))
            if items:
                return ToolResult(
                    tool_name="extract_structured_data",
                    success=True,
                    output=items[0].to_json()[:8000],
                    data=items[0].to_dict(),
                )
            return ToolResult(tool_name="extract_structured_data", success=False, error="No structured data found")
        except Exception as exc:
            return ToolResult(tool_name="extract_structured_data", success=False, error=str(exc))

    def crawl_links(self, url: str, max_depth: int = 2, **kwargs: Any) -> ToolResult:
        visited: set[str] = set()
        results: list[dict[str, Any]] = []
        queue: list[tuple[str, int]] = [(url, 0)]
        while queue and len(visited) < 20:
            current_url, depth = queue.pop(0)
            if current_url in visited or depth > max_depth:
                continue
            visited.add(current_url)
            try:
                html = self._fetch_html(current_url)
                selector = Selector(html, url=current_url)
                results.append({
                    "url": current_url,
                    "title": selector.title(),
                    "depth": depth,
                })
                if depth < max_depth:
                    for link in selector.links()[:10]:
                        if link not in visited and urlparse(link).scheme in ("http", "https"):
                            queue.append((link, depth + 1))
            except Exception as exc:
                results.append({"url": current_url, "error": str(exc)})
        return ToolResult(
            tool_name="crawl_links",
            success=True,
            output=json.dumps(results, indent=2, ensure_ascii=False)[:8000],
            data={"pages_crawled": len(results), "max_depth": max_depth, "results": results},
        )

    def register_spider(self, name: str, spider_class: type[Spider]) -> None:
        self._spiders[name] = spider_class

    def run_spider(self, name: str, start_urls: list[str]) -> list[ScrapedItem]:
        spider_class = self._spiders.get(name, GenericSpider)
        spider = spider_class()
        spider.start_urls = start_urls
        items: list[ScrapedItem] = []
        for url, callback_name in spider.start_requests():
            try:
                html = self._fetch_html(url)
                selector = Selector(html, url=url)
                callback = getattr(spider, callback_name, spider.parse)
                for result in callback(selector, url):
                    if isinstance(result, ScrapedItem):
                        items.append(result)
                    elif isinstance(result, tuple):
                        follow_url, follow_cb = result
                        if spider.is_allowed(follow_url) and follow_url not in start_urls:
                            start_urls.append(follow_url)
            except Exception as exc:
                logger.warning("Spider %s failed on %s: %s", name, url, exc)
        return items
