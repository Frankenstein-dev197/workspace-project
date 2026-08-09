"""Tests for scraping tools (Scrapy/Scrapling integration)."""

import pytest

from daemon_engine.tools.scraping_tools import (
    Selector,
    Spider,
    GenericSpider,
    StructuredDataSpider,
    ScrapedItem,
    ScrapingTools,
)
from daemon_engine.tools.tool_registry import ToolRegistry


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page for scraping">
    <script type="application/ld+json">
    {"@type": "Article", "headline": "Test Article", "author": "Test Author"}
    </script>
</head>
<body>
    <h1>Main Heading</h1>
    <h2>Subheading</h2>
    <p>First paragraph of content.</p>
    <p>Second paragraph with <a href="/link1">a link</a>.</p>
    <a href="/page1">Page 1</a>
    <a href="/page2">Page 2</a>
    <a href="https://example.com/external">External</a>
    <img src="/img1.jpg" alt="Image 1">
    <img src="/img2.jpg" alt="Image 2">
    <table>
        <tr><th>Name</th><th>Value</th></tr>
        <tr><td>A</td><td>1</td></tr>
        <tr><td>B</td><td>2</td></tr>
    </table>
</body>
</html>
"""


class TestSelector:
    def test_title(self):
        sel = Selector(SAMPLE_HTML)
        assert sel.title() == "Test Page"

    def test_text(self):
        sel = Selector(SAMPLE_HTML)
        text = sel.text()
        assert "Main Heading" in text
        assert "First paragraph" in text

    def test_links(self):
        sel = Selector(SAMPLE_HTML, url="https://example.com/")
        links = sel.links()
        assert "/link1" in links or "https://example.com/link1" in links
        assert len(links) >= 3

    def test_images(self):
        sel = Selector(SAMPLE_HTML, url="https://example.com/")
        images = sel.images()
        assert len(images) >= 2

    def test_meta_description(self):
        sel = Selector(SAMPLE_HTML)
        assert sel.meta_description() == "A test page for scraping"

    def test_headings(self):
        sel = Selector(SAMPLE_HTML)
        headings = sel.headings()
        assert "Main Heading" in headings["h1"]
        assert "Subheading" in headings["h2"]

    def test_tables(self):
        sel = Selector(SAMPLE_HTML)
        tables = sel.tables()
        assert len(tables) == 1
        assert tables[0][0] == ["Name", "Value"]
        assert tables[0][1] == ["A", "1"]

    def test_css_selector(self):
        sel = Selector(SAMPLE_HTML)
        paras = sel.css("p")
        assert len(paras) >= 2

    def test_xpath_selector(self):
        sel = Selector(SAMPLE_HTML)
        links = sel.xpath("//a/@href")
        assert len(links) >= 3


class TestScrapedItem:
    def test_to_dict(self):
        item = ScrapedItem(url="https://example.com", title="Test", data={"key": "value"})
        d = item.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Test"
        assert d["data"]["key"] == "value"

    def test_to_json(self):
        item = ScrapedItem(url="https://example.com", title="Test")
        import json
        data = json.loads(item.to_json())
        assert data["url"] == "https://example.com"


class TestSpider:
    def test_generic_spider(self):
        spider = GenericSpider()
        sel = Selector(SAMPLE_HTML, url="https://example.com")
        items = list(spider.parse(sel, "https://example.com"))
        assert len(items) == 1
        assert items[0].title == "Test Page"
        assert "links" in items[0].data

    def test_structured_data_spider(self):
        spider = StructuredDataSpider()
        sel = Selector(SAMPLE_HTML, url="https://example.com")
        items = list(spider.parse(sel, "https://example.com"))
        assert len(items) == 1
        assert len(items[0].data["json_ld"]) >= 1

    def test_is_allowed(self):
        spider = GenericSpider()
        spider.allowed_domains = ["example.com"]
        assert spider.is_allowed("https://example.com/page") is True
        assert spider.is_allowed("https://other.com/page") is False


class TestScrapingTools:
    def test_register_all(self):
        registry = ToolRegistry()
        tools = ScrapingTools()
        tools.register_all(registry)
        tool_names = registry.list_tools()
        assert "scrape_page" in tool_names
        assert "scrape_with_selector" in tool_names
        assert "scrape_multiple" in tool_names
        assert "extract_structured_data" in tool_names
        assert "crawl_links" in tool_names

    def test_scrape_multiple_no_urls(self):
        tools = ScrapingTools()
        result = tools.scrape_multiple(urls=[])
        assert result.success is False
