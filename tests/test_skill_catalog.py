"""Tests for skill catalog search."""

import pytest

from daemon_engine.core.skill_catalog import (
    SkillCatalog,
    build_catalog,
    _compile_catalog_regex,
    _catalog_regex_score,
)
from daemon_engine.core.system_prompt import SkillEntry


def make_skill(name: str, description: str = "") -> SkillEntry:
    return SkillEntry(name=name, description=description, content="content")


@pytest.fixture
def sample_skills():
    return [
        make_skill("code-review", "Review code for quality and bugs"),
        make_skill("pdf", "Generate PDF documents from LaTeX"),
        make_skill("agent-builder", "Build custom AI agents interactively"),
        make_skill("mcp-builder", "Create MCP server plugins"),
        make_skill("data-analysis", "Analyze datasets with Python"),
        make_skill("deep-research", "Comprehensive web research"),
        make_skill("podcast-gen", "Generate podcast audio from text"),
    ]


@pytest.fixture
def catalog(sample_skills):
    return build_catalog(sample_skills)


class TestCompileRegex:
    def test_valid_regex(self):
        pattern = _compile_catalog_regex("code")
        assert pattern.search("Code Review")

    def test_invalid_regex_fallback(self):
        pattern = _compile_catalog_regex("[unclosed")
        assert pattern.search("[unclosed")

    def test_case_insensitive(self):
        pattern = _compile_catalog_regex("PDF")
        assert pattern.search("pdf documents")


class TestCatalogRegexScore:
    def test_no_matches(self):
        pattern = _compile_catalog_regex("xyz")
        skill = make_skill("test", "description")
        assert _catalog_regex_score(pattern, skill) == 0

    def test_matches(self):
        pattern = _compile_catalog_regex("code")
        skill = make_skill("code-review", "Review code quality")
        assert _catalog_regex_score(pattern, skill) > 0


class TestSkillCatalog:
    def test_creation(self, catalog):
        assert catalog.count() == 7

    def test_names(self, catalog):
        assert "code-review" in catalog.names
        assert "pdf" in catalog.names

    def test_all_names(self, catalog):
        names = catalog.all_names()
        assert len(names) == 7
        assert "code-review" in names

    def test_get_by_name(self, catalog):
        skill = catalog.get("pdf")
        assert skill is not None
        assert skill.name == "pdf"

    def test_get_nonexistent(self, catalog):
        assert catalog.get("nonexistent") is None

    def test_search_empty_query(self, catalog):
        assert catalog.search("") == []

    def test_search_select_exact(self, catalog):
        results = catalog.search("select:pdf,code-review")
        assert len(results) == 2
        names = [r.name for r in results]
        assert "pdf" in names
        assert "code-review" in names

    def test_search_select_single(self, catalog):
        results = catalog.search("select:data-analysis")
        assert len(results) == 1
        assert results[0].name == "data-analysis"

    def test_search_select_nonexistent(self, catalog):
        results = catalog.search("select:nonexistent")
        assert len(results) == 0

    def test_search_required_prefix(self, catalog):
        results = catalog.search("+podcast")
        assert len(results) >= 1
        assert any("podcast" in r.name for r in results)

    def test_search_required_prefix_with_ranking(self, catalog):
        results = catalog.search("+podcast gen")
        assert len(results) >= 1
        assert results[0].name == "podcast-gen"

    def test_search_free_text(self, catalog):
        results = catalog.search("code")
        assert len(results) >= 1
        assert any("code" in r.name.lower() for r in results)

    def test_search_free_text_description(self, catalog):
        results = catalog.search("LaTeX")
        assert len(results) >= 1
        assert any("pdf" in r.name for r in results)

    def test_search_max_results(self, sample_skills):
        many_skills = [make_skill(f"skill-{i}", f"common keyword {i}") for i in range(20)]
        catalog = build_catalog(many_skills)
        results = catalog.search("common")
        assert len(results) <= 5

    def test_search_name_scores_higher(self, catalog):
        skill_name = make_skill("research", "research things")
        skill_desc = make_skill("other", "do research analysis")
        cat = build_catalog([skill_name, skill_desc])
        results = cat.search("research")
        assert results[0].name == "research"

    def test_describe(self, catalog):
        desc = catalog.describe("pdf")
        assert desc is not None
        assert desc["name"] == "pdf"
        assert "description" in desc

    def test_describe_nonexistent(self, catalog):
        assert catalog.describe("nonexistent") is None

    def test_to_index(self, catalog):
        index = catalog.to_index()
        assert "code-review" in index
        assert "pdf" in index

    def test_to_index_empty(self):
        catalog = build_catalog([])
        assert "no skills" in catalog.to_index().lower()

    def test_immutable(self, catalog):
        with pytest.raises(Exception):
            catalog.skills = ()

    def test_invalid_regex_degrades_gracefully(self, catalog):
        results = catalog.search("[unclosed")
        assert isinstance(results, list)

    def test_bare_plus(self, catalog):
        results = catalog.search("+")
        assert results == []

    def test_count(self, catalog):
        assert catalog.count() == 7

    def test_build_catalog_empty(self):
        catalog = build_catalog([])
        assert catalog.count() == 0
