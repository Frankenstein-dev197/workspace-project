"""Tests for system prompt builder and skill registry."""

import pytest

from daemon_engine.core.system_prompt import (
    SystemPromptBuilder,
    SkillRegistry,
    SkillEntry,
    create_prompt_builder,
    _parse_frontmatter,
    _parse_simple_frontmatter,
)


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("Hello world")
        assert meta == {}
        assert body == "Hello world"

    def test_with_frontmatter(self):
        text = "---\nname: test\ndescription: A test skill\n---\nBody content"
        meta, body = _parse_frontmatter(text)
        assert meta.get("name") == "test"
        assert "Body content" in body

    def test_malformed_frontmatter(self):
        text = "---\nbroken yaml: [unclosed\n---\nBody"
        meta, body = _parse_frontmatter(text)
        assert "Body" in body

    def test_simple_frontmatter(self):
        meta = _parse_simple_frontmatter("name: test\ndescription: A skill")
        assert meta["name"] == "test"
        assert meta["description"] == "A skill"


class TestSkillEntry:
    def test_creation(self):
        entry = SkillEntry(name="test", description="A test", content="content")
        assert entry.name == "test"
        assert entry.description == "A test"

    def test_to_dict(self):
        entry = SkillEntry(name="test", description="desc", content="content", path="/path")
        d = entry.to_dict()
        assert d["name"] == "test"
        assert d["path"] == "/path"


class TestSkillRegistry:
    def test_creation_empty(self):
        registry = SkillRegistry()
        registry.scan()
        assert registry.count() == 0

    def test_scan_with_dir(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill1 = skills_dir / "code-review"
        skill1.mkdir(parents=True)
        (skill1 / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: Review code\n---\nReview code here"
        )
        registry = SkillRegistry(skills_dir=skills_dir)
        registry.scan()
        assert registry.count() == 1
        skill = registry.get_skill("code-review")
        assert skill is not None
        assert skill.description == "Review code"

    def test_scan_multiple_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        for name in ["skill-a", "skill-b", "skill-c"]:
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}\n---\nBody")
        registry = SkillRegistry(skills_dir=skills_dir)
        registry.scan()
        assert registry.count() == 3

    def test_get_skill_nonexistent(self):
        registry = SkillRegistry()
        assert registry.get_skill("nonexistent") is None

    def test_load_skill_content(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "test"
        d.mkdir(parents=True)
        content = "---\nname: test\n---\nFull content here"
        (d / "SKILL.md").write_text(content)
        registry = SkillRegistry(skills_dir=skills_dir)
        registry.scan()
        loaded = registry.load_skill_content("test")
        assert loaded is not None
        assert "Full content" in loaded

    def test_load_skill_content_nonexistent(self):
        registry = SkillRegistry()
        assert registry.load_skill_content("nonexistent") is None

    def test_catalog(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "test"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: test\ndescription: A test skill\n---\nBody")
        registry = SkillRegistry(skills_dir=skills_dir)
        registry.scan()
        catalog = registry.catalog()
        assert "test" in catalog
        assert "A test skill" in catalog

    def test_catalog_empty(self):
        registry = SkillRegistry()
        registry.scan()
        catalog = registry.catalog()
        assert "no skills" in catalog.lower()

    def test_register_skill_manually(self):
        registry = SkillRegistry()
        entry = SkillEntry(name="manual", description="Manual", content="content")
        registry.register_skill(entry)
        assert registry.get_skill("manual") is not None

    def test_auto_scan_on_access(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "test"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: test\n---\nBody")
        registry = SkillRegistry(skills_dir=skills_dir)
        assert registry.count() == 1

    def test_skill_with_default_name(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "my-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("Just content without frontmatter")
        registry = SkillRegistry(skills_dir=skills_dir)
        registry.scan()
        skill = registry.get_skill("my-skill")
        assert skill is not None


class TestSystemPromptBuilder:
    def test_creation(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({})
        assert "coding agent" in prompt.lower() or "agent" in prompt.lower()

    def test_custom_identity(self):
        builder = SystemPromptBuilder(identity="You are a test agent.")
        prompt = builder.build({})
        assert "test agent" in prompt

    def test_tools_section(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({"enabled_tools": ["bash", "read", "write"]})
        assert "bash" in prompt
        assert "read" in prompt

    def test_workspace_section(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({"workspace": "/test/path"})
        assert "/test/path" in prompt

    def test_memory_section(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({"memories": "Remember to test"})
        assert "Remember to test" in prompt

    def test_no_memory_section_when_empty(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({"memories": ""})
        assert "Relevant memories" not in prompt

    def test_skills_section(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "test"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: test\ndescription: Test skill\n---\nBody")
        builder = SystemPromptBuilder(skills_dir=skills_dir)
        prompt = builder.build({})
        assert "test" in prompt
        assert "Test skill" in prompt

    def test_skills_disabled(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "test"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: test\n---\nBody")
        builder = SystemPromptBuilder(skills_dir=skills_dir)
        prompt = builder.build({"include_skills": False})
        assert "Skills available" not in prompt

    def test_instructions_section(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({"instructions": "Always test your code"})
        assert "Always test your code" in prompt

    def test_constraints_section(self):
        builder = SystemPromptBuilder()
        prompt = builder.build({"constraints": ["No sudo", "Stay in workspace"]})
        assert "No sudo" in prompt
        assert "Stay in workspace" in prompt

    def test_caching(self):
        builder = SystemPromptBuilder()
        context = {"workspace": "/test"}
        prompt1 = builder.build(context)
        prompt2 = builder.build(context)
        assert prompt1 == prompt2
        stats = builder.stats()
        assert stats["cache_hits"] == 1

    def test_cache_invalidation_on_context_change(self):
        builder = SystemPromptBuilder()
        prompt1 = builder.build({"workspace": "/path1"})
        prompt2 = builder.build({"workspace": "/path2"})
        assert prompt1 != prompt2
        stats = builder.stats()
        assert stats["cache_misses"] == 2

    def test_invalidate_cache(self):
        builder = SystemPromptBuilder()
        builder.build({"workspace": "/test"})
        builder.invalidate_cache()
        assert builder.stats()["cache_active"] is False

    def test_refresh_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        builder = SystemPromptBuilder(skills_dir=skills_dir)
        builder.build({})
        d = skills_dir / "new-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: new-skill\n---\nBody")
        builder.refresh_skills()
        prompt = builder.build({})
        assert "new-skill" in prompt

    def test_get_loaded_sections(self):
        builder = SystemPromptBuilder()
        sections = builder.get_loaded_sections({
            "enabled_tools": ["bash"],
            "workspace": "/test",
            "memories": "memory",
        })
        assert "identity" in sections
        assert "tools" in sections
        assert "workspace" in sections
        assert "memory" in sections

    def test_stats(self):
        builder = SystemPromptBuilder()
        builder.build({"workspace": "/test"})
        stats = builder.stats()
        assert stats["total_builds"] == 1
        assert stats["cache_misses"] == 1

    def test_stable_section_order(self):
        builder = SystemPromptBuilder()
        context = {
            "enabled_tools": ["bash"],
            "workspace": "/test",
            "memories": "mem",
            "instructions": "instr",
        }
        prompt = builder.build(context)
        identity_pos = prompt.find("coding agent")
        tools_pos = prompt.find("Available tools")
        workspace_pos = prompt.find("Working directory")
        memory_pos = prompt.find("Relevant memories")
        instructions_pos = prompt.find("instr")
        assert identity_pos < tools_pos < workspace_pos < memory_pos < instructions_pos


class TestCreatePromptBuilder:
    def test_default_creation(self):
        builder = create_prompt_builder()
        prompt = builder.build({})
        assert "agentic AI engine" in prompt

    def test_custom_identity(self):
        builder = create_prompt_builder(identity="Custom agent")
        prompt = builder.build({})
        assert "Custom agent" in prompt

    def test_with_skills_dir(self, tmp_path):
        builder = create_prompt_builder(skills_dir=tmp_path / "skills")
        prompt = builder.build({})
        assert prompt
