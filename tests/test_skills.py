"""Tests for the skills system."""

import tempfile
from pathlib import Path

import pytest

from daemon_engine.skills.skill import Skill, SkillCategory, SecretRequirement
from daemon_engine.skills.skill_loader import SkillLoader
from daemon_engine.skills.skill_registry import SkillRegistry


SKILL_MD_TEMPLATE = """---
name: test-skill
description: A test skill for unit testing
license: MIT
version: "1.0"
author: test-author
tags: [testing, unit-test]
compatibility: "Requires Python 3.10+"
enabled: true
---

# Test Skill

This is a test skill body with instructions.
It contains multiple paragraphs of content.

## Section 1
Some content here.

## Section 2
More content here.
"""


class TestSkill:
    def test_from_frontmatter(self):
        skill = Skill.from_frontmatter(SKILL_MD_TEMPLATE)
        assert skill.name == "test-skill"
        assert "test skill" in skill.description.lower()
        assert skill.license == "MIT"
        assert skill.version == "1.0"
        assert skill.author == "test-author"
        assert "testing" in skill.tags
        assert skill.enabled is True
        assert skill.compatibility is not None

    def test_content_hash(self):
        skill = Skill(name="test", description="test", body="content here")
        h1 = skill.content_hash()
        skill2 = Skill(name="test", description="test", body="content here")
        assert h1 == skill2.content_hash()

    def test_to_prompt(self):
        skill = Skill(name="my-skill", description="Does something", body="Body content")
        prompt = skill.to_prompt()
        assert "my-skill" in prompt
        assert "Does something" in prompt
        assert "Body content" in prompt

    def test_to_dict(self):
        skill = Skill(name="test", description="desc", version="2.0")
        d = skill.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "2.0"

    def test_secret_requirement(self):
        req = SecretRequirement(name="API_KEY", optional=False)
        assert req.name == "API_KEY"
        assert req.optional is False


class TestSkillLoader:
    def test_discover_from_directory(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SKILL_MD_TEMPLATE)

        loader = SkillLoader(search_paths=[tmp_path])
        skills = loader.discover()
        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_load_by_name(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SKILL_MD_TEMPLATE)

        loader = SkillLoader(search_paths=[tmp_path])
        skill = loader.load_by_name("test-skill")
        assert skill is not None
        assert skill.name == "test-skill"

    def test_search(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SKILL_MD_TEMPLATE)

        loader = SkillLoader(search_paths=[tmp_path])
        results = loader.search("testing")
        assert len(results) > 0

    def test_create_and_save_skill(self, tmp_path):
        loader = SkillLoader()
        skill = loader.create_skill("custom", "A custom skill", "Custom body content")
        assert skill.name == "custom"
        saved_path = loader.save_skill(skill, tmp_path)
        assert saved_path.exists()
        assert saved_path.name == "SKILL.md"


class TestSkillRegistry:
    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test skill", body="Body")
        registry.register(skill)
        assert registry.get("test") is skill

    def test_activate(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test", body="Body")
        registry.register(skill)
        assert registry.activate("test") is True
        assert registry.is_activated("test") is True

    def test_activate_unknown(self):
        registry = SkillRegistry()
        assert registry.activate("nonexistent") is False

    def test_deactivate(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test", body="Body")
        registry.register(skill)
        registry.activate("test")
        assert registry.deactivate("test") is True
        assert not registry.is_activated("test")

    def test_get_context(self):
        registry = SkillRegistry()
        skill1 = Skill(name="skill1", description="First", body="Body 1")
        skill2 = Skill(name="skill2", description="Second", body="Body 2")
        registry.register(skill1)
        registry.register(skill2)
        registry.activate("skill1")
        registry.activate("skill2")
        ctx = registry.get_context()
        assert "skill1" in ctx
        assert "skill2" in ctx

    def test_secret_check(self):
        registry = SkillRegistry()
        skill = Skill(
            name="secret-skill",
            description="Needs secrets",
            body="Body",
            required_secrets=(SecretRequirement(name="API_KEY"),),
        )
        registry.register(skill)
        assert registry.activate("secret-skill") is False
        registry.set_secret("API_KEY", "test-key")
        assert registry.activate("secret-skill") is True

    def test_stats(self):
        registry = SkillRegistry()
        registry.register(Skill(name="s1", description="d1"))
        registry.register(Skill(name="s2", description="d2", category=SkillCategory.CUSTOM))
        stats = registry.stats()
        assert stats["total_skills"] == 2
        assert stats["by_category"]["custom"] == 1
