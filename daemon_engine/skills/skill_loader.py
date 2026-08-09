"""Skill loader: discovers and parses SKILL.md files from directories.

Integrates Google Skills' directory structure (categories/ads/, categories/cloud/, etc.)
with DeerFlow's skill storage system. Recursively finds SKILL.md files and parses
their frontmatter into Skill objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from daemon_engine.skills.skill import Skill, SkillCategory

logger = logging.getLogger(__name__)

SKILL_MD_FILE = "SKILL.md"


class SkillLoader:
    """Discovers and loads skills from the filesystem."""

    def __init__(self, search_paths: list[Path | str] | None = None) -> None:
        self.search_paths: list[Path] = []
        if search_paths:
            self.search_paths = [Path(p) for p in search_paths]
        self._cache: dict[str, Skill] = {}

    def add_path(self, path: Path | str) -> None:
        p = Path(path)
        if p not in self.search_paths:
            self.search_paths.append(p)

    def discover(self) -> list[Skill]:
        skills: list[Skill] = []
        seen_names: set[str] = set()
        for base in self.search_paths:
            if not base.exists():
                continue
            for skill_file in self._find_skill_files(base):
                try:
                    content = skill_file.read_text(encoding="utf-8")
                    skill = Skill.from_frontmatter(content, skill_file=skill_file)
                    if skill.name in seen_names:
                        continue
                    seen_names.add(skill.name)
                    skills.append(skill)
                    self._cache[skill.name] = skill
                except Exception as exc:
                    logger.warning("Failed to load skill %s: %s", skill_file, exc)
        logger.info("Discovered %d skills from %d paths", len(skills), len(self.search_paths))
        return skills

    def _find_skill_files(self, base: Path) -> Iterator[Path]:
        if base.is_file() and base.name == SKILL_MD_FILE:
            yield base
            return
        if base.is_dir():
            yield from sorted(base.rglob(SKILL_MD_FILE))

    def load_by_name(self, name: str) -> Skill | None:
        if name in self._cache:
            return self._cache[name]
        if not self._cache:
            self.discover()
        return self._cache.get(name)

    def load_by_category(self, category: SkillCategory) -> list[Skill]:
        if not self._cache:
            self.discover()
        return [s for s in self._cache.values() if s.category == category]

    def search(self, query: str) -> list[Skill]:
        if not self._cache:
            self.discover()
        query_lower = query.lower()
        results: list[tuple[float, Skill]] = []
        for skill in self._cache.values():
            score = 0.0
            if query_lower in skill.name.lower():
                score += 3.0
            if query_lower in skill.description.lower():
                score += 2.0
            for tag in skill.tags:
                if query_lower in tag.lower():
                    score += 1.5
            if query_lower in skill.body.lower():
                score += 0.5
            if score > 0:
                results.append((score, skill))
        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]

    def create_skill(self, name: str, description: str, body: str, **kwargs) -> Skill:
        skill = Skill(
            name=name,
            description=description,
            body=body,
            category=SkillCategory.CUSTOM,
            **kwargs,
        )
        self._cache[name] = skill
        return skill

    def save_skill(self, skill: Skill, target_dir: Path | str) -> Path:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        skill_dir = target / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / SKILL_MD_FILE
        content = self._skill_to_markdown(skill)
        skill_file.write_text(content, encoding="utf-8")
        skill.skill_dir = skill_dir
        skill.skill_file = skill_file
        return skill_file

    def _skill_to_markdown(self, skill: Skill) -> str:
        lines = ["---"]
        lines.append(f"name: {skill.name}")
        desc = skill.description.replace("\n", " ")
        lines.append(f"description: {desc}")
        if skill.license:
            lines.append(f"license: {skill.license}")
        lines.append(f"category: {skill.category.value}")
        lines.append(f"enabled: {skill.enabled}")
        if skill.version:
            lines.append(f"version: \"{skill.version}\"")
        if skill.author:
            lines.append(f"author: {skill.author}")
        if skill.tags:
            lines.append(f"tags: [{', '.join(skill.tags)}]")
        if skill.compatibility:
            lines.append(f"compatibility: \"{skill.compatibility}\"")
        if skill.allowed_tools:
            tools = ", ".join(skill.allowed_tools)
            lines.append(f"allowed_tools: [{tools}]")
        lines.append("---")
        lines.append("")
        lines.append(skill.body)
        return "\n".join(lines)
