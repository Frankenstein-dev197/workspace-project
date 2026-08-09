"""Skill registry: central registry for skill activation and management.

Integrates DeerFlow's skill activation model (skills are activated on-demand,
their body loaded into agent context) with Google Skills' discovery system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daemon_engine.skills.skill import Skill, SkillCategory
from daemon_engine.skills.skill_loader import SkillLoader

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for skill discovery, activation, and management."""

    def __init__(self, loader: SkillLoader | None = None) -> None:
        self.loader = loader or SkillLoader()
        self._skills: dict[str, Skill] = {}
        self._activated: set[str] = set()
        self._secrets_provider: dict[str, str] = {}

    def discover(self) -> int:
        skills = self.loader.discover()
        for skill in skills:
            self._skills[skill.name] = skill
        return len(self._skills)

    def add_path(self, path: Path | str) -> None:
        self.loader.add_path(path)
        self.discover()

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def list_enabled(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.enabled]

    def list_activated(self) -> list[Skill]:
        return [self._skills[n] for n in self._activated if n in self._skills]

    def search(self, query: str) -> list[Skill]:
        return self.loader.search(query)

    def activate(self, name: str) -> bool:
        skill = self._skills.get(name)
        if not skill:
            logger.warning("Cannot activate unknown skill: %s", name)
            return False
        missing = self._check_secrets(skill)
        if missing:
            logger.warning("Skill %s missing secrets: %s", name, missing)
            return False
        self._activated.add(name)
        logger.info("Activated skill: %s", name)
        return True

    def deactivate(self, name: str) -> bool:
        if name in self._activated:
            self._activated.discard(name)
            logger.info("Deactivated skill: %s", name)
            return True
        return False

    def is_activated(self, name: str) -> bool:
        return name in self._activated

    def get_context(self, names: list[str] | None = None) -> str:
        if names is None:
            names = list(self._activated)
        parts: list[str] = []
        for name in names:
            skill = self._skills.get(name)
            if skill:
                parts.append(skill.to_prompt())
        if not parts:
            return ""
        return "\n\n---\n\n".join(parts)

    def get_tools_for_skill(self, name: str) -> list[str]:
        skill = self._skills.get(name)
        if not skill or not skill.allowed_tools:
            return []
        return list(skill.allowed_tools)

    def set_secret(self, name: str, value: str) -> None:
        self._secrets_provider[name] = value

    def _check_secrets(self, skill: Skill) -> list[str]:
        missing: list[str] = []
        for req in skill.required_secrets:
            if req.name not in self._secrets_provider:
                if not req.optional:
                    missing.append(req.name)
        return missing

    def create_skill(
        self, name: str, description: str, body: str, **kwargs: Any
    ) -> Skill:
        skill = self.loader.create_skill(name, description, body, **kwargs)
        self.register(skill)
        return skill

    def save_all(self, target_dir: Path | str) -> int:
        count = 0
        for skill in self._skills.values():
            if skill.category == SkillCategory.CUSTOM:
                try:
                    self.loader.save_skill(skill, target_dir)
                    count += 1
                except Exception as exc:
                    logger.error("Failed to save skill %s: %s", skill.name, exc)
        return count

    def stats(self) -> dict[str, Any]:
        return {
            "total_skills": len(self._skills),
            "enabled": len(self.list_enabled()),
            "activated": len(self._activated),
            "by_category": {
                cat.value: sum(1 for s in self._skills.values() if s.category == cat)
                for cat in SkillCategory
            },
        }
