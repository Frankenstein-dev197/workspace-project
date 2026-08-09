"""System prompt builder with dynamic sections and caching.

Integrates learn-claude-code s07 (skill loading) + s10 (system prompt):
- SystemPromptBuilder: assembles prompt from sections based on context
- SkillRegistry: scans SKILL.md files with YAML frontmatter parsing
- Deterministic caching: reassemble only when context changes
- Dynamic sections: identity, tools, workspace, memory, skills
- Skill catalog: cheap injection (names + descriptions only)
- load_skill: returns full SKILL.md content on demand

Prompt sections are assembled in stable order for API-level cache hits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class SkillEntry:
    """A skill loaded from SKILL.md."""
    name: str
    description: str
    content: str
    path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
        }


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        if yaml is not None:
            meta = yaml.safe_load(parts[1]) or {}
        else:
            meta = _parse_simple_frontmatter(parts[1])
    except Exception:
        meta = _parse_simple_frontmatter(parts[1])
    return meta, parts[2].strip()


def _parse_simple_frontmatter(text: str) -> dict[str, Any]:
    """Simple key: value parser for when yaml is not available."""
    meta: dict[str, Any] = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


class SkillRegistry:
    """Scans and manages skills from SKILL.md files."""

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._skills: dict[str, SkillEntry] = {}
        self._scanned = False

    def scan(self) -> None:
        """Scan skills directory and populate registry."""
        self._skills.clear()
        if not self._skills_dir or not self._skills_dir.exists():
            self._scanned = True
            return
        for d in sorted(self._skills_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest = d / "SKILL.md"
            if manifest.exists():
                self._load_skill_file(manifest, d.name)
        self._scanned = True
        logger.info("Scanned skills: %d found", len(self._skills))

    def _load_skill_file(self, manifest: Path, default_name: str) -> None:
        """Load a single SKILL.md file."""
        try:
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", default_name)
            desc = meta.get(
                "description",
                raw.split("\n")[0].lstrip("#").strip() if raw else "",
            )
            entry = SkillEntry(
                name=name,
                description=desc,
                content=raw,
                path=str(manifest.parent),
                meta=meta,
            )
            self._skills[name] = entry
        except Exception as exc:
            logger.error("Failed to load skill %s: %s", manifest, exc)

    def get_skill(self, name: str) -> SkillEntry | None:
        """Get a skill by name."""
        if not self._scanned:
            self.scan()
        return self._skills.get(name)

    def load_skill_content(self, name: str) -> str | None:
        """Load full SKILL.md content for a skill."""
        skill = self.get_skill(name)
        return skill.content if skill else None

    def list_skills(self) -> list[SkillEntry]:
        """List all registered skills."""
        if not self._scanned:
            self.scan()
        return list(self._skills.values())

    def catalog(self) -> str:
        """Generate skill catalog (names + descriptions only)."""
        skills = self.list_skills()
        if not skills:
            return "(no skills found)"
        return "\n".join(
            f"- **{s.name}**: {s.description}" for s in skills
        )

    def register_skill(self, entry: SkillEntry) -> None:
        """Manually register a skill."""
        self._scanned = True
        self._skills[entry.name] = entry

    def count(self) -> int:
        if not self._scanned:
            self.scan()
        return len(self._skills)


class SystemPromptBuilder:
    """Assembles system prompts from dynamic sections with caching.

    Sections are assembled in stable order for API-level prompt cache:
    1. identity (always)
    2. tools (dynamic)
    3. workspace (dynamic)
    4. skills catalog (dynamic)
    5. memory (conditional)
    6. instructions (conditional)
    """

    def __init__(
        self,
        identity: str = "You are a coding agent. Act, don't explain.",
        skills_dir: str | Path | None = None,
    ) -> None:
        self._identity = identity
        self._skill_registry = SkillRegistry(skills_dir)
        self._cache_key: str | None = None
        self._cached_prompt: str | None = None
        self._stats = {
            "total_builds": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    def _context_key(self, context: dict[str, Any]) -> str:
        """Generate deterministic cache key from context.

        Uses json.dumps with sorted keys for deterministic serialization.
        """
        return hashlib.sha256(
            json.dumps(context, sort_keys=True, default=str).encode()
        ).hexdigest()

    def assemble(self, context: dict[str, Any]) -> str:
        """Assemble system prompt from sections based on context."""
        sections: list[str] = []
        sections.append(self._identity)
        tools = context.get("enabled_tools", [])
        if tools:
            if isinstance(tools, list):
                tools_str = ", ".join(tools)
            else:
                tools_str = str(tools)
            sections.append(f"Available tools: {tools_str}.")
        workspace = context.get("workspace")
        if workspace:
            sections.append(f"Working directory: {workspace}")
        if context.get("include_skills", True):
            catalog = self._skill_registry.catalog()
            if catalog and catalog != "(no skills found)":
                sections.append(
                    f"Skills available:\n{catalog}\n"
                    "Use load_skill to get full details when needed."
                )
        memories = context.get("memories", "")
        if memories:
            sections.append(f"Relevant memories:\n{memories}")
        instructions = context.get("instructions", "")
        if instructions:
            sections.append(instructions)
        constraints = context.get("constraints", [])
        if constraints:
            constraint_text = "\n".join(f"- {c}" for c in constraints)
            sections.append(f"Constraints:\n{constraint_text}")
        return "\n\n".join(sections)

    def build(self, context: dict[str, Any]) -> str:
        """Build system prompt with caching. Reassembles only on context change."""
        key = self._context_key(context)
        self._stats["total_builds"] += 1
        if key == self._cache_key and self._cached_prompt is not None:
            self._stats["cache_hits"] += 1
            return self._cached_prompt
        self._stats["cache_misses"] += 1
        self._cache_key = key
        self._cached_prompt = self.assemble(context)
        return self._cached_prompt

    def invalidate_cache(self) -> None:
        """Invalidate the cached prompt."""
        self._cache_key = None
        self._cached_prompt = None

    def refresh_skills(self) -> None:
        """Rescan skills and invalidate cache."""
        self._skill_registry.scan()
        self.invalidate_cache()

    def get_loaded_sections(self, context: dict[str, Any]) -> list[str]:
        """Get list of section names that would be loaded."""
        loaded = ["identity"]
        if context.get("enabled_tools"):
            loaded.append("tools")
        if context.get("workspace"):
            loaded.append("workspace")
        if context.get("include_skills", True) and self._skill_registry.count() > 0:
            loaded.append("skills")
        if context.get("memories"):
            loaded.append("memory")
        if context.get("instructions"):
            loaded.append("instructions")
        if context.get("constraints"):
            loaded.append("constraints")
        return loaded

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "skills_count": self._skill_registry.count(),
            "cache_active": self._cached_prompt is not None,
        }


def create_prompt_builder(
    identity: str | None = None,
    skills_dir: str | Path | None = None,
) -> SystemPromptBuilder:
    """Create a system prompt builder with optional custom identity."""
    if identity is None:
        identity = (
            "You are a powerful agentic AI engine. "
            "Act decisively, reason carefully, and complete tasks fully."
        )
    return SystemPromptBuilder(identity=identity, skills_dir=skills_dir)
