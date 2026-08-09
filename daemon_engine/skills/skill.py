"""Skill model: metadata and content for a skill.

Integrates the SKILL.md format from Google Skills (frontmatter + markdown body)
and DeerFlow's Skill dataclass (category, allowed_tools, required_secrets).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SkillCategory(str, Enum):
    """Source category for a skill (from DeerFlow)."""
    PUBLIC = "public"
    CUSTOM = "custom"
    INTEGRATION = "integrations"
    LEGACY = "legacy"


@dataclass(frozen=True)
class SecretRequirement:
    """A secret a skill declares it needs (from DeerFlow issue #3861)."""
    name: str
    optional: bool = False


@dataclass
class Skill:
    """A skill with its metadata, content, and file path.

    Format: YAML frontmatter + Markdown body, inspired by Google Skills.
    """
    name: str
    description: str
    body: str = ""
    license: str | None = None
    skill_dir: Path | None = None
    skill_file: Path | None = None
    category: SkillCategory = SkillCategory.PUBLIC
    allowed_tools: tuple[str, ...] | None = None
    enabled: bool = False
    required_secrets: tuple[SecretRequirement, ...] = field(default_factory=tuple)
    secrets_autonomous: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    compatibility: str | None = None

    @property
    def skill_path(self) -> str:
        return str(self.skill_file) if self.skill_file else self.name

    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode()).hexdigest()[:16]

    def to_prompt(self, include_body: bool = True) -> str:
        parts = [f"# Skill: {self.name}", f"Description: {self.description}"]
        if self.compatibility:
            parts.append(f"Compatibility: {self.compatibility}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if include_body and self.body:
            parts.append("")
            parts.append(self.body)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "enabled": self.enabled,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "license": self.license,
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools else None,
            "required_secrets": [
                {"name": s.name, "optional": s.optional} for s in self.required_secrets
            ],
            "content_hash": self.content_hash(),
            "body_length": len(self.body),
        }

    @classmethod
    def from_frontmatter(cls, content: str, skill_file: Path | None = None) -> Skill:
        fm, body = _parse_frontmatter(content)
        name = fm.get("name", skill_file.stem if skill_file else "unnamed")
        description = fm.get("description", "")
        if isinstance(description, str) and len(description) > 500:
            description = description[:500]
        category_str = fm.get("category", "public")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.PUBLIC
        meta = fm.get("metadata", {})
        allowed = fm.get("allowed_tools")
        if isinstance(allowed, str):
            allowed = tuple(allowed.split(","))
        elif isinstance(allowed, list):
            allowed = tuple(allowed)
        secrets_raw = fm.get("required_secrets", [])
        secrets: list[SecretRequirement] = []
        if isinstance(secrets_raw, list):
            for s in secrets_raw:
                if isinstance(s, dict):
                    secrets.append(SecretRequirement(
                        name=s.get("name", ""),
                        optional=s.get("optional", False),
                    ))
                elif isinstance(s, str):
                    secrets.append(SecretRequirement(name=s))
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        return cls(
            name=name,
            description=str(description),
            body=body,
            license=fm.get("license"),
            skill_dir=skill_file.parent if skill_file else None,
            skill_file=skill_file,
            category=category,
            allowed_tools=allowed,
            enabled=fm.get("enabled", False),
            required_secrets=tuple(secrets),
            secrets_autonomous=fm.get("secrets-autonomous", True),
            metadata=meta if isinstance(meta, dict) else {},
            version=str(fm.get("version", meta.get("version", "1.0") if isinstance(meta, dict) else "1.0")),
            author=fm.get("author", meta.get("author") if isinstance(meta, dict) else None),
            tags=list(tags) if isinstance(tags, list) else [],
            compatibility=fm.get("compatibility"),
        )


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    fm_text, body = match.group(1), match.group(2)
    fm: dict[str, Any] = {}
    current_key: str | None = None
    current_val: list[str] = []
    for line in fm_text.split("\n"):
        if line.startswith("  ") and current_key:
            current_val.append(line.strip())
        elif ":" in line and not line.startswith(" "):
            if current_key and current_val:
                fm[current_key] = "\n".join(current_val)
            current_val = []
            key, _, val = line.partition(":")
            current_key = key.strip()
            val = val.strip()
            if val:
                val = val.strip('"').strip("'")
                fm[current_key] = val
                current_key = None
    if current_key and current_val:
        fm[current_key] = "\n".join(current_val)
    for key, val in list(fm.items()):
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            items = [item.strip().strip('"').strip("'") for item in val[1:-1].split(",") if item.strip()]
            fm[key] = items
        elif isinstance(val, str):
            lower = val.lower()
            if lower == "true":
                fm[key] = True
            elif lower == "false":
                fm[key] = False
    return fm, body.strip()
