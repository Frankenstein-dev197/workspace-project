"""Skill catalog: deferred skill discovery with search.

Integrates DeerFlow SkillCatalog pattern:
- SkillCatalog: immutable, searchable catalog of skills
- Deferred discovery: LLM sees skill names but reads metadata on demand
- Search query forms:
  - "select:name1,name2" — exact match by name
  - "+required ranking" — require token in name, rank by ranking
  - "free text" — regex match on name + description
- Relevance ranking: name matches score higher than description-only
- Regex fallback: invalid regex degrades to literal substring match

This keeps the system prompt compact (only skill names) while giving
the model autonomous skill discovery via describe_skill.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from daemon_engine.core.system_prompt import SkillEntry

logger = logging.getLogger(__name__)

MAX_RESULTS = 5


def _compile_catalog_regex(pattern: str) -> re.Pattern[str]:
    """Compile pattern case-insensitively, falling back to literal match.

    Search queries come from the model, so an invalid regex must degrade
    to a literal substring match rather than raise.
    """
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def _catalog_regex_score(pattern: re.Pattern[str], skill: SkillEntry) -> int:
    """Count regex hits across name + description for ranking."""
    searchable = f"{skill.name} {skill.description or ''}"
    return len(pattern.findall(searchable))


@dataclass(frozen=True)
class SkillCatalog:
    """Immutable catalog of skills. Pure search, no mutation.

    Query forms:
    - "select:data-analysis,deep-research" — exact match by name
    - "+podcast gen" — require 'podcast' in name, rank by 'gen'
    - "chart visualization" — regex match on name + description
    """

    skills: tuple[SkillEntry, ...]

    @cached_property
    def names(self) -> frozenset[str]:
        """All skill names."""
        return frozenset(s.name for s in self.skills)

    def search(self, query: str) -> list[SkillEntry]:
        """Match query against skill names and descriptions.

        Returns at most MAX_RESULTS skills, ranked by relevance.
        """
        query = query.strip()
        if not query:
            return []

        # Exact selection by name
        if query.startswith("select:"):
            wanted = {n.strip() for n in query[7:].split(",")}
            return [s for s in self.skills if s.name in wanted]

        # Required-prefix search: "+required ranking"
        if query.startswith("+"):
            parts = query[1:].split(None, 1)
            if not parts:
                return []
            required = parts[0].lower()
            candidates = [s for s in self.skills if required in s.name.lower()]
            if len(parts) > 1:
                pattern = _compile_catalog_regex(parts[1])
                candidates.sort(
                    key=lambda s: _catalog_regex_score(pattern, s),
                    reverse=True,
                )
            return candidates[:MAX_RESULTS]

        # Free-text regex search
        regex = _compile_catalog_regex(query)
        scored: list[tuple[int, SkillEntry]] = []
        for s in self.skills:
            searchable = f"{s.name} {s.description or ''}"
            if regex.search(searchable):
                score = 2 if regex.search(s.name) else 1
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored][:MAX_RESULTS]

    def get(self, name: str) -> SkillEntry | None:
        """Get a skill by exact name."""
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def describe(self, name: str) -> dict[str, Any] | None:
        """Get full metadata for a skill (for describe_skill tool)."""
        skill = self.get(name)
        if not skill:
            return None
        return {
            "name": skill.name,
            "description": skill.description,
            "path": skill.path,
            "meta": skill.meta,
        }

    def count(self) -> int:
        """Number of skills in the catalog."""
        return len(self.skills)

    def all_names(self) -> list[str]:
        """List all skill names in insertion order."""
        return [s.name for s in self.skills]

    def to_index(self) -> str:
        """Generate compact index for system prompt injection.

        Only includes names (no descriptions) to keep prompt small.
        """
        if not self.skills:
            return "(no skills available)"
        return "\n".join(f"- {s.name}" for s in self.skills)


def build_catalog(skills: list[SkillEntry]) -> SkillCatalog:
    """Build an immutable catalog from a list of skills."""
    return SkillCatalog(skills=tuple(skills))
