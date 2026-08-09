"""Skills system: dynamic skill loading and management.

Integrates Google Skills (SKILL.md format with frontmatter) and DeerFlow's
skill storage/activation model. Skills are markdown files with YAML
frontmatter that define specialized knowledge agents can load on demand.
"""

from daemon_engine.skills.skill import Skill, SkillCategory, SecretRequirement
from daemon_engine.skills.skill_loader import SkillLoader
from daemon_engine.skills.skill_registry import SkillRegistry

__all__ = [
    "Skill",
    "SkillCategory",
    "SecretRequirement",
    "SkillLoader",
    "SkillRegistry",
]
