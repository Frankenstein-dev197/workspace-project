"""
daemon-engine: A next-generation agentic AI engine.

Daemon-Engine unifies autonomous agents, multi-agent orchestration, persistent
memory, reasoning, software development, tool use, virtual execution, research,
and deployment automation into a single platform.

Architecture inspired by and integrating concepts from:
  - DeerFlow / Ruflo / LangChain / AutoGPT / learn-claude-code (agents & orchestration)
  - Transformers / DeepSeek-Reasonix (intelligence & reasoning)
  - Codebase Memory MCP / Google Skills / Headroom (memory & knowledge)
  - Browser-Use / Puppeteer / Scrapy / Scrapling / Sherlock / Ansible (tools)
  - Firecracker (execution environment)
  - Turborepo / Vercel (app architecture)
  - Expo / tsParticles / ui-buttons / Nerd Fonts (UI)
"""

from daemon_engine.engine import DaemonEngine

__version__ = "1.0.0"
__all__ = ["__version__", "DaemonEngine"]
