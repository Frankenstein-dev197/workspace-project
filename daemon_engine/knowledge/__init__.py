"""Knowledge base: curated knowledge from LeetCode, DevOps exercises, and reference docs.

Integrates content patterns from:
- azl397985856/leetcode (algorithmic problem patterns)
- bregman-arie/devops-exercises (DevOps Q&A knowledge)
- jaywcjlove/reference (quick reference sheets)
- langgenius/dify (knowledge base segmentation model)
"""

from daemon_engine.knowledge.knowledge_base import KnowledgeBase, KnowledgeEntry, KnowledgeSource
from daemon_engine.knowledge.algorithm_patterns import AlgorithmPatternLibrary
from daemon_engine.knowledge.devops_knowledge import DevOpsKnowledgeBase

__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeSource",
    "AlgorithmPatternLibrary",
    "DevOpsKnowledgeBase",
]
