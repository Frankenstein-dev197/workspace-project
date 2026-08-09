# daemon-engine

Powerful agentic AI engine built from patterns extracted from 28 cloned repositories in `/workspace/project/cloned-repos/`.

## Architecture

Internal engine only — no UI, no mobile, no frontend. Subsystems:

- **core/** — Agent brain, reasoning, planning, watchdog, background tasks, system prompt builder, skill catalog, dedupe store, guardrails, MCP plugin, action history
- **models/** — LLM integration
- **multi_agent/** — Agent manager, orchestrator, communication, message bus, team protocols, swarm, channels
- **memory/** — Code memory, knowledge memory, long-term memory, recall, unified memory, savings ledger, conversation buffer
- **tools/** — Browser, research, scraping, DevOps, automation tools, auto-throttle, priority scheduler
- **runtime/** — Firecracker/sandbox, cron scheduler, worktree isolation, file operation locks, sandbox security

## Build & Test

```bash
cd /workspace/project/daemon-engine
python -m pytest tests/ -q          # 1150 tests, ~14s
python -m pytest tests/test_X.py -q # single module
```

## Key Patterns (by source)

### learn-claude-code sessions
- s07+s10 → SystemPromptBuilder (deterministic SHA-256 caching, SkillRegistry)
- s08 → ContextCompactor (4-level compaction: snip, micro, budget, LLM summary)
- s11 → ErrorRecoveryManager (retry with backoff, error classification)
- s12 → TaskGraph (DAG with blockedBy, claim/complete)
- s13 → BackgroundTaskManager (slow operation detection)
- s14 → CronScheduler (5-field cron, DOM/DOW OR semantics)
- s15 → MessageBus (file-based JSONL inboxes)
- s16 → ProtocolManager (request/response coordination)
- s17 → Autonomous teammates (idle poll, auto-claim)
- s18 → WorktreeManager (git worktree isolation)
- s19 → MCP Plugin System (MCPClient, MCPCatalog, tool discovery + dispatch)
- s20 → Comprehensive agent patterns

### AutoGPT
- Episode/EpisodicActionHistory (cursor-based, action+result pairs, summaries)
- Watchdog (loop detection)
- Context components (compaction)

### DeerFlow
- SkillCatalog (deferred discovery, select/+prefix/free-text search)
- MemoryDedupeStore (TTL + max entries, thread-safe, protocol for shared stores)
- File operation locks (WeakValueDictionary per-sandbox-path)
- Sandbox provider security gating (host bash restrictions, blocked commands)
- Channel system (IM platform abstraction, run policies, message envelopes)
- Extension-API contracts

### Headroom
- Hierarchical memory
- SavingsLedger (durable JSONL, cost-at-write-time, windowed aggregation)

### LangChain
- ConversationSummaryBufferMemory (sliding window + running summary, prune)
- Memory patterns: buffer, summary buffer, KG

### Scrapling
- AutoThrottle (per-domain adaptive rate limiting, Retry-After parsing)
- Scheduler priority queue (URL dedup, fingerprinting, snapshot/restore)

### browser-use
- Browser automation patterns

## Conventions

- Pure stdlib where possible; external deps only when needed (anthropic, etc.)
- Dataclasses with `to_dict`/`from_dict` for serialization
- Thread-safe modules with `threading.Lock`
- `__init__.py` re-exports with `__all__`
- Each module: comprehensive tests in `tests/test_<module>.py`
- Commit message convention: module summary + pattern source + test count

## Version Control

- Repo: /workspace/project/daemon-engine
- Remote: https://github.com/Frankenstein-dev197/workspace-project.git
- Branch: main
- Co-author: openhands <openhands@all-hands.dev>
