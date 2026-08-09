# Architecture

## System Overview

Daemon-Engine is built as a layered architecture where each subsystem is
independently functional but designed to integrate seamlessly through the
top-level `DaemonEngine` class.

```
┌─────────────────────────────────────────────────────────┐
│                    DaemonEngine                          │
│              (engine.py — top-level facade)              │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  CORE    │MULTI-AGENT│ MEMORY  │  TOOLS   │   RUNTIME   │
│ ENGINE   │ SYSTEM    │ SYSTEM  │  SYSTEM  │   ENGINE    │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│Agent     │Orchest-  │Code     │Browser   │Sandbox      │
│Engine    │rator     │Memory   │Tools     │             │
│          │          │         │          ├─────────────┤
│Reasoning │Agent     │Knowledge│Research  │Virtual      │
│Engine    │Manager   │Memory   │Tools     │Computer     │
│          │          │         │          │Engine       │
│Task      │Communi-  │Long-Term│DevOps    │             │
│Planner   │cation    │Memory   │Tools     │             │
│          │System    │         │          │             │
│Decision  │          │Unified  │Automation│             │
│System    │          │Memory   │Tools     │             │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│                   MODEL INTEGRATION                      │
│      (OpenAI / Anthropic / Local / Mock providers)       │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Single-Agent Task Execution

```
User Goal
    │
    ▼
TaskPlanner.decompose(goal)
    │  ── LLM generates subtasks
    ▼
Task (leaf node)
    │
    ▼
Agent.run(task)
    │
    ├──► Agent._think()  ──► LLM.chat(messages)
    │         │
    │         ▼
    ├──► Agent._act()    ──► ToolRegistry.execute(tool, input)
    │         │                    │
    │         │              ┌─────┴─────┐
    │         │              ▼           ▼
    │         │         Safety Check  Tool Handler
    │         │              │           │
    │         │              └─────┬─────┘
    │         ▼                   │
    ├──► Agent._observe() ◄────────┘
    │         │
    │         ├──► Memory.store(step)
    │         │
    │    (loop until complete or max_turns)
    │
    ▼
AgentResult
```

### Multi-Agent Workflow

```
User Goal
    │
    ▼
Orchestrator.execute_goal(goal)
    │
    ├──► TaskPlanner.decompose(goal)
    │         ── Creates task tree
    │
    ├──► For each ready task:
    │         │
    │         ├──► AgentManager.spawn_agent(role)
    │         │         ── Selects agent by task type
    │         │
    │         ├──► CommunicationSystem.send(assignment)
    │         │
    │         ├──► Agent.run(task)
    │         │
    │         └──► CommunicationSystem.send(result)
    │
    ├──► Aggregate results
    │
    ▼
Workflow (completed/failed)
```

## Component Details

### Core Engine

#### Agent Engine
- Implements think-act-observe loop (learn-claude-code pattern)
- Supports configurable max turns, timeout, and tool subsets
- Integrates memory for context injection
- Returns structured AgentResult with step history

#### Reasoning Engine
- Five strategies: CoT, ReAct, Tree-of-Thought, Reflection, Self-Consistency
- Each strategy produces structured ReasoningStep list
- Self-Consistency runs 3 samples and aggregates
- Strategy comparison utility for benchmarking

#### Task Planner
- Recursive task decomposition with configurable depth
- Task tree with parent-child relationships
- Status tracking: pending → in_progress → completed/failed
- Priority levels: low, medium, high, critical
- Ready-task detection based on dependency completion

#### Decision System
- Four strategies: utility-based, rule-based, LLM-based, learning-based
- Utility scoring with configurable weights
- Learning-based uses historical success rates
- Full decision history tracking

### Multi-Agent System

#### Orchestrator
- Goal → task decomposition → agent assignment → result aggregation
- Automatic role selection based on task description keywords
- Workflow progress tracking
- Cancellation support

#### Agent Manager
- Role-based agent templates (researcher, coder, analyst, devops, orchestrator)
- Agent pool with health monitoring
- Available-agent selection (round-robin by completed tasks)
- Custom role registration

#### Communication System
- Direct messaging (agent to agent)
- Broadcast messaging (agent to all)
- Pub/sub subscriber callbacks
- Shared state dictionary
- Full message log

### Memory System

#### Code Memory
- Stores code artifacts (snippets, functions, classes, modules)
- SHA-256-based semantic embedding (no external dependencies)
- File path and tag indexing
- JSON persistence to disk
- Cosine similarity search

#### Knowledge Memory
- Graph structure: nodes (concepts) + edges (relationships)
- Category and tag organization
- Neighbor traversal with configurable depth
- JSON persistence

#### Long-Term Memory
- Five memory types: episodic, semantic, procedural, feedback, pattern
- Importance scoring with decay over time
- Reinforcement mechanism (boost important memories)
- Consolidation (prune low-importance, unaccessed memories)
- Recency-weighted recall

#### Unified Memory
- Single interface across all three memory layers
- Type-based routing (code/knowledge/episodic)
- Cross-layer search
- Unified stats and persistence

### Tool System

#### Tool Registry
- Central registration and dispatch
- Dangerous pattern detection (rm -rf, sudo, etc.)
- Category-based organization
- Execution logging
- Safe/unsafe tool classification

#### Browser Tools
- Web fetching with urllib
- Mock web search (replace with real search API)
- Link extraction via regex
- Text extraction (HTML tag stripping)
- Browser session navigation

#### Research Tools
- Web scraping (title, meta, headings extraction)
- OSINT username lookup (Sherlock-style platform enumeration)
- Data extraction (emails, URLs, phones, IPs)
- URL summarization
- Fact checking framework

#### DevOps Tools
- Shell command execution with safety checks
- File read/write/list operations
- Docker build/run (with simulation fallback)
- Ansible playbook loading (with simulation fallback)

#### Automation Tools
- Multi-step pipeline execution
- Task scheduling (simulated cron)
- Application deployment (Vercel-style)
- Test runner (pytest/npm test)
- Git operations (status, add, commit, push, etc.)

### Runtime Engine

#### Sandbox
- Isolated Python execution in temporary directory
- Blocked dangerous imports (os.system, subprocess.Popen, etc.)
- Resource limits (memory, CPU time, wall time)
- File tracking (created files detection)
- Cleanup on exit

#### Virtual Computer Engine
- Process management with PIDs
- Virtual filesystem tree
- Environment variable management
- System info (uptime, process counts, file counts)
- Graceful shutdown

### Model Integration

#### Base LLM Interface
- Abstract `chat()` and `embed()` methods
- Streaming support (`stream()`)
- Configurable model, temperature, max_tokens, timeout

#### Providers
- **MockProvider** — deterministic responses for testing (no API key needed)
- **OpenAIProvider** — OpenAI and compatible APIs
- **AnthropicProvider** — Claude models
- **LocalProvider** — Ollama (via HTTP) or Transformers (local inference)

## Integration Points

The `DaemonEngine` class in `engine.py` wires all subsystems together:

```python
DaemonEngine
├── llm: BaseLLM                    # Shared LLM instance
├── memory: UnifiedMemory           # Persistent memory
├── tool_registry: ToolRegistry     # 22 registered tools
├── communication: CommunicationSystem
├── agent_manager: AgentManager     # Agent pool
├── agent_engine: AgentEngine       # Single-agent execution
├── task_planner: TaskPlanner       # Task decomposition
├── reasoning_engine: ReasoningEngine
├── decision_system: DecisionSystem
├── sandbox: Sandbox                # Isolated execution
├── virtual_computer: VirtualComputerEngine
└── orchestrator: Orchestrator      # Multi-agent workflows
```

All subsystems share the same LLM, memory, and tool registry, enabling
agents to use tools, remember context, and reason within a unified
environment.
