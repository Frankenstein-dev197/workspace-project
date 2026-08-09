# Daemon Engine

**A next-generation agentic AI engine.**

Daemon-Engine is a unified, autonomous AI agent platform that integrates agents, reasoning, memory, tools, runtime execution, and multi-agent orchestration into a single functional engine.

## Overview

Daemon-Engine unifies autonomous agents, multi-agent orchestration, persistent memory, reasoning, software development, tool use, virtual execution, research, and deployment automation into a single platform.

The engine integrates architectural concepts and patterns from 28 open-source repositories:

### Agents & Orchestration
- **DeerFlow** — subagent executor and registry patterns
- **Ruflo** — swarm management and learning-based routing
- **LangChain** — agent and tool abstractions
- **AutoGPT** — autonomous goal-seeking loop
- **learn-claude-code** — agent loop, tool use, and team protocols

### Intelligence & Reasoning
- **Transformers** — model integration and local inference
- **DeepSeek-Reasonix** — structured reasoning chains, bounded LLM, memory hierarchy

### Memory & Knowledge
- **Codebase Memory MCP** — code-aware persistent memory with LSP-style indexing
- **Google Skills** — skill-based knowledge organization
- **Headroom** — graph-based knowledge store
- **LeetCode / DevOps Exercises / Reference** — knowledge base content

### Tools & Actions
- **Browser-Use** — AI-driven browser automation
- **Puppeteer** — headless browser control patterns
- **Scrapy / Scrapling** — structured web scraping
- **Sherlock** — OSINT username lookup
- **Ansible** — playbook execution and infrastructure automation

### Execution Environment
- **Firecracker** — microVM sandboxing concepts

### App Architecture
- **Turborepo / Vercel** — monorepo and deployment patterns

## Architecture

```
daemon_engine/
├── core/
│   ├── agent_engine.py        # Autonomous agent with think-act-observe loop
│   ├── reasoning_engine.py    # Multi-strategy reasoning (CoT, ReAct, ToT, Reflection)
│   ├── task_planner.py        # Hierarchical task decomposition
│   └── decision_system.py     # Utility, rule, LLM, and learning-based decisions
│
├── multi_agent/
│   ├── orchestrator.py        # Top-level workflow coordination
│   ├── agent_manager.py       # Agent pool lifecycle and role management
│   └── communication_system.py # Inter-agent messaging (pub/sub, direct, broadcast)
│
├── memory/
│   ├── code_memory.py         # Code artifact storage with semantic search
│   ├── knowledge_memory.py    # Graph-based knowledge store
│   ├── long_term_memory.py    # Episodic/semantic/procedural memory with decay
│   └── unified.py             # Aggregated memory interface
│
├── tools/
│   ├── tool_registry.py       # Central tool registry with safety checks
│   ├── browser_tools.py       # Web fetching, search, content extraction
│   ├── research_tools.py      # OSINT, data extraction, fact checking
│   ├── devops_tools.py        # Shell, file I/O, Docker, Ansible
│   └── automation_tools.py    # Pipelines, scheduling, deployment, git
│
├── runtime/
│   ├── sandbox.py             # Isolated code execution with resource limits
│   └── virtual_computer_engine.py # Virtual filesystem, processes, env vars
│
├── models/
│   ├── base.py                # Abstract LLM interface
│   └── providers.py           # OpenAI, Anthropic, Local (Ollama/Transformers), Mock
│
├── engine.py                  # DaemonEngine — connects all subsystems
└── cli.py                     # Command-line interface
```

## Installation

```bash
# Basic installation
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With Anthropic support
pip install -e ".[anthropic]"

# With local model support (Transformers)
pip install -e ".[local]"

# With everything
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from daemon_engine import DaemonEngine

# Create the engine (uses MockProvider without API keys)
engine = DaemonEngine()

# Run a multi-agent workflow
workflow = engine.run_goal("Build a REST API for a todo app")
print(f"Status: {workflow.status.value}")
print(f"Tasks: {len(workflow.tasks)}")

# Reason about a problem
result = engine.reason("How to optimize database queries?")
print(f"Conclusion: {result.conclusion}")

# Use a tool
result = engine.use_tool("web_search", query="python tutorials")
print(result.output)

# Execute code in sandbox
result = engine.execute_code("print('Hello from sandbox!')")
print(result.result.stdout)

# Store and recall memories
engine.remember("Project uses Python 3.12", memory_type="semantic")
print(engine.recall("Python"))

# Check system status
import json
print(json.dumps(engine.system_status(), indent=2, default=str))

# Shutdown
engine.shutdown()
```

### Using a Real LLM

```python
from daemon_engine import DaemonEngine
from daemon_engine.models.base import LLMConfig
from daemon_engine.models.providers import OpenAIProvider

# Configure with OpenAI
llm = OpenAIProvider(LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
))
engine = DaemonEngine(llm=llm)
```

Or set environment variables:

```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

### CLI

```bash
# Show engine status
daemon-engine status

# Run a multi-agent workflow
daemon-engine run "Build a web scraper"

# Reason about a problem
daemon-engine reason "How to scale a web application?"

# Plan tasks for a goal
daemon-engine plan "Create a REST API"

# Execute code in sandbox
daemon-engine exec "print('hello world')"

# Use a specific tool
daemon-engine tool web_search query="python tutorials"

# Interactive REPL mode
daemon-engine interact
```

## Core Concepts

### Agent Engine

The agent engine implements a **think-act-observe loop**:

1. **Think**: The agent receives the task and reasons about the next step
2. **Act**: The agent selects and executes a tool
3. **Observe**: The agent processes the tool result and decides whether to continue

```python
from daemon_engine.core.agent_engine import Agent, AgentConfig

config = AgentConfig(
    name="my-agent",
    max_turns=20,
    tools=["web_search", "bash"],
)
agent = engine.create_agent()
result = agent.run(task)
```

### Multi-Agent Orchestration

The orchestrator decomposes goals into tasks, assigns them to specialized agents, and aggregates results:

- **Researcher** — finds and collects information
- **Coder** — writes, reviews, and debugs code
- **Analyst** — analyzes data and provides insights
- **DevOps** — handles deployment and infrastructure
- **Orchestrator** — coordinates other agents

```python
workflow = engine.run_goal("Research and compare Python web frameworks")
summary = engine.orchestrator.get_workflow_summary(workflow.id)
```

### Reasoning Engine

Supports five reasoning strategies:

- **Chain of Thought** — step-by-step linear reasoning
- **ReAct** — reasoning + acting framework
- **Tree of Thought** — multi-path exploration
- **Reflection** — propose, critique, refine
- **Self-Consistency** — multiple samples aggregated

```python
from daemon_engine.core.reasoning_engine import ReasoningStrategy

result = engine.reason("Complex problem", strategy=ReasoningStrategy.TREE_OF_THOUGHT)
```

### Memory System

Three-layer memory architecture:

- **Code Memory** — stores code artifacts with semantic search (inspired by Codebase Memory MCP)
- **Knowledge Memory** — graph-based knowledge store with relationships (inspired by Headroom)
- **Long-Term Memory** — episodic, semantic, procedural memory with importance decay (inspired by DeepSeek-Reasonix)

```python
# Store memories
engine.remember("def hello(): print('hi')", memory_type="code", language="python")
engine.remember("Python is great for AI", memory_type="knowledge", concept="Python")
engine.remember("I solved a bug today", memory_type="episodic")

# Recall
print(engine.recall("Python"))
```

### Tool System

22 built-in tools across 4 categories:

| Category | Tools |
|----------|-------|
| Browser | web_fetch, web_search, extract_links, extract_text, browser_navigate |
| Research | web_scraper, osint_lookup, data_extract, summarize_url, fact_check |
| DevOps | bash, file_read, file_write, file_list, docker_build, docker_run, ansible_playbook |
| Automation | run_pipeline, schedule_task, deploy_application, run_tests, git_operations |

```python
# Use tools directly
result = engine.use_tool("bash", command="ls -la")
result = engine.use_tool("osint_lookup", username="someuser")
```

### Runtime Engine

Secure sandbox execution with:

- Python code execution with blocked dangerous patterns
- Shell command execution with safety checks
- Virtual computer with filesystem, process table, and environment variables
- Resource limits (memory, CPU time, wall time)

```python
# Execute Python in sandbox
result = engine.execute_code("import math; print(math.pi)")

# Execute shell command
result = engine.execute_command("echo 'hello'")

# Virtual computer info
print(engine.virtual_computer.system_info())
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=daemon_engine

# Run specific test module
pytest tests/test_agent_engine.py
```

All 144 tests pass:

```
============================= 144 passed in 2.72s ==============================
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DAEMON_LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, `local`, `mock` |
| `DAEMON_LLM_MODEL` | Model name (e.g., `gpt-4`, `claude-sonnet-4-20250514`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible base URL |
| `ANTHROPIC_BASE_URL` | Custom Anthropic base URL |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `DAEMON_LOCAL_MODEL` | Local model name (e.g., `ollama/llama3`) |

## License

MIT
