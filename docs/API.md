# API Reference

## DaemonEngine

The top-level facade connecting all subsystems.

### Constructor

```python
DaemonEngine(
    llm: BaseLLM | None = None,
    memory_path: str | Path | None = None,
    workdir: str | Path | None = None,
    config: dict[str, Any] | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `run_task(description, agent_config=None)` | Run a single-agent task |
| `run_goal(goal, max_tasks=20)` | Execute a multi-agent workflow |
| `reason(problem, strategy=CoT)` | Reason about a problem |
| `decide(options, context=None, strategy=None)` | Make a decision |
| `plan_tasks(goal, max_depth=3)` | Plan and return task tree |
| `create_agent(role="worker", config=None)` | Create a new agent |
| `execute_code(code, timeout=None)` | Execute Python in sandbox |
| `execute_command(command, timeout=None)` | Execute shell command |
| `use_tool(tool_name, **kwargs)` | Execute a specific tool |
| `remember(content, memory_type="episodic")` | Store a memory |
| `recall(query, limit=5)` | Retrieve memories |
| `save_state()` | Persist memory to disk |
| `system_status()` | Get full system status dict |
| `shutdown()` | Clean shutdown |

## Core Engine

### Agent (`daemon_engine.core.agent_engine`)

```python
Agent(config=AgentConfig(), llm=BaseLLM, tool_registry=ToolRegistry, memory=UnifiedMemory)
```

| Method | Description |
|--------|-------------|
| `run(task) -> AgentResult` | Execute the think-act-observe loop |
| `cancel()` | Cancel the agent |
| `reset()` | Reset state and history |

### AgentConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | "default-agent" | Agent name |
| `description` | str | — | Agent description |
| `system_prompt` | str | — | System prompt for LLM |
| `model` | str \| None | None | Model override |
| `max_turns` | int | 25 | Maximum think-act-observe iterations |
| `timeout_seconds` | int | 300 | Execution timeout |
| `tools` | list[str] | [] | Allowed tool names |
| `memory_enabled` | bool | True | Enable memory integration |
| `reasoning_enabled` | bool | True | Enable reasoning |

### AgentResult

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | str | Agent UUID |
| `task_id` | str | Task UUID |
| `status` | AgentState | Final state |
| `output` | str | Agent output |
| `steps` | list[AgentStep] | Step history |
| `error` | str \| None | Error message |
| `duration` | float | Execution time |

### ReasoningEngine (`daemon_engine.core.reasoning_engine`)

```python
ReasoningEngine(llm=BaseLLM)
```

| Method | Description |
|--------|-------------|
| `reason(problem, strategy=CoT, context=None) -> ReasoningResult` | Reason about a problem |
| `compare_strategies(problem) -> dict` | Compare all strategies |

### ReasoningStrategy

| Value | Description |
|-------|-------------|
| `CHAIN_OF_THOUGHT` | Linear step-by-step reasoning |
| `REACT` | Reasoning + acting framework |
| `TREE_OF_THOUGHT` | Multi-path exploration |
| `REFLECTION` | Propose, critique, refine |
| `SELF_CONSISTENCY` | Multiple samples aggregated |

### TaskPlanner (`daemon_engine.core.task_planner`)

```python
TaskPlanner(llm=BaseLLM)
```

| Method | Description |
|--------|-------------|
| `create_task(description, parent_id=None, priority=MEDIUM) -> Task` | Create a task |
| `decompose(goal, parent_id=None, max_depth=3) -> Task` | Decompose a goal |
| `get_task(task_id) -> Task \| None` | Get a task by ID |
| `get_subtasks(task_id) -> list[Task]` | Get child tasks |
| `get_ready_tasks() -> list[Task]` | Get executable tasks |
| `update_status(task_id, status, result="") -> bool` | Update task status |
| `assign_agent(task_id, agent_id) -> bool` | Assign an agent |
| `get_task_tree(task_id=None) -> str` | Get formatted task tree |
| `all_tasks() -> list[Task]` | Get all tasks |

### DecisionSystem (`daemon_engine.core.decision_system`)

```python
DecisionSystem(llm=BaseLLM, strategy=UTILITY_BASED, rules=None)
```

| Method | Description |
|--------|-------------|
| `decide(options, context=None, strategy=None) -> Decision` | Make a decision |
| `add_rule(rule) -> None` | Add a decision rule |
| `set_utility_weight(factor, weight) -> None` | Set utility weight |
| `get_history() -> list[Decision]` | Get decision history |
| `clear_history() -> None` | Clear history |

## Multi-Agent System

### Orchestrator (`daemon_engine.multi_agent.orchestrator`)

| Method | Description |
|--------|-------------|
| `execute_goal(goal, max_tasks=20) -> Workflow` | Execute a multi-agent workflow |
| `get_workflow(workflow_id) -> Workflow \| None` | Get a workflow |
| `list_workflows() -> list[Workflow]` | List all workflows |
| `cancel_workflow(workflow_id) -> bool` | Cancel a workflow |
| `get_workflow_summary(workflow_id) -> dict` | Get workflow summary |

### AgentManager (`daemon_engine.multi_agent.agent_manager`)

| Method | Description |
|--------|-------------|
| `spawn_agent(role="worker", config=None) -> Agent` | Create and register an agent |
| `get_agent(agent_id) -> Agent \| None` | Get an agent |
| `get_agents_by_role(role) -> list[Agent]` | Get agents by role |
| `get_available_agent(role=None) -> Agent \| None` | Get an idle agent |
| `update_status(agent_id, status) -> None` | Update agent status |
| `record_completion(agent_id, success) -> None` | Record task completion |
| `terminate_agent(agent_id) -> bool` | Terminate an agent |
| `health_check() -> dict` | Get fleet health |
| `terminate_all() -> None` | Terminate all agents |

### CommunicationSystem (`daemon_engine.multi_agent.communication_system`)

| Method | Description |
|--------|-------------|
| `register_agent(agent_id) -> None` | Register an agent |
| `send(message) -> bool` | Send a message |
| `receive(agent_id, timeout=0.1) -> Message \| None` | Receive a message |
| `subscribe(agent_id, callback) -> None` | Subscribe to messages |
| `set_shared_state(key, value) -> None` | Set shared state |
| `get_shared_state(key, default=None) -> Any` | Get shared state |
| `get_message_log(agent_id=None) -> list[Message]` | Get message log |

## Memory System

### UnifiedMemory (`daemon_engine.memory.unified`)

| Method | Description |
|--------|-------------|
| `store(content, memory_type="episodic", **kwargs) -> str` | Store a memory |
| `recall(query, limit=5) -> str` | Recall memories |
| `search_all(query, limit=5) -> dict` | Search all layers |
| `save_all() -> None` | Persist to disk |
| `consolidate() -> dict` | Prune low-importance |
| `stats() -> dict` | Get memory stats |
| `clear() -> None` | Clear all memory |

### Memory Types

| Type | Description |
|------|-------------|
| `code` | Code artifacts (stored in CodeMemory) |
| `knowledge` | Knowledge nodes (stored in KnowledgeMemory) |
| `episodic` | Event-based memories (LongTermMemory) |
| `semantic` | Factual knowledge (LongTermMemory) |
| `procedural` | How-to/skills (LongTermMemory) |
| `feedback` | User corrections (LongTermMemory) |
| `pattern` | Learned patterns (LongTermMemory) |

## Tool System

### ToolRegistry (`daemon_engine.tools.tool_registry`)

| Method | Description |
|--------|-------------|
| `register(name, description, handler, category, ...)` | Register a tool |
| `unregister(name) -> bool` | Unregister a tool |
| `list_tools() -> list[str]` | List all tool names |
| `list_by_category(category) -> list[str]` | List tools by category |
| `get_descriptions() -> dict` | Get all descriptions |
| `execute(tool_name, **kwargs) -> ToolResult` | Execute a tool |

### Available Tools

| Tool | Category | Description |
|------|----------|-------------|
| `web_fetch` | browser | Fetch web page content |
| `web_search` | browser | Search the web |
| `extract_links` | browser | Extract links from page |
| `extract_text` | browser | Extract text from page |
| `browser_navigate` | browser | Navigate to URL |
| `web_scraper` | research | Scrape structured data |
| `osint_lookup` | research | Username OSINT lookup |
| `data_extract` | research | Extract emails/URLs/phones |
| `summarize_url` | research | Summarize a URL |
| `fact_check` | research | Fact-check a claim |
| `bash` | devops | Execute shell command |
| `file_read` | devops | Read a file |
| `file_write` | devops | Write a file |
| `file_list` | devops | List directory |
| `docker_build` | devops | Build Docker image |
| `docker_run` | devops | Run Docker container |
| `ansible_playbook` | devops | Run Ansible playbook |
| `run_pipeline` | automation | Execute multi-step pipeline |
| `schedule_task` | automation | Schedule a task |
| `deploy_application` | automation | Deploy an app |
| `run_tests` | automation | Run test suite |
| `git_operations` | automation | Git operations |

## Runtime Engine

### Sandbox (`daemon_engine.runtime.sandbox`)

| Method | Description |
|--------|-------------|
| `execute_python(code, timeout=None) -> ExecutionResult` | Execute Python |
| `execute_shell(command, timeout=None) -> ExecutionResult` | Execute shell |
| `write_file(name, content) -> Path` | Write file in sandbox |
| `read_file(name) -> str` | Read file from sandbox |
| `list_files() -> list[str]` | List sandbox files |
| `cleanup() -> None` | Clean up sandbox |
| `info() -> dict` | Get sandbox info |

### VirtualComputerEngine (`daemon_engine.runtime.virtual_computer_engine`)

| Method | Description |
|--------|-------------|
| `execute(command, timeout=None) -> VirtualProcess` | Execute command |
| `execute_code(code, timeout=None) -> VirtualProcess` | Execute Python |
| `create_file(path, content) -> str` | Create file |
| `read_file(path) -> str` | Read file |
| `list_directory(path=".") -> list[str]` | List directory |
| `get_process(pid) -> VirtualProcess \| None` | Get process by PID |
| `list_processes(status=None) -> list[VirtualProcess]` | List processes |
| `kill_process(pid) -> bool` | Kill a process |
| `set_env(key, value) -> None` | Set environment variable |
| `get_env(key) -> str \| None` | Get environment variable |
| `system_info() -> dict` | Get system info |
| `shutdown() -> None` | Shutdown |

## Model Integration

### BaseLLM (`daemon_engine.models.base`)

| Method | Description |
|--------|-------------|
| `chat(messages, **kwargs) -> str` | Chat completion |
| `embed(text) -> list[float]` | Generate embedding |
| `stream(messages, **kwargs)` | Stream completion |

### Providers

| Provider | Class | API Key Env |
|----------|-------|-------------|
| Mock | `MockProvider` | None |
| OpenAI | `OpenAIProvider` | `OPENAI_API_KEY` |
| Anthropic | `AnthropicProvider` | `ANTHROPIC_API_KEY` |
| Local | `LocalProvider` | None (Ollama/Transformers) |

### LLMConfig

| Field | Type | Default |
|-------|------|---------|
| `provider` | str | "mock" |
| `model` | str | "gpt-4" |
| `api_key` | str \| None | None |
| `base_url` | str \| None | None |
| `temperature` | float | 0.7 |
| `max_tokens` | int | 4096 |
| `timeout` | int | 60 |
