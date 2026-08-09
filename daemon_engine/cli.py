#!/usr/bin/env python3
"""Daemon Engine CLI: command-line interface for the agentic AI engine.

Usage:
  daemon-engine run "Build a REST API"
  daemon-engine reason "How to optimize database queries?"
  daemon-engine plan "Create a web scraper"
  daemon-engine status
  daemon-engine tool web_search query="python tutorials"
  daemon-engine exec "print('hello world')"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from daemon_engine.engine import DaemonEngine


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(engine: DaemonEngine, args: argparse.Namespace) -> int:
    print(f"\n🎯 Goal: {args.goal}\n")
    workflow = engine.run_goal(args.goal, max_tasks=args.max_tasks)
    summary = engine.orchestrator.get_workflow_summary(workflow.id)
    print("\n" + "=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"  Status:     {summary['status']}")
    print(f"  Progress:   {summary['progress']}")
    print(f"  Tasks:      {summary['completed_tasks']}/{summary['total_tasks']} completed")
    print(f"  Failed:     {summary['failed_tasks']}")
    print(f"  Results:    {summary['results_count']}")
    if summary.get("error"):
        print(f"  Error:      {summary['error']}")
    print("\nTask Tree:")
    print(summary["task_tree"])
    print("=" * 60)
    return 0 if summary["status"] == "completed" else 1


def cmd_reason(engine: DaemonEngine, args: argparse.Namespace) -> int:
    print(f"\n🧠 Reasoning about: {args.problem}\n")
    result = engine.reason(args.problem)
    print(f"Strategy: {result.strategy.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nSteps ({result.num_steps}):")
    for step in result.steps:
        print(f"  {step.step_number}. {step.thought[:100]}")
    print(f"\nConclusion:\n{result.conclusion[:500]}")
    return 0


def cmd_plan(engine: DaemonEngine, args: argparse.Namespace) -> int:
    print(f"\n📋 Planning: {args.goal}\n")
    tree = engine.plan_tasks(args.goal, max_depth=args.depth)
    print(tree)
    print(f"\nTotal tasks: {len(engine.task_planner.all_tasks())}")
    return 0


def cmd_status(engine: DaemonEngine, args: argparse.Namespace) -> int:
    status = engine.system_status()
    print("\n" + "=" * 60)
    print("DAEMON ENGINE STATUS")
    print("=" * 60)
    print(f"  LLM Model:       {status['llm_model']}")
    print(f"  Tools:           {status['tool_count']}")
    for tool in status["tools_available"]:
        print(f"    - {tool}")
    print(f"\n  Agents:")
    for key, val in status["agents"].items():
        print(f"    {key}: {val}")
    print(f"\n  Memory:")
    for subsystem, stats in status["memory"].items():
        print(f"    {subsystem}: {stats}")
    print(f"\n  Virtual Computer:")
    for key, val in status["virtual_computer"].items():
        print(f"    {key}: {val}")
    print(f"\n  Workflows:       {status['workflows']}")
    print(f"  Tasks:           {status['tasks']}")
    print("=" * 60)
    return 0


def cmd_tool(engine: DaemonEngine, args: argparse.Namespace) -> int:
    kwargs = {}
    for kv in args.params:
        if "=" in kv:
            key, value = kv.split("=", 1)
            kwargs[key] = value
    print(f"\n🔧 Tool: {args.tool_name}")
    print(f"   Params: {kwargs}\n")
    result = engine.use_tool(args.tool_name, **kwargs)
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")
    if result.error:
        print(f"Error: {result.error}")
    return 0 if result.success else 1


def cmd_exec(engine: DaemonEngine, args: argparse.Namespace) -> int:
    print(f"\n💻 Executing code:\n{args.code}\n")
    result = engine.execute_code(args.code)
    print(f"Success: {result.result.success if result.result else False}")
    if result.result:
        print(f"stdout:\n{result.result.stdout}")
        if result.result.stderr:
            print(f"stderr:\n{result.result.stderr}")
    return 0 if (result.result and result.result.success) else 1


def cmd_interact(engine: DaemonEngine, args: argparse.Namespace) -> int:
    print("\n🤖 Daemon Engine Interactive Mode")
    print("Type 'exit' or 'quit' to leave. Type 'help' for commands.\n")
    while True:
        try:
            user_input = input("daemon> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user_input.lower() == "help":
            print("Commands: run <goal>, reason <problem>, plan <goal>, status, tool <name>, exit")
            continue
        if user_input.lower() == "status":
            cmd_status(engine, args)
            continue
        parts = user_input.split(None, 1)
        cmd = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if cmd == "run" and rest:
            args.goal = rest
            cmd_run(engine, args)
        elif cmd == "reason" and rest:
            args.problem = rest
            cmd_reason(engine, args)
        elif cmd == "plan" and rest:
            args.goal = rest
            cmd_plan(engine, args)
        else:
            result = engine.run_task(user_input)
            print(f"\nStatus: {result.status.value}")
            print(f"Output: {result.output[:500]}")
            if result.error:
                print(f"Error: {result.error}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daemon Engine — A next-generation agentic AI engine",
        prog="daemon-engine",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--memory-path", type=str, default=None, help="Path for persistent memory storage")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for the engine")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run a multi-agent workflow toward a goal")
    run_parser.add_argument("goal", type=str, help="The goal to achieve")
    run_parser.add_argument("--max-tasks", type=int, default=20, help="Maximum tasks to execute")

    reason_parser = subparsers.add_parser("reason", help="Reason about a problem")
    reason_parser.add_argument("problem", type=str, help="The problem to reason about")

    plan_parser = subparsers.add_parser("plan", help="Plan tasks for a goal")
    plan_parser.add_argument("goal", type=str, help="The goal to plan for")
    plan_parser.add_argument("--depth", type=int, default=3, help="Max decomposition depth")

    subparsers.add_parser("status", help="Show engine status")

    tool_parser = subparsers.add_parser("tool", help="Execute a specific tool")
    tool_parser.add_argument("tool_name", type=str, help="Name of the tool to execute")
    tool_parser.add_argument("params", nargs="*", help="Parameters as key=value pairs")

    exec_parser = subparsers.add_parser("exec", help="Execute Python code in the sandbox")
    exec_parser.add_argument("code", type=str, help="Python code to execute")

    subparsers.add_parser("interact", help="Interactive REPL mode")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    engine = DaemonEngine(memory_path=args.memory_path, workdir=args.workdir)
    commands = {
        "run": cmd_run,
        "reason": cmd_reason,
        "plan": cmd_plan,
        "status": cmd_status,
        "tool": cmd_tool,
        "exec": cmd_exec,
        "interact": cmd_interact,
    }
    handler = commands.get(args.command)
    if handler:
        try:
            return handler(engine, args)
        finally:
            engine.shutdown()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
