"""MCP plugin system: Model Context Protocol tool discovery and invocation.

Integrates learn-claude-code s19 MCP plugin pattern:
- MCPClient: discovers and calls tools on an MCP server
- MCPCatalog: registry of connected MCP clients
- Tool naming: mcp__{server}__{tool} with name normalization
- assemble_tool_pool: combines builtin + MCP tools into one pool
- Tool annotations: readOnly/destructive classification
- Handler-based dispatch: per-tool callable handlers
- Mock server support: factory functions for testing

MCP (Model Context Protocol) lets agents dynamically connect to external
tool servers and discover their tools at runtime, extending capabilities
without modifying the agent core.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    """Replace non [a-zA-Z0-9_-] characters with underscore."""
    return _DISALLOWED_CHARS.sub("_", name)


def make_mcp_tool_name(server: str, tool: str) -> str:
    """Create the mcp__{server}__{tool} prefixed name."""
    return f"mcp__{normalize_mcp_name(server)}__{normalize_mcp_name(tool)}"


@dataclass
class MCPToolDef:
    """Definition of a tool provided by an MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "read_only": self.read_only,
        }


class MCPClient:
    """Discovers and calls tools on an MCP server.

    A client connects to a server, discovers its tool definitions,
    and dispatches calls to registered handlers.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: list[MCPToolDef] = []
        self._handlers: dict[str, Callable[..., str]] = {}

    def register(
        self,
        tool_defs: list[MCPToolDef],
        handlers: dict[str, Callable[..., str]],
    ) -> None:
        """Register tool definitions and their handlers."""
        self.tools = tool_defs
        self._handlers = handlers
        logger.info(
            "MCP server '%s' registered %d tools: %s",
            self.name,
            len(tool_defs),
            [t.name for t in tool_defs],
        )

    def register_tool(
        self,
        tool_def: MCPToolDef,
        handler: Callable[..., str],
    ) -> None:
        """Register a single tool."""
        self.tools.append(tool_def)
        self._handlers[tool_def.name] = handler

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Call a tool by name with the given arguments."""
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return handler(**args)
        except TypeError as e:
            return f"MCP error: bad arguments: {e}"
        except Exception as e:
            return f"MCP error: {e}"

    def list_tools(self) -> list[str]:
        """List tool names."""
        return [t.name for t in self.tools]

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._handlers

    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self.tools)


class MCPCatalog:
    """Registry of connected MCP clients (servers).

    Manages multiple MCP server connections and assembles their tools
    into a unified tool pool with mcp__{server}__{tool} naming.
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    def connect(self, client: MCPClient) -> str:
        """Connect to an MCP server (register its client)."""
        if client.name in self._clients:
            return f"MCP server '{client.name}' already connected"
        self._clients[client.name] = client
        tool_names = client.list_tools()
        logger.info(
            "MCP connected: %s → %s",
            client.name,
            tool_names,
        )
        return (
            f"Connected to MCP server '{client.name}'. "
            f"Discovered {client.tool_count()} tools: {', '.join(tool_names)}"
        )

    def disconnect(self, name: str) -> str:
        """Disconnect from an MCP server."""
        if name not in self._clients:
            return f"MCP server '{name}' not connected"
        self._clients.pop(name)
        return f"Disconnected from MCP server '{name}'"

    def get_client(self, name: str) -> MCPClient | None:
        """Get a connected client by server name."""
        return self._clients.get(name)

    def list_servers(self) -> list[str]:
        """List connected server names."""
        return list(self._clients.keys())

    def server_count(self) -> int:
        """Number of connected servers."""
        return len(self._clients)

    def total_tool_count(self) -> int:
        """Total tools across all servers."""
        return sum(c.tool_count() for c in self._clients.values())

    def assemble_tool_pool(
        self,
        builtin_tools: list[dict[str, Any]] | None = None,
        builtin_handlers: dict[str, Callable[..., str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]]]:
        """Assemble builtin + MCP tools into one pool.

        Returns (tools, handlers) where tools is a list of tool defs
        and handlers maps tool names to callables.
        """
        tools = list(builtin_tools or [])
        handlers = dict(builtin_handlers or {})

        for server_name, client in self._clients.items():
            safe_server = normalize_mcp_name(server_name)
            for tool_def in client.tools:
                safe_tool = normalize_mcp_name(tool_def.name)
                prefixed = f"mcp__{safe_server}__{safe_tool}"
                tools.append({
                    "name": prefixed,
                    "description": tool_def.description,
                    "input_schema": tool_def.input_schema,
                })
                handlers[prefixed] = _make_mcp_handler(client, tool_def.name)

        return tools, handlers

    def call_mcp_tool(self, server: str, tool: str, args: dict[str, Any]) -> str:
        """Call a tool on a specific MCP server."""
        client = self._clients.get(server)
        if not client:
            return f"MCP error: server '{server}' not connected"
        return client.call_tool(tool, args)

    def clear(self) -> None:
        """Disconnect all servers."""
        self._clients.clear()


def _make_mcp_handler(
    client: MCPClient,
    tool_name: str,
) -> Callable[..., str]:
    """Create a handler closure for an MCP tool."""
    def handler(**kwargs: Any) -> str:
        return client.call_tool(tool_name, kwargs)
    return handler
