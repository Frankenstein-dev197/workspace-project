"""Tests for MCP plugin system."""

import pytest

from daemon_engine.core.mcp_plugin import (
    MCPClient,
    MCPCatalog,
    MCPToolDef,
    normalize_mcp_name,
    make_mcp_tool_name,
)


class TestNormalizeMcpName:
    def test_alphanumeric(self):
        assert normalize_mcp_name("hello123") == "hello123"

    def test_dashes_underscores(self):
        assert normalize_mcp_name("hello-world_test") == "hello-world_test"

    def test_disallowed_replaced(self):
        assert normalize_mcp_name("hello.world") == "hello_world"
        assert normalize_mcp_name("hello world") == "hello_world"
        assert normalize_mcp_name("hello/world") == "hello_world"

    def test_special_chars(self):
        assert normalize_mcp_name("tool@v2!") == "tool_v2_"

    def test_empty(self):
        assert normalize_mcp_name("") == ""


class TestMakeMcpToolName:
    def test_basic(self):
        assert make_mcp_tool_name("docs", "search") == "mcp__docs__search"

    def test_normalizes(self):
        assert make_mcp_tool_name("my.server", "tool.name") == "mcp__my_server__tool_name"


class TestMCPToolDef:
    def test_creation(self):
        tool = MCPToolDef(name="search", description="Search docs")
        assert tool.name == "search"
        assert tool.read_only is True

    def test_destructive(self):
        tool = MCPToolDef(name="delete", description="Delete", read_only=False)
        assert tool.read_only is False

    def test_to_dict(self):
        tool = MCPToolDef(name="search", description="Search", input_schema={"type": "object"})
        d = tool.to_dict()
        assert d["name"] == "search"
        assert d["read_only"] is True


class TestMCPClient:
    def test_creation(self):
        client = MCPClient("docs")
        assert client.name == "docs"
        assert client.tool_count() == 0

    def test_register(self):
        client = MCPClient("docs")
        tools = [MCPToolDef(name="search", description="Search")]
        handlers = {"search": lambda query: f"results for {query}"}
        client.register(tools, handlers)
        assert client.tool_count() == 1

    def test_register_tool(self):
        client = MCPClient("docs")
        client.register_tool(
            MCPToolDef(name="search", description="Search"),
            lambda query: f"results for {query}",
        )
        assert client.tool_count() == 1
        assert client.has_tool("search") is True

    def test_call_tool(self):
        client = MCPClient("docs")
        client.register_tool(
            MCPToolDef(name="search", description="Search"),
            lambda query: f"Found: {query}",
        )
        result = client.call_tool("search", {"query": "test"})
        assert result == "Found: test"

    def test_call_unknown_tool(self):
        client = MCPClient("docs")
        result = client.call_tool("unknown", {})
        assert "unknown tool" in result

    def test_call_tool_error(self):
        client = MCPClient("docs")
        client.register_tool(
            MCPToolDef(name="fail", description="Fails"),
            lambda x: 1 / 0,
        )
        result = client.call_tool("fail", {"x": 1})
        assert "MCP error" in result

    def test_call_tool_bad_args(self):
        client = MCPClient("docs")
        client.register_tool(
            MCPToolDef(name="search", description="Search"),
            lambda query: f"results for {query}",
        )
        result = client.call_tool("search", {"wrong": "arg"})
        assert "MCP error" in result

    def test_list_tools(self):
        client = MCPClient("docs")
        client.register_tool(MCPToolDef(name="search", description=""), lambda: "")
        client.register_tool(MCPToolDef(name="get", description=""), lambda: "")
        tools = client.list_tools()
        assert "search" in tools
        assert "get" in tools

    def test_has_tool(self):
        client = MCPClient("docs")
        client.register_tool(MCPToolDef(name="search", description=""), lambda: "")
        assert client.has_tool("search") is True
        assert client.has_tool("unknown") is False


class TestMCPCatalog:
    def test_creation(self):
        catalog = MCPCatalog()
        assert catalog.server_count() == 0

    def test_connect(self):
        catalog = MCPCatalog()
        client = MCPClient("docs")
        client.register_tool(MCPToolDef(name="search", description=""), lambda query: "")
        result = catalog.connect(client)
        assert "Connected" in result
        assert catalog.server_count() == 1

    def test_connect_duplicate(self):
        catalog = MCPCatalog()
        client = MCPClient("docs")
        catalog.connect(client)
        result = catalog.connect(MCPClient("docs"))
        assert "already connected" in result

    def test_disconnect(self):
        catalog = MCPCatalog()
        catalog.connect(MCPClient("docs"))
        result = catalog.disconnect("docs")
        assert "Disconnected" in result
        assert catalog.server_count() == 0

    def test_disconnect_nonexistent(self):
        catalog = MCPCatalog()
        result = catalog.disconnect("nonexistent")
        assert "not connected" in result

    def test_get_client(self):
        catalog = MCPCatalog()
        client = MCPClient("docs")
        catalog.connect(client)
        retrieved = catalog.get_client("docs")
        assert retrieved is client

    def test_get_client_nonexistent(self):
        catalog = MCPCatalog()
        assert catalog.get_client("nonexistent") is None

    def test_list_servers(self):
        catalog = MCPCatalog()
        catalog.connect(MCPClient("docs"))
        catalog.connect(MCPClient("deploy"))
        servers = catalog.list_servers()
        assert "docs" in servers
        assert "deploy" in servers

    def test_total_tool_count(self):
        catalog = MCPCatalog()
        c1 = MCPClient("docs")
        c1.register_tool(MCPToolDef(name="search", description=""), lambda: "")
        c2 = MCPClient("deploy")
        c2.register_tool(MCPToolDef(name="trigger", description=""), lambda: "")
        c2.register_tool(MCPToolDef(name="status", description=""), lambda: "")
        catalog.connect(c1)
        catalog.connect(c2)
        assert catalog.total_tool_count() == 3

    def test_assemble_tool_pool_empty(self):
        catalog = MCPCatalog()
        tools, handlers = catalog.assemble_tool_pool()
        assert tools == []
        assert handlers == {}

    def test_assemble_tool_pool_with_builtin(self):
        catalog = MCPCatalog()
        builtin_tools = [{"name": "bash", "description": "Run bash"}]
        builtin_handlers = {"bash": lambda command: "output"}
        tools, handlers = catalog.assemble_tool_pool(builtin_tools, builtin_handlers)
        assert len(tools) == 1
        assert "bash" in handlers

    def test_assemble_tool_pool_with_mcp(self):
        catalog = MCPCatalog()
        client = MCPClient("docs")
        client.register_tool(
            MCPToolDef(name="search", description="Search docs"),
            lambda query: f"results for {query}",
        )
        catalog.connect(client)
        tools, handlers = catalog.assemble_tool_pool()
        tool_names = [t["name"] for t in tools]
        assert "mcp__docs__search" in tool_names
        assert "mcp__docs__search" in handlers
        result = handlers["mcp__docs__search"](query="test")
        assert "results for test" in result

    def test_assemble_tool_pool_mixed(self):
        catalog = MCPCatalog()
        client = MCPClient("my.server")
        client.register_tool(
            MCPToolDef(name="tool.name", description="Test"),
            lambda: "ok",
        )
        catalog.connect(client)
        builtin = [{"name": "bash", "description": "Bash"}]
        builtin_h = {"bash": lambda: "bash"}
        tools, handlers = catalog.assemble_tool_pool(builtin, builtin_h)
        names = [t["name"] for t in tools]
        assert "bash" in names
        assert "mcp__my_server__tool_name" in names

    def test_call_mcp_tool(self):
        catalog = MCPCatalog()
        client = MCPClient("docs")
        client.register_tool(
            MCPToolDef(name="search", description=""),
            lambda query: f"results for {query}",
        )
        catalog.connect(client)
        result = catalog.call_mcp_tool("docs", "search", {"query": "test"})
        assert "results for test" in result

    def test_call_mcp_tool_server_not_connected(self):
        catalog = MCPCatalog()
        result = catalog.call_mcp_tool("unknown", "search", {})
        assert "not connected" in result

    def test_clear(self):
        catalog = MCPCatalog()
        catalog.connect(MCPClient("docs"))
        catalog.connect(MCPClient("deploy"))
        catalog.clear()
        assert catalog.server_count() == 0

    def test_multiple_servers_no_name_collision(self):
        catalog = MCPCatalog()
        c1 = MCPClient("docs")
        c1.register_tool(MCPToolDef(name="search", description=""), lambda: "docs")
        c2 = MCPClient("web")
        c2.register_tool(MCPToolDef(name="search", description=""), lambda: "web")
        catalog.connect(c1)
        catalog.connect(c2)
        tools, handlers = catalog.assemble_tool_pool()
        names = [t["name"] for t in tools]
        assert "mcp__docs__search" in names
        assert "mcp__web__search" in names
        assert handlers["mcp__docs__search"]() == "docs"
        assert handlers["mcp__web__search"]() == "web"
