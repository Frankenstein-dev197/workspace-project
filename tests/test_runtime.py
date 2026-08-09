"""Tests for the runtime engine."""

import pytest

from daemon_engine.runtime.sandbox import Sandbox, SandboxConfig, ExecutionResult
from daemon_engine.runtime.virtual_computer_engine import VirtualComputerEngine


class TestSandbox:
    def test_execute_python_success(self):
        with Sandbox() as sandbox:
            result = sandbox.execute_python("print('hello from sandbox')")
            assert result.success is True
            assert "hello from sandbox" in result.stdout

    def test_execute_python_error(self):
        with Sandbox() as sandbox:
            result = sandbox.execute_python("raise ValueError('test error')")
            assert result.success is False
            assert "test error" in result.stderr

    def test_execute_python_blocked_import(self):
        config = SandboxConfig(blocked_imports=["os.system"])
        with Sandbox(config=config) as sandbox:
            result = sandbox.execute_python("import os; os.system('echo bad')")
            assert result.success is False
            assert "blocked" in result.error.lower()

    def test_execute_shell_success(self):
        with Sandbox() as sandbox:
            result = sandbox.execute_shell("echo 'shell test'")
            assert result.success is True
            assert "shell test" in result.stdout

    def test_execute_shell_dangerous(self):
        with Sandbox() as sandbox:
            result = sandbox.execute_shell("rm -rf /")
            assert result.success is False
            assert "blocked" in result.error.lower() or "dangerous" in result.error.lower()

    def test_write_and_read_file(self):
        with Sandbox() as sandbox:
            sandbox.write_file("test.txt", "file content")
            content = sandbox.read_file("test.txt")
            assert content == "file content"

    def test_list_files(self):
        with Sandbox() as sandbox:
            sandbox.write_file("a.txt", "a")
            sandbox.write_file("b.txt", "b")
            files = sandbox.list_files()
            assert "a.txt" in files
            assert "b.txt" in files

    def test_info(self):
        with Sandbox() as sandbox:
            info = sandbox.info()
            assert "workdir" in info
            assert "memory_limit_mb" in info

    def test_timeout(self):
        config = SandboxConfig(wall_time_limit=2)
        with Sandbox(config=config) as sandbox:
            result = sandbox.execute_python("import time; time.sleep(10)")
            assert result.success is False
            assert "timed out" in result.error.lower()


class TestVirtualComputerEngine:
    def test_execute_command(self):
        engine = VirtualComputerEngine()
        try:
            process = engine.execute("echo 'virtual test'")
            assert process.status == "completed"
            assert "virtual test" in process.result.stdout
        finally:
            engine.shutdown()

    def test_execute_code(self):
        engine = VirtualComputerEngine()
        try:
            process = engine.execute_code("result = 2 + 3; print(result)")
            assert process.status == "completed"
            assert "5" in process.result.stdout
        finally:
            engine.shutdown()

    def test_create_and_read_file(self):
        engine = VirtualComputerEngine()
        try:
            msg = engine.create_file("vc_test.txt", "virtual file content")
            assert "Created file" in msg
            content = engine.read_file("vc_test.txt")
            assert content == "virtual file content"
        finally:
            engine.shutdown()

    def test_list_processes(self):
        engine = VirtualComputerEngine()
        try:
            engine.execute("echo 'process 1'")
            engine.execute("echo 'process 2'")
            processes = engine.list_processes()
            assert len(processes) == 2
        finally:
            engine.shutdown()

    def test_get_process(self):
        engine = VirtualComputerEngine()
        try:
            process = engine.execute("echo 'test'")
            fetched = engine.get_process(process.pid)
            assert fetched is process
        finally:
            engine.shutdown()

    def test_system_info(self):
        engine = VirtualComputerEngine()
        try:
            info = engine.system_info()
            assert "uptime_seconds" in info
            assert "total_processes" in info
            assert info["total_processes"] == 0
        finally:
            engine.shutdown()

    def test_env_vars(self):
        engine = VirtualComputerEngine()
        try:
            engine.set_env("CUSTOM_VAR", "custom_value")
            assert engine.get_env("CUSTOM_VAR") == "custom_value"
            assert "CUSTOM_VAR" in engine.get_env_all()
        finally:
            engine.shutdown()

    def test_uptime(self):
        engine = VirtualComputerEngine()
        try:
            import time

            time.sleep(0.1)
            assert engine.uptime > 0
        finally:
            engine.shutdown()
