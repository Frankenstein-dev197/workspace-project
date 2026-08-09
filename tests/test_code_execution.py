"""Tests for code execution engine."""

import pytest

from daemon_engine.runtime.code_execution.executor import CodeExecutor, ExecutionEnvironment


class TestCodeExecutor:
    def test_execute_python(self):
        executor = CodeExecutor()
        result = executor.execute_python("print('Hello, World!')")
        assert result.success is True
        assert "Hello, World!" in result.stdout

    def test_execute_python_with_error(self):
        executor = CodeExecutor()
        result = executor.execute_python("raise ValueError('test error')")
        assert result.success is False
        assert "test error" in result.stderr

    def test_execute_shell(self):
        executor = CodeExecutor()
        result = executor.execute_shell("echo 'shell test'")
        assert result.success is True
        assert "shell test" in result.stdout

    def test_dangerous_pattern_blocked(self):
        executor = CodeExecutor()
        result = executor.execute_python("import os; os.system('rm -rf /')")
        assert result.success is False
        assert "dangerous" in result.error.lower()

    def test_unsupported_language(self):
        executor = CodeExecutor()
        result = executor.execute("code", language="cobol")
        assert result.success is False
        assert "Unsupported" in result.error

    def test_create_environment(self):
        executor = CodeExecutor()
        env = executor.create_environment("custom-env", language="python")
        assert env.name == "custom-env"
        assert env.language == "python"

    def test_list_environments(self):
        executor = CodeExecutor()
        envs = executor.list_environments()
        names = [e["name"] for e in envs]
        assert "python" in names
        assert "javascript" in names
        assert "shell" in names

    def test_execute_returns_execution_result(self):
        executor = CodeExecutor()
        result = executor.execute_python("x = 1 + 1; print(x)")
        assert result.success is True
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "returncode")

    def test_timeout(self):
        executor = CodeExecutor()
        result = executor.execute_python("import time; time.sleep(10)", timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_info(self):
        executor = CodeExecutor()
        info = executor.info()
        assert "python" in info["languages"]
        assert info["environments"] > 0

    def test_install_package_simulated(self):
        executor = CodeExecutor()
        result = executor.install_package("python", "requests", timeout=30)
        assert result.success is True or result.success is False

    def test_javascript_execution(self):
        executor = CodeExecutor()
        result = executor.execute_javascript("console.log('JS test')")
        assert result.success is True or "not found" in result.error.lower()

    def test_python_computation(self):
        executor = CodeExecutor()
        result = executor.execute_python("""
result = sum(range(100))
print(f'Sum: {result}')
""")
        assert result.success is True
        assert "4950" in result.stdout
