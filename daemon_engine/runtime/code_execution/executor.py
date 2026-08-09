"""Code execution engine: multi-language sandboxed code execution.

Integrates patterns from:
- DeepSeek-Reasonix's sandbox (OS-level jail, write-root confinement)
- learn-claude-code's bash tool (dangerous command blocking)
- Firecracker's isolation model (separate execution environments)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daemon_engine.runtime.sandbox import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionEnvironment:
    """A named execution environment with its own working directory."""
    name: str
    workdir: Path
    language: str = "python"
    env_vars: dict[str, str] = field(default_factory=dict)
    packages: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def set_env(self, key: str, value: str) -> None:
        self.env_vars[key] = value

    def install_package(self, name: str) -> None:
        if name not in self.packages:
            self.packages.append(name)


class CodeExecutor:
    """Multi-language code executor with environment management.

    Supports Python, JavaScript (Node), Shell, and can be extended.
    Each language gets its own execution environment with isolation.
    """

    DANGEROUS_PATTERNS = [
        "rm -rf /", "sudo ", "shutdown", "reboot", "mkfs",
        "dd if=/dev/zero", "> /dev/sda", ":(){:|:&};:",
        "import os; os.system", "subprocess.call('rm'",
    ]

    LANGUAGE_CONFIG = {
        "python": {
            "extension": ".py",
            "runner": ["python3"],
            "comment": "#",
        },
        "javascript": {
            "extension": ".js",
            "runner": ["node"],
            "comment": "//",
        },
        "shell": {
            "extension": ".sh",
            "runner": ["bash"],
            "comment": "#",
        },
        "ruby": {
            "extension": ".rb",
            "runner": ["ruby"],
            "comment": "#",
        },
        "go": {
            "extension": ".go",
            "runner": ["go", "run"],
            "comment": "//",
        },
    }

    def __init__(self, base_workdir: str | Path | None = None) -> None:
        self.base_workdir = Path(base_workdir) if base_workdir else Path(tempfile.mkdtemp(prefix="daemon_exec_"))
        self.base_workdir.mkdir(parents=True, exist_ok=True)
        self._environments: dict[str, ExecutionEnvironment] = {}
        self._create_default_environments()

    def _create_default_environments(self) -> None:
        for lang in self.LANGUAGE_CONFIG:
            env_dir = self.base_workdir / lang
            env_dir.mkdir(parents=True, exist_ok=True)
            self._environments[lang] = ExecutionEnvironment(
                name=lang,
                workdir=env_dir,
                language=lang,
            )

    def get_environment(self, name: str) -> ExecutionEnvironment | None:
        return self._environments.get(name)

    def create_environment(self, name: str, language: str = "python") -> ExecutionEnvironment:
        env_dir = self.base_workdir / name
        env_dir.mkdir(parents=True, exist_ok=True)
        env = ExecutionEnvironment(name=name, workdir=env_dir, language=language)
        self._environments[name] = env
        return env

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        env_name: str | None = None,
        stdin: str | None = None,
    ) -> ExecutionResult:
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                logger.warning("Blocked dangerous code pattern: %s", pattern)
                return ExecutionResult(
                    success=False,
                    error=f"Blocked dangerous pattern: {pattern}",
                )
        lang_config = self.LANGUAGE_CONFIG.get(language)
        if not lang_config:
            return ExecutionResult(
                success=False,
                error=f"Unsupported language: {language}. Supported: {list(self.LANGUAGE_CONFIG.keys())}",
            )
        env = self._environments.get(env_name or language, self._environments[language])
        script_path = env.workdir / f"script_{int(time.time() * 1000)}{lang_config['extension']}"
        script_path.write_text(code)
        cmd = lang_config["runner"] + [str(script_path)]
        start = time.time()
        try:
            run_env = dict(os.environ)
            run_env.update(env.env_vars)
            run_env["PYTHONPATH"] = str(env.workdir)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(env.workdir),
                env=run_env,
                input=stdin,
            )
            duration = time.time() - start
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:10000],
                returncode=result.returncode,
                duration=duration,
                files_created=self._get_created_files(env.workdir, script_path),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                duration=time.time() - start,
            )
        except FileNotFoundError as exc:
            return ExecutionResult(
                success=False,
                error=f"Runtime not found: {exc}. Is {language} installed?",
                duration=time.time() - start,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=str(exc),
                duration=time.time() - start,
            )

    def execute_python(self, code: str, timeout: int = 30, **kwargs: Any) -> ExecutionResult:
        return self.execute(code, language="python", timeout=timeout, **kwargs)

    def execute_javascript(self, code: str, timeout: int = 30, **kwargs: Any) -> ExecutionResult:
        return self.execute(code, language="javascript", timeout=timeout, **kwargs)

    def execute_shell(self, code: str, timeout: int = 30, **kwargs: Any) -> ExecutionResult:
        return self.execute(code, language="shell", timeout=timeout, **kwargs)

    def execute_file(self, file_path: str | Path, language: str | None = None, timeout: int = 30) -> ExecutionResult:
        path = Path(file_path)
        if not path.exists():
            return ExecutionResult(success=False, error=f"File not found: {path}")
        if language is None:
            ext = path.suffix.lower()
            lang_map = {v["extension"]: k for k, v in self.LANGUAGE_CONFIG.items()}
            language = lang_map.get(ext, "python")
        code = path.read_text()
        return self.execute(code, language=language, timeout=timeout)

    def install_package(self, language: str, package: str, timeout: int = 60) -> ExecutionResult:
        env = self._environments.get(language)
        if not env:
            return ExecutionResult(success=False, error=f"No environment for {language}")
        installers = {
            "python": ["pip", "install"],
            "javascript": ["npm", "install"],
            "ruby": ["gem", "install"],
            "go": ["go", "get"],
        }
        installer = installers.get(language)
        if not installer:
            return ExecutionResult(success=False, error=f"No installer for {language}")
        cmd = installer + [package]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(env.workdir),
            )
            if result.returncode == 0:
                env.install_package(package)
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:5000],
                stderr=result.stderr[:5000],
                returncode=result.returncode,
            )
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))

    def run_tests(self, test_path: str | Path, framework: str = "pytest", timeout: int = 120) -> ExecutionResult:
        path = Path(test_path)
        frameworks = {
            "pytest": ["python3", "-m", "pytest", "--tb=short", "-q"],
            "unittest": ["python3", "-m", "unittest"],
            "jest": ["npx", "jest"],
            "mocha": ["npx", "mocha"],
            "rspec": ["rspec"],
            "go": ["go", "test"],
        }
        runner = frameworks.get(framework)
        if not runner:
            return ExecutionResult(success=False, error=f"Unknown test framework: {framework}")
        cmd = runner + [str(path)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(path.parent) if path.is_file() else str(path),
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:10000],
                returncode=result.returncode,
            )
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))

    def _get_created_files(self, workdir: Path, script_path: Path) -> list[str]:
        files: list[str] = []
        if workdir.exists():
            for item in workdir.iterdir():
                if item != script_path and item.is_file() and not item.name.startswith("."):
                    files.append(item.name)
        return sorted(files)

    def list_environments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": env.name,
                "language": env.language,
                "workdir": str(env.workdir),
                "packages": env.packages,
                "env_vars": len(env.env_vars),
            }
            for env in self._environments.values()
        ]

    def cleanup(self) -> None:
        for env in self._environments.values():
            try:
                for item in env.workdir.iterdir():
                    if item.is_file() and item.name.startswith("script_"):
                        item.unlink()
            except Exception:
                pass

    def info(self) -> dict[str, Any]:
        return {
            "base_workdir": str(self.base_workdir),
            "languages": list(self.LANGUAGE_CONFIG.keys()),
            "environments": len(self._environments),
            "dangerous_patterns": len(self.DANGEROUS_PATTERNS),
        }
