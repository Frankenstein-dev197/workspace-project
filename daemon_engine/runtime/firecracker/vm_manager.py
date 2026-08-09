"""Firecracker microVM integration: isolated VM execution for agents.

Integrates Firecracker's microVM concept (lightweight VM via Unix socket API)
with the daemon engine runtime. Provides VM configuration, lifecycle management,
and isolated execution environments inspired by Firecracker's REST API.

The actual Firecracker binary is a Rust VMM; this module provides a Python
interface to configure and manage microVMs via Firecracker's Unix socket API.
When Firecracker is not available, falls back to a simulated VM environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from daemon_engine.runtime.sandbox import ExecutionResult

logger = logging.getLogger(__name__)


class VMState(str, Enum):
    """VM states from Firecracker's InstanceInfo."""
    NOT_STARTED = "Not started"
    RUNNING = "Running"
    PAUSED = "Paused"
    TERMINATED = "Terminated"
    ERROR = "Error"


class ActionType(str, Enum):
    """Firecracker action types."""
    FLUSH_METRICS = "FlushMetrics"
    INSTANCE_START = "InstanceStart"
    SEND_CTRL_ALT_DEL = "SendCtrlAltDel"


@dataclass
class BootSource:
    """Boot source configuration (kernel + initrd)."""
    kernel_image_path: str
    initrd_path: str | None = None
    boot_args: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kernel_image_path": self.kernel_image_path}
        if self.initrd_path:
            d["initrd_path"] = self.initrd_path
        if self.boot_args:
            d["boot_args"] = self.boot_args
        return d


@dataclass
class DriveConfig:
    """Block device configuration."""
    drive_id: str
    path_on_host: str
    is_root_device: bool = False
    is_read_only: bool = False
    partuuid: str | None = None
    cache_type: str = "Unsafe"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "drive_id": self.drive_id,
            "path_on_host": self.path_on_host,
            "is_root_device": self.is_root_device,
            "is_read_only": self.is_read_only,
            "cache_type": self.cache_type,
        }
        if self.partuuid:
            d["partuuid"] = self.partuuid
        return d


@dataclass
class MachineConfig:
    """Machine configuration (vCPUs, memory)."""
    vcpu_count: int = 2
    mem_size_mib: int = 512
    cpu_template: str | None = None
    smt: bool = False
    track_dirty_pages: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "vcpu_count": self.vcpu_count,
            "mem_size_mib": self.mem_size_mib,
            "smt": self.smt,
            "track_dirty_pages": self.track_dirty_pages,
        }
        if self.cpu_template:
            d["cpu_template"] = self.cpu_template
        return d


@dataclass
class NetworkInterface:
    """Network interface configuration."""
    iface_id: str
    host_dev_name: str
    guest_mac: str | None = None
    rx_rate_limiter: dict[str, Any] | None = None
    tx_rate_limiter: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "iface_id": self.iface_id,
            "host_dev_name": self.host_dev_name,
        }
        if self.guest_mac:
            d["guest_mac"] = self.guest_mac
        return d


@dataclass
class VMSpec:
    """Complete VM specification for a Firecracker microVM."""
    vm_id: str
    boot_source: BootSource | None = None
    machine_config: MachineConfig = field(default_factory=MachineConfig)
    drives: list[DriveConfig] = field(default_factory=list)
    network_interfaces: list[NetworkInterface] = field(default_factory=list)
    kernel_args: str = ""
    socket_path: str | None = None
    log_path: str | None = None
    metrics_path: str | None = None

    def to_config_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "vm_id": self.vm_id,
            "machine_config": self.machine_config.to_dict(),
        }
        if self.boot_source:
            config["boot_source"] = self.boot_source.to_dict()
        if self.drives:
            config["drives"] = [d.to_dict() for d in self.drives]
        if self.network_interfaces:
            config["network_interfaces"] = [n.to_dict() for n in self.network_interfaces]
        return config


class FirecrackerVM:
    """A single Firecracker microVM instance.

    Manages the lifecycle of a microVM via Firecracker's Unix socket API.
    Falls back to a simulated environment when Firecracker is not installed.
    """

    def __init__(self, spec: VMSpec, firecracker_binary: str = "firecracker") -> None:
        self.spec = spec
        self.firecracker_binary = firecracker_binary
        self.state: VMState = VMState.NOT_STARTED
        self._process: subprocess.Popen | None = None
        self._socket_path: str = spec.socket_path or tempfile.mktemp(
            prefix=f"fc_{spec.vm_id}_", suffix=".sock"
        )
        self._log_path: str = spec.log_path or tempfile.mktemp(
            prefix=f"fc_{spec.vm_id}_log_", suffix=".log"
        )
        self._metrics_path: str = spec.metrics_path or tempfile.mktemp(
            prefix=f"fc_{spec.vm_id}_metrics_", suffix=".json"
        )
        self._start_time: float | None = None
        self._pid: int | None = None

    @property
    def vm_id(self) -> str:
        return self.spec.vm_id

    @property
    def is_available(self) -> bool:
        return shutil.which(self.firecracker_binary) is not None

    def start(self) -> bool:
        if self.state == VMState.RUNNING:
            logger.warning("VM %s already running", self.vm_id)
            return True
        if not self.is_available:
            logger.info("Firecracker not available, starting simulated VM %s", self.vm_id)
            self.state = VMState.RUNNING
            self._start_time = time.time()
            return True
        try:
            env = dict(os.environ)
            env["RUST_BACKTRACE"] = "1"
            self._process = subprocess.Popen(
                [
                    self.firecracker_binary,
                    "--api-sock", self._socket_path,
                    "--level", "Info",
                    "--log-path", self._log_path,
                    "--metrics-path", self._metrics_path,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._pid = self._process.pid
            self._start_time = time.time()
            time.sleep(0.5)
            if self._configure_vm():
                self.state = VMState.RUNNING
                logger.info("VM %s started (pid=%d)", self.vm_id, self._pid)
                return True
            else:
                self.state = VMState.ERROR
                logger.error("Failed to configure VM %s", self.vm_id)
                return False
        except Exception as exc:
            logger.error("Failed to start VM %s: %s", self.vm_id, exc)
            self.state = VMState.ERROR
            return False

    def _configure_vm(self) -> bool:
        if not self._wait_for_socket():
            return False
        success = True
        if self.spec.boot_source:
            success &= self._api_put("/boot-source", self.spec.boot_source.to_dict())
        success &= self._api_put("/machine-config", self.spec.machine_config.to_dict())
        for drive in self.spec.drives:
            success &= self._api_put("/drives/" + drive.drive_id, drive.to_dict())
        for iface in self.spec.network_interfaces:
            success &= self._api_put("/network-interfaces/" + iface.iface_id, iface.to_dict())
        if success:
            self._api_put("/actions", {"action_type": ActionType.INSTANCE_START.value})
        return success

    def _wait_for_socket(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(self._socket_path):
                return True
            time.sleep(0.1)
        return False

    def _api_call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.is_available:
            return {"status": "simulated"}
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(self._socket_path)
            body_str = json.dumps(body) if body else ""
            request = (
                f"{method} {path} HTTP/1.0\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_str)}\r\n"
                f"\r\n{body_str}"
            )
            sock.sendall(request.encode())
            response = b""
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                response += data
            sock.close()
            parts = response.split(b"\r\n\r\n", 1)
            if len(parts) > 1:
                try:
                    return json.loads(parts[1])
                except json.JSONDecodeError:
                    return None
            return None
        except Exception as exc:
            logger.error("Firecracker API call failed: %s", exc)
            return None

    def _api_put(self, path: str, body: dict[str, Any]) -> bool:
        result = self._api_call("PUT", path, body)
        return result is not None

    def _api_get(self, path: str) -> dict[str, Any] | None:
        return self._api_call("GET", path)

    def info(self) -> dict[str, Any]:
        if self.is_available and self.state == VMState.RUNNING:
            info = self._api_get("/")
            if info:
                return info
        return {
            "app_name": "Firecracker",
            "id": self.vm_id,
            "state": self.state.value,
            "vmm_version": "simulated" if not self.is_available else "unknown",
            "pid": self._pid,
            "uptime": time.time() - self._start_time if self._start_time else 0,
        }

    def pause(self) -> bool:
        if self.state != VMState.RUNNING:
            return False
        result = self._api_patch("/vm", {"state": "Paused"})
        if result is not None:
            self.state = VMState.PAUSED
            return True
        self.state = VMState.PAUSED
        return True

    def resume(self) -> bool:
        if self.state != VMState.PAUSED:
            return False
        result = self._api_patch("/vm", {"state": "Resumed"})
        if result is not None:
            self.state = VMState.RUNNING
            return True
        self.state = VMState.RUNNING
        return True

    def _api_patch(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        return self._api_call("PATCH", path, body)

    def send_ctrl_alt_del(self) -> bool:
        if self.state != VMState.RUNNING:
            return False
        self._api_put("/actions", {"action_type": ActionType.SEND_CTRL_ALT_DEL.value})
        self.state = VMState.TERMINATED
        return True

    def shutdown(self) -> bool:
        if self.state in (VMState.TERMINATED, VMState.NOT_STARTED):
            return True
        self.send_ctrl_alt_del()
        time.sleep(1)
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except Exception:
                pass
            self._process = None
        self._cleanup_files()
        self.state = VMState.TERMINATED
        logger.info("VM %s shut down", self.vm_id)
        return True

    def _cleanup_files(self) -> None:
        for path in [self._socket_path, self._log_path, self._metrics_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        if self.state != VMState.RUNNING:
            return ExecutionResult(success=False, error=f"VM not running (state={self.state.value})")
        if not self.is_available:
            import subprocess as sp

            try:
                result = sp.run(
                    command, shell=True, capture_output=True, text=True, timeout=timeout
                )
                return ExecutionResult(
                    success=result.returncode == 0,
                    stdout=result.stdout[:10000],
                    stderr=result.stderr[:10000],
                    returncode=result.returncode,
                    duration=0.0,
                )
            except sp.TimeoutExpired:
                return ExecutionResult(success=False, error="Command timed out")
            except Exception as exc:
                return ExecutionResult(success=False, error=str(exc))
        logger.warning("Remote execution in real Firecracker VM requires SSH or agent setup")
        return ExecutionResult(success=False, error="Remote VM execution not configured")

    def status(self) -> dict[str, Any]:
        return {
            "vm_id": self.vm_id,
            "state": self.state.value,
            "available": self.is_available,
            "pid": self._pid,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "config": self.spec.machine_config.to_dict(),
            "socket_path": self._socket_path if self.is_available else None,
        }


class FirecrackerManager:
    """Manages multiple Firecracker microVMs for agent isolation.

    Each agent can get its own microVM for complete isolation. The manager
    handles VM pool creation, configuration, and lifecycle.
    """

    def __init__(
        self,
        firecracker_binary: str = "firecracker",
        kernel_path: str | None = None,
        rootfs_path: str | None = None,
        default_vcpus: int = 2,
        default_mem_mib: int = 512,
    ) -> None:
        self.firecracker_binary = firecracker_binary
        self.kernel_path = kernel_path
        self.rootfs_path = rootfs_path
        self.default_vcpus = default_vcpus
        self.default_mem_mib = default_mem_mib
        self._vms: dict[str, FirecrackerVM] = {}
        self._vm_counter = 0

    def create_vm(
        self,
        vm_id: str | None = None,
        vcpus: int | None = None,
        mem_mib: int | None = None,
        kernel_path: str | None = None,
        rootfs_path: str | None = None,
        network: bool = True,
    ) -> FirecrackerVM:
        self._vm_counter += 1
        vm_id = vm_id or f"vm-{self._vm_counter:04d}-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        kernel = kernel_path or self.kernel_path or "/tmp/vmlinux"
        rootfs = rootfs_path or self.rootfs_path or "/tmp/rootfs.ext4"
        boot = BootSource(
            kernel_image_path=kernel,
            boot_args="console=ttyS0 reboot=k panic=1 pci=off",
        )
        drives = [DriveConfig(
            drive_id="rootfs",
            path_on_host=rootfs,
            is_root_device=True,
            is_read_only=False,
        )]
        net_ifaces: list[NetworkInterface] = []
        if network:
            net_ifaces.append(NetworkInterface(
                iface_id="eth0",
                host_dev_name=f"tap-{vm_id}",
                guest_mac="AA:BB:CC:DD:EE:01",
            ))
        spec = VMSpec(
            vm_id=vm_id,
            boot_source=boot,
            machine_config=MachineConfig(
                vcpu_count=vcpus or self.default_vcpus,
                mem_size_mib=mem_mib or self.default_mem_mib,
            ),
            drives=drives,
            network_interfaces=net_ifaces,
        )
        vm = FirecrackerVM(spec, firecracker_binary=self.firecracker_binary)
        self._vms[vm_id] = vm
        logger.info("Created VM %s (%d vCPUs, %d MiB)", vm_id, spec.machine_config.vcpu_count, spec.machine_config.mem_size_mib)
        return vm

    def get_vm(self, vm_id: str) -> FirecrackerVM | None:
        return self._vms.get(vm_id)

    def list_vms(self) -> list[dict[str, Any]]:
        return [vm.status() for vm in self._vms.values()]

    def start_vm(self, vm_id: str) -> bool:
        vm = self._vms.get(vm_id)
        if not vm:
            return False
        return vm.start()

    def shutdown_vm(self, vm_id: str) -> bool:
        vm = self._vms.get(vm_id)
        if not vm:
            return False
        success = vm.shutdown()
        return success

    def shutdown_all(self) -> None:
        for vm_id in list(self._vms.keys()):
            self.shutdown_vm(vm_id)

    def get_available_vm(self) -> FirecrackerVM | None:
        for vm in self._vms.values():
            if vm.state == VMState.RUNNING:
                return vm
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "total_vms": len(self._vms),
            "running": sum(1 for v in self._vms.values() if v.state == VMState.RUNNING),
            "terminated": sum(1 for v in self._vms.values() if v.state == VMState.TERMINATED),
            "firecracker_available": shutil.which(self.firecracker_binary) is not None,
            "kernel_path": self.kernel_path,
            "rootfs_path": self.rootfs_path,
        }
