"""Firecracker microVM runtime integration.

Integrates Firecracker's microVM isolation with the daemon engine runtime.
Provides VM-backed sandboxed execution for maximum isolation.
"""

from daemon_engine.runtime.firecracker.vm_manager import (
    FirecrackerVM,
    FirecrackerManager,
    VMSpec,
    MachineConfig,
    BootSource,
    DriveConfig,
    NetworkInterface,
    VMState,
    ActionType,
)

__all__ = [
    "FirecrackerVM",
    "FirecrackerManager",
    "VMSpec",
    "MachineConfig",
    "BootSource",
    "DriveConfig",
    "NetworkInterface",
    "VMState",
    "ActionType",
]
