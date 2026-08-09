"""Tests for Firecracker VM manager."""

import pytest

from daemon_engine.runtime.firecracker.vm_manager import (
    FirecrackerManager,
    FirecrackerVM,
    VMSpec,
    MachineConfig,
    BootSource,
    DriveConfig,
    NetworkInterface,
    VMState,
    ActionType,
)


class TestMachineConfig:
    def test_defaults(self):
        config = MachineConfig()
        assert config.vcpu_count == 2
        assert config.mem_size_mib == 512

    def test_to_dict(self):
        config = MachineConfig(vcpu_count=4, mem_size_mib=1024)
        d = config.to_dict()
        assert d["vcpu_count"] == 4
        assert d["mem_size_mib"] == 1024


class TestBootSource:
    def test_to_dict(self):
        boot = BootSource(kernel_image_path="/vmlinux", boot_args="console=ttyS0")
        d = boot.to_dict()
        assert d["kernel_image_path"] == "/vmlinux"
        assert d["boot_args"] == "console=ttyS0"

    def test_with_initrd(self):
        boot = BootSource(kernel_image_path="/vmlinux", initrd_path="/initrd")
        d = boot.to_dict()
        assert d["initrd_path"] == "/initrd"


class TestDriveConfig:
    def test_to_dict(self):
        drive = DriveConfig(drive_id="rootfs", path_on_host="/rootfs.ext4", is_root_device=True)
        d = drive.to_dict()
        assert d["drive_id"] == "rootfs"
        assert d["is_root_device"] is True


class TestVMSpec:
    def test_to_config_dict(self):
        spec = VMSpec(
            vm_id="test-vm",
            boot_source=BootSource(kernel_image_path="/vmlinux"),
            machine_config=MachineConfig(vcpu_count=4),
            drives=[DriveConfig(drive_id="rootfs", path_on_host="/rootfs")],
        )
        config = spec.to_config_dict()
        assert config["vm_id"] == "test-vm"
        assert config["machine_config"]["vcpu_count"] == 4
        assert len(config["drives"]) == 1


class TestFirecrackerVM:
    def test_initial_state(self):
        spec = VMSpec(vm_id="test-vm")
        vm = FirecrackerVM(spec)
        assert vm.state == VMState.NOT_STARTED
        assert vm.vm_id == "test-vm"

    def test_start_simulated(self):
        spec = VMSpec(vm_id="test-vm")
        vm = FirecrackerVM(spec, firecracker_binary="nonexistent-binary")
        assert vm.is_available is False
        assert vm.start() is True
        assert vm.state == VMState.RUNNING
        info = vm.info()
        assert info["state"] == "Running"

    def test_shutdown(self):
        spec = VMSpec(vm_id="test-vm")
        vm = FirecrackerVM(spec, firecracker_binary="nonexistent-binary")
        vm.start()
        assert vm.shutdown() is True
        assert vm.state == VMState.TERMINATED

    def test_execute_when_not_running(self):
        spec = VMSpec(vm_id="test-vm")
        vm = FirecrackerVM(spec, firecracker_binary="nonexistent-binary")
        result = vm.execute("echo hello")
        assert result.success is False

    def test_status(self):
        spec = VMSpec(vm_id="test-vm")
        vm = FirecrackerVM(spec, firecracker_binary="nonexistent-binary")
        vm.start()
        status = vm.status()
        assert status["vm_id"] == "test-vm"
        assert status["state"] == "Running"


class TestFirecrackerManager:
    def test_create_vm(self):
        manager = FirecrackerManager(firecracker_binary="nonexistent-binary")
        vm = manager.create_vm(vm_id="test-1")
        assert vm.vm_id == "test-1"
        assert vm in manager._vms.values() or manager.get_vm("test-1") is not None

    def test_list_vms(self):
        manager = FirecrackerManager(firecracker_binary="nonexistent-binary")
        manager.create_vm(vm_id="test-1")
        manager.create_vm(vm_id="test-2")
        vms = manager.list_vms()
        assert len(vms) == 2

    def test_start_and_shutdown(self):
        manager = FirecrackerManager(firecracker_binary="nonexistent-binary")
        vm = manager.create_vm(vm_id="test-1")
        assert manager.start_vm("test-1") is True
        assert manager.shutdown_vm("test-1") is True

    def test_stats(self):
        manager = FirecrackerManager(firecracker_binary="nonexistent-binary")
        manager.create_vm(vm_id="test-1")
        stats = manager.stats()
        assert stats["total_vms"] == 1
        assert "firecracker_available" in stats

    def test_shutdown_all(self):
        manager = FirecrackerManager(firecracker_binary="nonexistent-binary")
        manager.create_vm(vm_id="test-1")
        manager.create_vm(vm_id="test-2")
        manager.start_vm("test-1")
        manager.start_vm("test-2")
        manager.shutdown_all()
        assert all(vm.state == VMState.TERMINATED for vm in manager._vms.values())
