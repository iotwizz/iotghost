"""QEMU process manager -- launches, monitors, and controls firmware emulation.

Handles architecture-specific QEMU configurations, kernel selection,
drive mounting, serial console capture, and process lifecycle management.
The AI agent interacts with QEMU through this module's structured interface
while also being able to issue raw commands via the shell tools.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from iotghost.prompts import ARCH_QEMU_MAP, KERNEL_APPEND_TEMPLATES

logger = logging.getLogger(__name__)


class EmulationStatus(str, Enum):
    """Current state of the QEMU emulation."""
    NOT_STARTED = "not_started"
    STARTING = "starting"
    BOOTING = "booting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"
    TIMEOUT = "timeout"


@dataclass
class QemuConfig:
    """QEMU launch configuration assembled by the pipeline."""
    architecture: str = "mipsel"
    kernel_path: str = ""
    rootfs_path: str = ""
    rootfs_format: str = "raw"          # raw, qcow2
    memory_mb: int = 256
    cpu: str = ""                        # auto-detected from arch
    machine: str = ""                    # auto-detected from arch
    nic_model: str = ""                  # auto-detected from arch
    console: str = ""                    # auto-detected from arch
    kernel_append: str = ""              # auto-generated from arch
    extra_args: list[str] = field(default_factory=list)
    network_tap: str = "tap0"
    network_mode: str = "tap"            # tap, user, none
    serial_log: str = ""                 # path to serial output log
    enable_gdb: bool = False
    gdb_port: int = 1234
    qemu_binary: str = ""               # auto-detected from arch

    def apply_arch_defaults(self) -> None:
        """Fill in architecture-specific defaults from ARCH_QEMU_MAP."""
        arch_cfg = ARCH_QEMU_MAP.get(self.architecture, {})
        if not self.qemu_binary:
            self.qemu_binary = arch_cfg.get("qemu_binary", f"qemu-system-{self.architecture}")
        if not self.machine:
            self.machine = arch_cfg.get("machine", "virt")
        if not self.cpu:
            self.cpu = arch_cfg.get("cpu", "")
        if not self.nic_model:
            self.nic_model = arch_cfg.get("nic_model", "e1000")
        if not self.console:
            self.console = arch_cfg.get("console", "ttyS0")
        if not self.kernel_append:
            self.kernel_append = KERNEL_APPEND_TEMPLATES.get(
                self.architecture,
                f"root=/dev/sda1 console={self.console} rw rootwait",
            )


@dataclass
class BootEvent:
    """Event detected during QEMU boot monitoring."""
    timestamp: float
    event_type: str          # kernel_start, init_start, panic, service_up, login_prompt, etc.
    message: str
    raw_line: str


@dataclass
class EmulationState:
    """Mutable state of a running QEMU instance."""
    status: EmulationStatus = EmulationStatus.NOT_STARTED
    process: subprocess.Popen | None = None
    pid: int | None = None
    boot_events: list[BootEvent] = field(default_factory=list)
    serial_output: list[str] = field(default_factory=list)
    start_time: float = 0.0
    detected_services: list[tuple[str, int]] = field(default_factory=list)  # (name, port)
    detected_ip: str | None = None
    last_error: str | None = None
    boot_complete: bool = False

    @property
    def uptime(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time


# ---------------------------------------------------------------------------
# Boot output pattern matching
# ---------------------------------------------------------------------------

# Patterns to detect in serial console output during boot
_BOOT_PATTERNS: list[tuple[str, str]] = [
    (r"Linux version \d+\.\d+", "kernel_start"),
    (r"Kernel command line:", "kernel_cmdline"),
    (r"VFS: Mounted root", "rootfs_mounted"),
    (r"init started|/etc/init|rcS|sysinit", "init_start"),
    (r"Kernel panic", "kernel_panic"),
    (r"not syncing", "kernel_panic"),
    (r"Unable to mount root", "rootfs_error"),
    (r"Segmentation fault|segfault", "segfault"),
    (r"httpd|lighttpd|nginx|uhttpd|mini_httpd", "web_server"),
    (r"listening on .* port \d+|bound to port", "service_listening"),
    (r"login:|Login:", "login_prompt"),
    (r"Please press Enter to activate", "console_ready"),
    (r"BusyBox", "busybox_init"),
    (r"Starting .*\.\.\.", "service_starting"),
    (r"ifconfig|ip addr|eth0|br0", "network_config"),
    (r"NVRAM|nvram", "nvram_access"),
]

import re

def classify_boot_line(line: str) -> str | None:
    """Check a serial output line against known boot patterns.

    Returns the event type string if a pattern matches, None otherwise.
    """
    for pattern, event_type in _BOOT_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return event_type
    return None


# ---------------------------------------------------------------------------
# QEMU command builder
# ---------------------------------------------------------------------------

def build_qemu_command(config: QemuConfig) -> list[str]:
    """Build the QEMU command line from configuration.

    Returns a list of command-line arguments suitable for subprocess.
    """
    config.apply_arch_defaults()

    cmd = [config.qemu_binary]

    # Machine and CPU
    cmd.extend(["-M", config.machine])
    if config.cpu:
        cmd.extend(["-cpu", config.cpu])

    # Memory
    cmd.extend(["-m", str(config.memory_mb)])

    # Kernel
    if config.kernel_path:
        cmd.extend(["-kernel", config.kernel_path])
        if config.kernel_append:
            cmd.extend(["-append", config.kernel_append])

    # Root filesystem drive
    if config.rootfs_path:
        if config.rootfs_format == "qcow2":
            cmd.extend([
                "-drive", f"file={config.rootfs_path},format=qcow2,if=ide",
            ])
        else:
            # Raw image -- try as IDE drive
            cmd.extend([
                "-drive", f"file={config.rootfs_path},format=raw,if=ide",
            ])

    # Network
    if config.network_mode == "tap":
        cmd.extend([
            "-netdev", f"tap,id=net0,ifname={config.network_tap},script=no,downscript=no",
            "-device", f"{config.nic_model},netdev=net0",
        ])
    elif config.network_mode == "user":
        cmd.extend([
            "-netdev", "user,id=net0,hostfwd=tcp::8080-:80,hostfwd=tcp::2222-:22",
            "-device", f"{config.nic_model},netdev=net0",
        ])
    # else: no network

    # Serial console to stdio
    cmd.extend(["-nographic", "-serial", "stdio"])

    # GDB server
    if config.enable_gdb:
        cmd.extend(["-gdb", f"tcp::{config.gdb_port}", "-S"])

    # Extra arguments
    cmd.extend(config.extra_args)

    return cmd


# ---------------------------------------------------------------------------
# QEMU process manager
# ---------------------------------------------------------------------------

class QemuManager:
    """Manages a QEMU process for firmware emulation.

    Handles launching, monitoring serial output, detecting boot events,
    and graceful/forceful shutdown. Provides callbacks for the TUI to
    display real-time boot progress.
    """

    def __init__(
        self,
        config: QemuConfig,
        on_boot_event: Callable[[BootEvent], None] | None = None,
        on_serial_line: Callable[[str], None] | None = None,
        on_status_change: Callable[[EmulationStatus], None] | None = None,
    ) -> None:
        self.config = config
        self.state = EmulationState()

        # Callbacks
        self._on_boot_event = on_boot_event
        self._on_serial_line = on_serial_line
        self._on_status_change = on_status_change

        # Serial monitor thread
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _set_status(self, status: EmulationStatus) -> None:
        """Update status and fire callback."""
        old = self.state.status
        self.state.status = status
        if self._on_status_change and old != status:
            self._on_status_change(status)

    def start(self) -> bool:
        """Launch QEMU and begin monitoring.

        Returns True if the process started successfully.
        """
        cmd = build_qemu_command(self.config)
        cmd_str = " ".join(cmd)
        logger.info("Starting QEMU: %s", cmd_str)

        self._set_status(EmulationStatus.STARTING)

        try:
            self.state.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env={**os.environ, "TERM": "dumb"},
            )
            self.state.pid = self.state.process.pid
            self.state.start_time = time.time()

            logger.info("QEMU started with PID %d", self.state.pid)
            self._set_status(EmulationStatus.BOOTING)

            # Start serial output monitor thread
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_serial,
                daemon=True,
                name="qemu-serial-monitor",
            )
            self._monitor_thread.start()

            return True

        except FileNotFoundError:
            err = f"QEMU binary not found: {self.config.qemu_binary}"
            logger.error(err)
            self.state.last_error = err
            self._set_status(EmulationStatus.FAILED)
            return False
        except Exception as exc:
            err = f"Failed to start QEMU: {type(exc).__name__}: {exc}"
            logger.error(err)
            self.state.last_error = err
            self._set_status(EmulationStatus.FAILED)
            return False

    def _monitor_serial(self) -> None:
        """Background thread that reads QEMU serial output line by line.

        Classifies each line against boot patterns and fires events.
        Also writes to serial log file if configured.
        """
        proc = self.state.process
        if not proc or not proc.stdout:
            return

        log_file = None
        if self.config.serial_log:
            try:
                log_file = open(self.config.serial_log, "w")
            except OSError as exc:
                logger.warning("Cannot open serial log %s: %s", self.config.serial_log, exc)

        try:
            for line in proc.stdout:
                if self._stop_event.is_set():
                    break

                line = line.rstrip("\n")
                self.state.serial_output.append(line)

                # Write to log file
                if log_file:
                    log_file.write(line + "\n")
                    log_file.flush()

                # Callback for TUI
                if self._on_serial_line:
                    self._on_serial_line(line)

                # Classify boot event
                event_type = classify_boot_line(line)
                if event_type:
                    event = BootEvent(
                        timestamp=time.time(),
                        event_type=event_type,
                        message=line[:200],
                        raw_line=line,
                    )
                    self.state.boot_events.append(event)
                    logger.info("Boot event: %s -- %s", event_type, line[:100])

                    if self._on_boot_event:
                        self._on_boot_event(event)

                    # State transitions based on events
                    if event_type == "kernel_panic":
                        self.state.last_error = line
                        self._set_status(EmulationStatus.FAILED)
                    elif event_type == "login_prompt" or event_type == "console_ready":
                        self.state.boot_complete = True
                        self._set_status(EmulationStatus.RUNNING)

        except Exception as exc:
            logger.error("Serial monitor error: %s", exc)
        finally:
            if log_file:
                log_file.close()

        # Check if process exited
        if proc.poll() is not None and self.state.status == EmulationStatus.BOOTING:
            self.state.last_error = f"QEMU exited with code {proc.returncode}"
            self._set_status(EmulationStatus.FAILED)

    def send_command(self, command: str) -> None:
        """Send a command to QEMU's serial console (stdin)."""
        proc = self.state.process
        if proc and proc.stdin and proc.poll() is None:
            try:
                proc.stdin.write(command + "\n")
                proc.stdin.flush()
                logger.debug("Sent to QEMU: %s", command[:100])
            except (BrokenPipeError, OSError) as exc:
                logger.error("Failed to send command to QEMU: %s", exc)

    def wait_for_boot(self, timeout: int = 120) -> bool:
        """Wait until boot completes or timeout.

        Returns True if the firmware booted successfully.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.state.boot_complete:
                return True
            if self.state.status == EmulationStatus.FAILED:
                return False
            if self.state.process and self.state.process.poll() is not None:
                return False
            time.sleep(1)

        self._set_status(EmulationStatus.TIMEOUT)
        self.state.last_error = f"Boot timeout after {timeout}s"
        return False

    def get_recent_output(self, lines: int = 50) -> str:
        """Get the most recent serial console output."""
        recent = self.state.serial_output[-lines:]
        return "\n".join(recent)

    def get_boot_summary(self) -> dict:
        """Summarize boot events for the AI agent."""
        return {
            "status": self.state.status.value,
            "uptime": round(self.state.uptime, 1),
            "events": [
                {"type": e.event_type, "message": e.message[:100]}
                for e in self.state.boot_events
            ],
            "services_detected": self.state.detected_services,
            "last_error": self.state.last_error,
            "boot_complete": self.state.boot_complete,
            "total_output_lines": len(self.state.serial_output),
        }

    def stop(self, force: bool = False) -> None:
        """Stop the QEMU process.

        Tries graceful shutdown first (SIGTERM), then SIGKILL if needed.
        """
        self._stop_event.set()
        proc = self.state.process

        if not proc or proc.poll() is not None:
            self._set_status(EmulationStatus.STOPPED)
            return

        if force:
            logger.info("Force killing QEMU (PID %d)", proc.pid)
            proc.kill()
        else:
            logger.info("Gracefully stopping QEMU (PID %d)", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("QEMU did not exit gracefully, killing")
                proc.kill()

        self._set_status(EmulationStatus.STOPPED)
        logger.info("QEMU stopped")

    def is_running(self) -> bool:
        """Check if QEMU process is still running."""
        return (
            self.state.process is not None
            and self.state.process.poll() is None
        )


# ---------------------------------------------------------------------------
# Rootfs image creation utilities
# ---------------------------------------------------------------------------

def _has_tool(name: str) -> bool:
    """Check if a command-line tool is available on PATH."""
    return shutil.which(name) is not None


def _create_image_genext2fs(
    rootfs_dir: str, output_path: str, size_mb: int,
) -> None:
    """Create ext2 image with genext2fs (no root required)."""
    block_size = 1024
    num_blocks = size_mb * 1024  # 1KB blocks
    subprocess.run(
        [
            "genext2fs", "-b", str(num_blocks),
            "-d", rootfs_dir, output_path,
        ],
        check=True, capture_output=True,
    )
    # Optionally upgrade to ext4 if tune2fs is available
    if _has_tool("tune2fs"):
        subprocess.run(
            ["tune2fs", "-O", "extents,uninit_bg,dir_index,has_journal", output_path],
            capture_output=True,  # best-effort, ext2 still works fine
        )
    logger.info("Created rootfs image via genext2fs (no root needed)")


def _create_image_debugfs(
    rootfs_dir: str, output_path: str, size_mb: int, fs_type: str,
) -> None:
    """Create image with dd+mkfs then populate via debugfs (no root required)."""
    subprocess.run(
        ["dd", "if=/dev/zero", f"of={output_path}", "bs=1M", f"count={size_mb}"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [f"mkfs.{fs_type}", "-F", output_path],
        check=True, capture_output=True,
    )
    # Use debugfs to copy directory tree into image without mounting
    # Build a command script for debugfs
    rootfs = Path(rootfs_dir)
    cmds: list[str] = []
    for dirpath, dirnames, filenames in os.walk(rootfs):
        rel = os.path.relpath(dirpath, rootfs)
        if rel != ".":
            cmds.append(f"mkdir {rel}")
        for fname in filenames:
            src = os.path.join(dirpath, fname)
            dest = os.path.join(rel, fname) if rel != "." else fname
            # debugfs 'write' command: write <local_file> <ext_path>
            cmds.append(f"write {src} {dest}")
    cmd_script = "\n".join(cmds)
    subprocess.run(
        ["debugfs", "-w", "-f", "/dev/stdin", output_path],
        input=cmd_script.encode(),
        check=True, capture_output=True,
    )
    logger.info("Created rootfs image via debugfs (no root needed)")


def _create_image_mount(
    rootfs_dir: str, output_path: str, size_mb: int, fs_type: str,
) -> None:
    """Create image with mount -o loop (requires root/sudo)."""
    subprocess.run(
        ["dd", "if=/dev/zero", f"of={output_path}", "bs=1M", f"count={size_mb}"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [f"mkfs.{fs_type}", "-F", output_path],
        check=True, capture_output=True,
    )
    mount_point = f"{output_path}.mount"
    os.makedirs(mount_point, exist_ok=True)
    try:
        subprocess.run(
            ["mount", "-o", "loop", output_path, mount_point],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["cp", "-af", f"{rootfs_dir}/.", mount_point],
            check=True, capture_output=True,
        )
    finally:
        subprocess.run(["umount", mount_point], capture_output=True)
        if os.path.isdir(mount_point):
            shutil.rmtree(mount_point, ignore_errors=True)
    logger.info("Created rootfs image via mount (required root)")


def create_rootfs_image(
    rootfs_dir: str,
    output_path: str,
    size_mb: int = 256,
    fs_type: str = "ext4",
) -> str:
    """Create a mountable disk image from an extracted rootfs directory.

    QEMU needs a disk image, not a directory. Uses a fallback chain:
    1. genext2fs -- no root needed, widely available
    2. debugfs   -- no root needed, part of e2fsprogs
    3. mount -o loop -- requires root/sudo (legacy fallback)

    Returns the path to the created image.
    """
    logger.info("Creating %dMB %s image from %s", size_mb, fs_type, rootfs_dir)

    errors: list[str] = []

    # --- Strategy 1: genext2fs (preferred, no root) ---
    if _has_tool("genext2fs"):
        try:
            _create_image_genext2fs(rootfs_dir, output_path, size_mb)
            return _finalise_image(output_path)
        except subprocess.CalledProcessError as exc:
            errors.append(f"genext2fs failed: {exc}")
            logger.warning("genext2fs failed, trying next method: %s", exc)
    else:
        errors.append("genext2fs not installed")

    # --- Strategy 2: debugfs (no root, part of e2fsprogs) ---
    if _has_tool("debugfs") and _has_tool(f"mkfs.{fs_type}"):
        try:
            _create_image_debugfs(rootfs_dir, output_path, size_mb, fs_type)
            return _finalise_image(output_path)
        except subprocess.CalledProcessError as exc:
            errors.append(f"debugfs failed: {exc}")
            logger.warning("debugfs failed, trying next method: %s", exc)
    else:
        errors.append("debugfs or mkfs not installed")

    # --- Strategy 3: mount -o loop (requires root) ---
    if os.geteuid() == 0 or _has_tool("sudo"):
        try:
            _create_image_mount(rootfs_dir, output_path, size_mb, fs_type)
            return _finalise_image(output_path)
        except subprocess.CalledProcessError as exc:
            errors.append(f"mount failed (exit {exc.returncode}): {exc.stderr[:200] if exc.stderr else ''}")
            logger.warning("mount -o loop failed: %s", exc)
        except PermissionError as exc:
            errors.append(f"mount permission denied: {exc}")
    else:
        errors.append("mount -o loop requires root and sudo not available")

    # --- All strategies failed ---
    raise RuntimeError(
        f"Could not create rootfs image. Tried {len(errors)} methods:\n"
        + "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
        + "\n\nFix: install genext2fs (apt install genext2fs) or run with sudo."
    )


def _finalise_image(output_path: str) -> str:
    """Log final image size and return path."""
    size_actual = os.path.getsize(output_path)
    logger.info("Created rootfs image: %s (%d bytes)", output_path, size_actual)
    return output_path


def find_kernel(architecture: str, kernels_dir: str | None = None) -> str | None:
    """Find a pre-built kernel matching the given architecture.

    Searches the kernels directory for files matching the arch prefix.
    Returns the path to the best matching kernel, or None.
    """
    if not kernels_dir:
        # Default: package's kernels/ directory
        kernels_dir = str(Path(__file__).parent / "kernels")

    arch_cfg = ARCH_QEMU_MAP.get(architecture, {})
    prefix = arch_cfg.get("kernel_prefix", "")
    if not prefix:
        return None

    kernels_path = Path(kernels_dir)
    if not kernels_path.exists():
        logger.warning("Kernels directory not found: %s", kernels_dir)
        return None

    # Find all matching kernels, prefer higher versions
    matches = sorted(
        kernels_path.glob(f"{prefix}*"),
        key=lambda p: p.name,
        reverse=True,
    )

    if matches:
        logger.info("Found kernel: %s", matches[0])
        return str(matches[0])

    logger.warning("No kernel found for architecture '%s' (prefix: %s)", architecture, prefix)
    return None
