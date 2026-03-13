"""Main orchestration pipeline -- drives the full firmware emulation workflow.

Chains the phases: Extract -> Analyze -> Prepare -> Emulate -> Fix -> Verify.
The AI agent (ShellAgent) drives each phase autonomously, receiving phase-specific
prompts and using shell tools to accomplish each step. The pipeline manages
state transitions, retry logic, and coordination between modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from iotghost.agent import AgentConfig, ShellAgent
from iotghost.emulator import (
    EmulationStatus,
    QemuConfig,
    QemuManager,
    create_rootfs_image,
    find_kernel,
)
from iotghost.extractors import (
    Architecture,
    ExtractionResult,
    detect_architecture,
    detect_kernel_version,
    find_rootfs,
    parse_binwalk_output,
)
from iotghost.network import NetworkConfig, NetworkManager, detect_host_interface
from iotghost.nvram import NvramConfig, deploy_nvram, scan_rootfs_for_nvram
from iotghost.prompts import (
    ANALYZE_PROMPT,
    EMULATE_PROMPT,
    EXTRACT_PROMPT,
    FIX_PROMPT,
    PREPARE_PROMPT,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
)
from iotghost.tools import register_all_tools

logger = logging.getLogger(__name__)


class PipelinePhase(str, Enum):
    """Phases of the emulation pipeline."""
    INIT = "init"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    PREPARE = "prepare"
    EMULATE = "emulate"
    FIX = "fix"
    VERIFY = "verify"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the emulation pipeline."""
    firmware_path: str = ""
    workdir: str = ""
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    network_mode: str = "tap"       # tap, user, none
    device_ip: str = "192.168.1.1"
    max_fix_attempts: int = 5
    boot_timeout: int = 120
    auto_cleanup: bool = True
    kernels_dir: str | None = None
    force_arch: str | None = None    # override auto-detection


@dataclass
class PipelineState:
    """Mutable state tracked across the pipeline."""
    phase: PipelinePhase = PipelinePhase.INIT
    firmware_path: str = ""
    workdir: str = ""
    rootfs_path: str | None = None
    rootfs_image: str | None = None
    architecture: str = "unknown"
    endianness: str = "unknown"
    kernel_version: str | None = None
    kernel_path: str | None = None
    needs_nvram: bool = False
    nvram_config: NvramConfig | None = None
    qemu_config: QemuConfig | None = None
    fix_attempts: int = 0
    last_error: str | None = None
    start_time: float = field(default_factory=time.time)
    phase_times: dict[str, float] = field(default_factory=dict)
    accessible_services: list[tuple[str, int]] = field(default_factory=list)
    final_status: str = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class EmulationPipeline:
    """Orchestrates the full firmware emulation workflow.

    The pipeline creates and manages:
    - ShellAgent: the AI that drives each phase
    - QemuManager: QEMU process lifecycle
    - NetworkManager: host network setup
    - NVRAM configuration and deployment

    Each phase injects a prompt into the agent, then lets the agent
    run autonomously until it completes or fails. On failure, the pipeline
    transitions to the FIX phase before retrying.
    """

    def __init__(
        self,
        config: PipelineConfig,
        on_phase_change: Callable[[PipelinePhase, PipelinePhase], None] | None = None,
        on_agent_message: Callable[[str], None] | None = None,
        on_status_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.state = PipelineState()

        # Callbacks for TUI integration
        self._on_phase_change = on_phase_change
        self._on_agent_message = on_agent_message
        self._on_status_update = on_status_update

        # Components (initialized during setup)
        self.agent: ShellAgent | None = None
        self.qemu: QemuManager | None = None
        self.network: NetworkManager | None = None

    def _set_phase(self, phase: PipelinePhase) -> None:
        """Transition to a new pipeline phase."""
        old = self.state.phase
        self.state.phase = phase

        # Record phase timing
        self.state.phase_times[old.value] = time.time()

        if self._on_phase_change:
            self._on_phase_change(old, phase)

        logger.info("Pipeline phase: %s -> %s", old.value, phase.value)

    def _notify_status(self) -> None:
        """Send current status to TUI callback."""
        if self._on_status_update:
            self._on_status_update(self.get_status())

    def setup(self) -> None:
        """Initialize pipeline components."""
        logger.info("Pipeline setup: firmware=%s", self.config.firmware_path)

        # Validate firmware file
        fw_path = Path(self.config.firmware_path)
        if not fw_path.exists():
            raise FileNotFoundError(f"Firmware file not found: {self.config.firmware_path}")

        self.state.firmware_path = str(fw_path.resolve())

        # Create working directory
        if self.config.workdir:
            workdir = Path(self.config.workdir)
        else:
            workdir = fw_path.parent / f"iotghost_work_{fw_path.stem}"
        workdir.mkdir(parents=True, exist_ok=True)
        self.state.workdir = str(workdir)

        # Register shell tools
        register_all_tools()

        # Initialize AI agent
        self.agent = ShellAgent(
            config=self.config.agent_config,
            on_message=lambda msg: (
                self._on_agent_message(msg.content)
                if self._on_agent_message and msg.content
                else None
            ),
        )
        self.agent.initialize(SYSTEM_PROMPT)

        # Set agent context variables
        self.agent.set_context_var("firmware_path", self.state.firmware_path)
        self.agent.set_context_var("workdir", self.state.workdir)

        logger.info("Pipeline setup complete: workdir=%s", self.state.workdir)

    def run(self) -> PipelineState:
        """Execute the full emulation pipeline.

        Returns the final pipeline state with results.
        """
        try:
            self.setup()
            self._run_extract()
            self._run_analyze()
            self._run_prepare()
            self._run_emulate_loop()
            self._run_verify()
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self.state.final_status = "interrupted"
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            self.state.final_status = f"error: {type(exc).__name__}: {exc}"
            self._set_phase(PipelinePhase.FAILED)
        finally:
            self._cleanup()

        return self.state

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _run_extract(self) -> None:
        """Phase 1: Extract firmware rootfs."""
        self._set_phase(PipelinePhase.EXTRACT)

        prompt = EXTRACT_PROMPT.format(
            firmware_path=self.state.firmware_path,
        )
        self.agent.inject_context(prompt)
        result = self.agent.run_until_done(max_iterations=15)

        # Check if agent found a rootfs
        rootfs = self.agent.get_context_var("rootfs_path")
        if not rootfs:
            # Try to find it ourselves from the workdir
            rootfs = find_rootfs(self.state.workdir)

        if not rootfs:
            raise RuntimeError(
                "Extraction failed: could not locate root filesystem. "
                f"Agent output: {result[:500]}"
            )

        self.state.rootfs_path = rootfs
        self.agent.set_context_var("rootfs_path", rootfs)
        logger.info("Extraction complete: rootfs at %s", rootfs)

    def _run_analyze(self) -> None:
        """Phase 2: Analyze extracted rootfs."""
        self._set_phase(PipelinePhase.ANALYZE)

        prompt = ANALYZE_PROMPT.format(
            rootfs_path=self.state.rootfs_path,
        )
        self.agent.inject_context(prompt)
        result = self.agent.run_until_done(max_iterations=10)

        # Override with forced arch if specified
        if self.config.force_arch:
            self.state.architecture = self.config.force_arch
        else:
            arch = self.agent.get_context_var("architecture")
            if arch:
                self.state.architecture = arch

        # Get NVRAM scan results
        nvram_scan = scan_rootfs_for_nvram(self.state.rootfs_path)
        self.state.needs_nvram = nvram_scan.needs_nvram
        self.state.nvram_config = NvramConfig(
            needed=nvram_scan.needs_nvram,
            vendor_family=nvram_scan.recommended_vendor,
            extracted_keys=nvram_scan.extracted_defaults,
            binary_refs=nvram_scan.nvram_binaries,
        )

        # Find a suitable kernel
        self.state.kernel_path = find_kernel(
            self.state.architecture,
            self.config.kernels_dir,
        )

        self.agent.set_context_var("architecture", self.state.architecture)
        self.agent.set_context_var("needs_nvram", str(self.state.needs_nvram))
        logger.info(
            "Analysis complete: arch=%s, nvram=%s, kernel=%s",
            self.state.architecture, self.state.needs_nvram, self.state.kernel_path,
        )

    def _run_prepare(self) -> None:
        """Phase 3: Prepare rootfs for emulation."""
        self._set_phase(PipelinePhase.PREPARE)

        # Deploy NVRAM if needed
        if self.state.needs_nvram and self.state.nvram_config:
            actions = deploy_nvram(
                self.state.rootfs_path,
                self.state.nvram_config,
                device_ip=self.config.device_ip,
            )
            for action in actions:
                logger.info("NVRAM: %s", action)

        # Let agent do additional preparation
        prompt = PREPARE_PROMPT.format(
            rootfs_path=self.state.rootfs_path,
            architecture=self.state.architecture,
            needs_nvram=self.state.needs_nvram,
        )
        self.agent.inject_context(prompt)
        self.agent.run_until_done(max_iterations=15)

        # Create rootfs disk image for QEMU
        image_path = str(Path(self.state.workdir) / "rootfs.img")
        self.state.rootfs_image = create_rootfs_image(
            self.state.rootfs_path,
            image_path,
            size_mb=256,
        )

        logger.info("Preparation complete: image=%s", self.state.rootfs_image)

    def _run_emulate_loop(self) -> None:
        """Phase 4+5: Emulate with fix loop.

        Tries to boot in QEMU. If boot fails, enters FIX phase
        to diagnose and repair, then retries. Loops up to max_fix_attempts.
        """
        self._set_phase(PipelinePhase.EMULATE)

        # Build QEMU configuration
        qemu_config = QemuConfig(
            architecture=self.state.architecture,
            kernel_path=self.state.kernel_path or "",
            rootfs_path=self.state.rootfs_image or self.state.rootfs_path,
            memory_mb=256,
            network_mode=self.config.network_mode,
            serial_log=str(Path(self.state.workdir) / "serial.log"),
        )
        qemu_config.apply_arch_defaults()
        self.state.qemu_config = qemu_config

        # Setup network
        if self.config.network_mode == "tap":
            host_iface = detect_host_interface() or "eth0"
            net_config = NetworkConfig(
                device_ip=self.config.device_ip,
                nat_interface=host_iface,
                qemu_mode="tap",
            )
            self.network = NetworkManager(net_config)
            self.network.setup()

        while self.state.fix_attempts <= self.config.max_fix_attempts:
            # Launch QEMU
            self.qemu = QemuManager(
                config=qemu_config,
                on_serial_line=lambda line: logger.debug("SERIAL: %s", line),
            )

            if not self.qemu.start():
                self.state.last_error = self.qemu.state.last_error
                logger.error("QEMU failed to start: %s", self.state.last_error)
                break

            # Let agent monitor boot via its own commands
            prompt = EMULATE_PROMPT.format(
                qemu_binary=qemu_config.qemu_binary,
                kernel_path=qemu_config.kernel_path,
                rootfs_path=qemu_config.rootfs_path,
                architecture=self.state.architecture,
            )
            self.agent.inject_context(prompt)
            agent_result = self.agent.run_until_done(max_iterations=10)

            # Wait for boot
            boot_success = self.qemu.wait_for_boot(timeout=self.config.boot_timeout)

            if boot_success:
                logger.info("Boot successful!")
                return

            # Boot failed -- enter fix phase
            self.state.fix_attempts += 1
            self.state.last_error = self.qemu.state.last_error or "Unknown boot failure"

            if self.state.fix_attempts > self.config.max_fix_attempts:
                logger.error(
                    "Max fix attempts (%d) reached. Giving up.",
                    self.config.max_fix_attempts,
                )
                self._set_phase(PipelinePhase.FAILED)
                self.state.final_status = "failed_max_retries"
                return

            # Stop current QEMU instance
            self.qemu.stop()

            # Run fix phase
            self._set_phase(PipelinePhase.FIX)
            error_output = self.qemu.get_recent_output(lines=30)
            previous_fixes = "\n".join(
                f"- {fix}" for fix in self.agent.state.attempted_fixes
            ) or "None yet"

            prompt = FIX_PROMPT.format(
                error_output=error_output,
                previous_fixes=previous_fixes,
            )
            self.agent.inject_context(prompt)
            fix_result = self.agent.run_until_done(max_iterations=10)
            self.agent.state.attempted_fixes.append(fix_result[:200])

            # Back to emulate phase for retry
            self._set_phase(PipelinePhase.EMULATE)
            logger.info("Fix attempt %d complete, retrying emulation", self.state.fix_attempts)

    def _run_verify(self) -> None:
        """Phase 6: Verify the firmware is running and services are accessible."""
        self._set_phase(PipelinePhase.VERIFY)

        prompt = VERIFY_PROMPT.format(
            device_ip=self.config.device_ip,
        )
        self.agent.inject_context(prompt)
        result = self.agent.run_until_done(max_iterations=10)

        # Parse verification results
        if "RUNNING" in result.upper():
            self.state.final_status = "running"
        elif "PARTIAL" in result.upper():
            self.state.final_status = "partial"
        else:
            self.state.final_status = "verify_uncertain"

        self._set_phase(PipelinePhase.DONE)
        logger.info("Pipeline complete: status=%s", self.state.final_status)

    # ------------------------------------------------------------------
    # Cleanup and status
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.qemu and self.qemu.is_running():
            if self.state.final_status not in ("running", "partial"):
                self.qemu.stop()

        if self.network and self.config.auto_cleanup:
            if self.state.final_status not in ("running", "partial"):
                self.network.teardown()

        if self.agent:
            self.agent.shutdown()

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive pipeline status."""
        status = {
            "phase": self.state.phase.value,
            "firmware": self.state.firmware_path,
            "architecture": self.state.architecture,
            "rootfs": self.state.rootfs_path,
            "needs_nvram": self.state.needs_nvram,
            "kernel": self.state.kernel_path,
            "fix_attempts": self.state.fix_attempts,
            "last_error": self.state.last_error,
            "elapsed": round(self.state.elapsed, 1),
            "final_status": self.state.final_status,
        }

        if self.agent:
            status["agent"] = self.agent.get_stats()
        if self.qemu:
            status["qemu"] = self.qemu.get_boot_summary()
        if self.network:
            status["network"] = self.network.get_status()

        return status

    def stop(self) -> None:
        """Gracefully stop the pipeline."""
        logger.info("Pipeline stop requested")
        if self.qemu:
            self.qemu.stop()
        if self.network:
            self.network.teardown()
        if self.agent:
            self.agent.shutdown()
        self._set_phase(PipelinePhase.FAILED)
        self.state.final_status = "stopped"
