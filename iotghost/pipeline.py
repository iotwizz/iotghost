"""Main orchestration pipeline -- drives the full firmware emulation workflow.

Chains the phases: Extract -> Analyze -> Prepare -> Emulate -> Fix -> Verify.
The AI agent (ShellAgent) drives each phase autonomously, receiving phase-specific
prompts and using shell tools to accomplish each step. The pipeline manages
state transitions, retry logic, and coordination between modules.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
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
    BINARY_FIX_PROMPT,
    EMULATE_PROMPT,
    EXTRACT_PROMPT,
    FIX_PROMPT,
    KERNEL_BUILD_PROMPT,
    NVRAM_RECOVERY_PROMPT,
    PREPARE_PROMPT,
    QEMU_BOOT_FIX_PROMPT,
    SYSTEM_PROMPT,
    VENDOR_RECOVERY_PROMPTS,
    VERIFY_PROMPT,
)
from iotghost.tools import register_all_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boot diagnosis helpers
# ---------------------------------------------------------------------------

@dataclass
class BootDiagnosis:
    """Structured diagnosis of a QEMU boot failure."""
    failure_type: str       # kernel_panic, init_fail, nvram_crash, lib_missing,
                            # arch_mismatch, network_fail, qemu_crash, unknown
    root_cause: str         # human-readable root cause
    recommended_fix: str    # actionable fix description
    raw_evidence: str       # the matching log line(s)
    binary_path: str = ""   # relevant binary if applicable


# Ordered by priority -- first match wins
_BOOT_FAILURE_PATTERNS: list[tuple[str, re.Pattern, str, str]] = [
    (
        "arch_mismatch",
        re.compile(
            r"(?:Invalid ELF|ELF binary.*unexpected|Exec format error"
            r"|cannot execute binary|wrong ELF class)",
            re.IGNORECASE,
        ),
        "Architecture mismatch: kernel and rootfs binaries are for different CPUs.",
        "Verify kernel arch matches rootfs: run `file <kernel>` and `file <rootfs>/bin/busybox`. "
        "If mismatched, select the correct pre-built kernel or build one with Buildroot.",
    ),
    (
        "kernel_panic",
        re.compile(
            r"Kernel panic.*(?:not syncing|VFS|Unable to mount root|No init found"
            r"|No working init|Attempted to kill init)",
            re.IGNORECASE,
        ),
        "Kernel panic during boot: {match}",
        "Check: 1) root= parameter matches actual device (sda1/vda), "
        "2) kernel has driver for rootfs filesystem type (ext4/squashfs), "
        "3) /sbin/init or /etc/init.d/rcS exists in rootfs. "
        "Try booting with init=/bin/sh to isolate kernel vs userspace.",
    ),
    (
        "init_fail",
        re.compile(
            r"(?:can't run '/etc/init\.d/rcS'|can't run '/sbin/init'"
            r"|Failed to execute /init|Run /sbin/init as init process"
            r"|No init found|applet not found)",
            re.IGNORECASE,
        ),
        "Init system failure: the init binary or script is missing/broken.",
        "Check: 1) /sbin/init exists and is executable, "
        "2) /etc/inittab references correct paths, "
        "3) busybox applets are properly linked. "
        "Try creating a minimal init: #!/bin/sh\\nexec /etc/init.d/rcS",
    ),
    (
        "nvram_crash",
        re.compile(
            r"(?:nvram_get.*(?:not found|segfault|SIGSEGV)"
            r"|libnvram.*(?:not found|No such)"
            r"|LD_PRELOAD.*(?:cannot|failed|not found)"
            r"|nvram.*(?:Segmentation fault|core dumped))",
            re.IGNORECASE,
        ),
        "NVRAM emulation failure: services crash because libnvram interception is broken.",
        "Verify: 1) libnvram.so exists at LD_PRELOAD path AND matches rootfs arch, "
        "2) nvram.ini has required vendor keys, "
        "3) LD_PRELOAD path is absolute inside chroot. "
        "Test: chroot <rootfs> qemu-<arch>-static -E LD_PRELOAD=/libnvram.so /bin/sh",
    ),
    (
        "lib_missing",
        re.compile(
            r"(?:error while loading shared libraries"
            r"|cannot open shared object file"
            r"|NEEDED.*not found)",
            re.IGNORECASE,
        ),
        "Missing shared library: a critical binary cannot find required .so files.",
        "Run `readelf -d <binary> | grep NEEDED` to list deps, "
        "then `find <rootfs>/lib -name '<lib>'`. Copy missing libs or create symlinks.",
    ),
    (
        "lib_missing",
        re.compile(r"Segmentation fault|SIGSEGV|Illegal instruction|SIGILL", re.IGNORECASE),
        "Binary crash (segfault/illegal instruction): likely arch mismatch or missing library.",
        "Run `readelf -h <binary>` to verify architecture. "
        "Then `strace -f <binary>` to find failing syscall.",
    ),
    (
        "network_fail",
        re.compile(
            r"(?:Network is unreachable|SIOCSIFADDR.*No such device"
            r"|eth0.*not found|br0.*does not exist)",
            re.IGNORECASE,
        ),
        "Network configuration failure inside emulated firmware.",
        "Check QEMU NIC model matches kernel driver, firmware interface names, "
        "and ifconfig/ip commands in init scripts.",
    ),
    (
        "qemu_crash",
        re.compile(
            r"(?:qemu.*(?:Segmentation fault|abort|core dump)"
            r"|qemu:.*(?:unsupported|unhandled)"
            r"|QEMU.*(?:error|fatal)"
            r"|Could not open.*No such file)",
            re.IGNORECASE,
        ),
        "QEMU process itself crashed or failed to start.",
        "Check: 1) qemu-system-<arch> installed and correct, "
        "2) kernel file exists and is valid ELF, "
        "3) rootfs image valid and non-empty, "
        "4) machine type (-M) compatible with kernel.",
    ),
]


def _diagnose_boot_failure(serial_log: str, qemu_stderr: str = "") -> BootDiagnosis:
    """Parse serial log and QEMU stderr for known failure signatures."""
    combined = serial_log + "\n" + qemu_stderr

    for failure_type, pattern, cause_tpl, fix_tpl in _BOOT_FAILURE_PATTERNS:
        match = pattern.search(combined)
        if match:
            matched_text = match.group(0)[:200]
            match_start = match.start()
            ctx_start = combined.rfind("\n", 0, max(0, match_start - 200))
            ctx_end = combined.find("\n", min(len(combined), match_start + 300))
            evidence = combined[max(0, ctx_start):min(len(combined), ctx_end)].strip()

            root_cause = cause_tpl.format(match=matched_text) if "{match}" in cause_tpl else cause_tpl

            binary_path = ""
            bin_match = re.search(r"(/\S+(?:httpd|nginx|sbin/init|bin/sh|lighttpd|uhttpd))", evidence)
            if bin_match:
                binary_path = bin_match.group(1)

            return BootDiagnosis(
                failure_type=failure_type,
                root_cause=root_cause,
                recommended_fix=fix_tpl,
                raw_evidence=evidence[:500],
                binary_path=binary_path,
            )

    last_lines = "\n".join(combined.strip().splitlines()[-20:])
    return BootDiagnosis(
        failure_type="unknown",
        root_cause="Boot failed but no recognized error pattern found in log.",
        recommended_fix="Try: 1) init=/bin/sh to test basic kernel+rootfs, "
                        "2) Check dmesg for driver errors, "
                        "3) Verify QEMU machine type and CPU model.",
        raw_evidence=last_lines[:500],
    )


def _detect_vendor(rootfs_path: str) -> str | None:
    """Detect firmware vendor from rootfs file contents."""
    rootfs = Path(rootfs_path)
    vendor_indicators: dict[str, list[str]] = {
        "tenda": ["tenda", "Tenda", "TENDA", "tdcore", "tdhttpd"],
        "dlink": ["D-Link", "dlink", "DLINK", "mydlink"],
        "tplink": ["TP-LINK", "TP-Link", "tplink"],
        "netgear": ["NETGEAR", "Netgear", "netgear"],
        "asus": ["ASUS", "Asus", "RT-AC", "RT-N", "asuswrt"],
        "hikvision": ["Hikvision", "hikvision", "HIKVISION", "hi35"],
        "dahua": ["Dahua", "dahua", "DAHUA"],
    }
    scan_paths = [
        "etc/banner", "etc/issue", "etc/version", "etc/os-release",
        "www/index.html", "www/login.html", "web/index.html",
    ]
    for scan_rel in scan_paths:
        scan_file = rootfs / scan_rel
        if scan_file.is_file():
            try:
                content = scan_file.read_text(errors="replace")[:4096]
                for vendor, indicators in vendor_indicators.items():
                    for indicator in indicators:
                        if indicator in content:
                            logger.info("Vendor detected: %s (found '%s' in %s)", vendor, indicator, scan_rel)
                            return vendor
            except OSError:
                continue
    return None


def _select_fix_prompt(
    diagnosis: BootDiagnosis,
    rootfs_path: str,
    arch: str,
    vendor: str | None,
    previous_fixes: str,
) -> str:
    """Select the best fix prompt based on boot failure diagnosis.

    Returns a fully-formatted prompt string combining the structured
    FIX_PROMPT with diagnosis-specific expert guidance.
    """
    # Base fix prompt with diagnosis
    base_prompt = FIX_PROMPT.format(
        diagnosis_summary=f"[{diagnosis.failure_type.upper()}] {diagnosis.root_cause}",
        root_cause=diagnosis.root_cause,
        recommended_fix=diagnosis.recommended_fix,
        error_output=diagnosis.raw_evidence,
        previous_fixes=previous_fixes or "None yet",
    )

    # Add expert prompt based on failure type
    expert_section = ""

    if diagnosis.failure_type in ("kernel_panic", "arch_mismatch"):
        expert_section = "\n\n## EXPERT: KERNEL RECOVERY\n" + KERNEL_BUILD_PROMPT
    elif diagnosis.failure_type == "lib_missing":
        expert_section = "\n\n## EXPERT: BINARY/LIBRARY RECOVERY\n" + BINARY_FIX_PROMPT
    elif diagnosis.failure_type == "nvram_crash":
        try:
            expert_section = "\n\n## EXPERT: NVRAM RECOVERY\n" + NVRAM_RECOVERY_PROMPT.format(
                rootfs_path=rootfs_path,
                arch=arch,
                vendor=vendor or "unknown",
            )
        except KeyError as exc:
            logger.warning("NVRAM_RECOVERY_PROMPT format error (missing key %s), using raw", exc)
            expert_section = "\n\n## EXPERT: NVRAM RECOVERY\n" + NVRAM_RECOVERY_PROMPT
    elif diagnosis.failure_type == "qemu_crash":
        try:
            expert_section = "\n\n## EXPERT: QEMU BOOT FIX\n" + QEMU_BOOT_FIX_PROMPT.format(
                diagnosis=diagnosis.root_cause,
                serial_evidence=diagnosis.raw_evidence[:300],
                recommended_fix=diagnosis.recommended_fix,
            )
        except KeyError as exc:
            logger.warning("QEMU_BOOT_FIX_PROMPT format error (missing key %s), using raw", exc)
            expert_section = "\n\n## EXPERT: QEMU BOOT FIX\n" + QEMU_BOOT_FIX_PROMPT

    # Add vendor-specific guidance if available
    vendor_section = ""
    if vendor and vendor in VENDOR_RECOVERY_PROMPTS:
        vendor_section = "\n\n## VENDOR-SPECIFIC GUIDANCE\n" + VENDOR_RECOVERY_PROMPTS[vendor]

    return base_prompt + expert_section + vendor_section


# ---------------------------------------------------------------------------
# Pipeline phases and state
# ---------------------------------------------------------------------------

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
    # --- New fields for diagnosis-aware emulation ---
    vendor: str | None = None
    emulation_status: str = ""
    degraded_boot: bool = False
    degraded_diagnosis: str = ""

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
        on_tool_call: Callable | None = None,
        on_tool_result: Callable | None = None,
        on_status_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.state = PipelineState()

        # Callbacks for TUI integration
        self._on_phase_change = on_phase_change
        self._on_agent_message = on_agent_message
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
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

        # Initialize AI agent with all callbacks
        self.agent = ShellAgent(
            config=self.config.agent_config,
            on_message=lambda msg: (
                self._on_agent_message(msg.content)
                if self._on_agent_message and msg.content
                else None
            ),
            on_tool_call=self._on_tool_call or None,
            on_tool_result=self._on_tool_result or None,
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
            # Truncate status to avoid dumping LLM output into the summary
            err_msg = str(exc)
            if len(err_msg) > 200:
                err_msg = err_msg[:200] + "..."
            self.state.final_status = f"error: {type(exc).__name__}: {err_msg}"
            self.state.last_error = err_msg
            self._set_phase(PipelinePhase.FAILED)
        finally:
            self._cleanup()

        return self.state

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _run_extract(self) -> None:
        """Phase 1: Extract firmware rootfs.

        Runs up to 3 extraction attempts with progressively more aggressive
        strategies. After each agent run we validate that a real rootfs
        directory exists on disk -- the agent's textual summary is never
        trusted as proof of success.
        """
        self._set_phase(PipelinePhase.EXTRACT)

        max_extract_retries = 3
        extraction_strategies = [
            # Attempt 1: standard binwalk via agent
            EXTRACT_PROMPT.format(firmware_path=self.state.firmware_path),
            # Attempt 2: force recursive matryoshka + sasquatch
            (
                "The previous extraction attempt did NOT produce a valid root "
                "filesystem.  Try these strategies in order and stop as soon as "
                "one yields a directory containing /bin, /etc, /lib, /sbin:\n"
                "1. binwalk -Me -C {workdir} {firmware_path}  (recursive matryoshka)\n"
                "2. For each SquashFS blob found by binwalk, try: "
                "sasquatch -d {workdir}/squashfs-root <blob>\n"
                "3. If a TRX header is present, skip the first 28-64 bytes and "
                "re-run binwalk on the payload: "
                "dd if={firmware_path} bs=1 skip=<offset> | binwalk -Me -C {workdir}\n"
                "Report the EXACT path to the extracted rootfs directory."
            ).format(
                workdir=self.state.workdir,
                firmware_path=self.state.firmware_path,
            ),
            # Attempt 3: brute-force offset carving
            (
                "Extraction is still failing.  Use a brute-force approach:\n"
                "1. Run: binwalk -E {firmware_path} to check for encryption "
                "(high entropy > 0.95 across the image means encrypted -- report and stop).\n"
                "2. List every SquashFS/CramFS/JFFS2 signature offset: "
                "binwalk -y squashfs -y cramfs -y jffs2 {firmware_path}\n"
                "3. For each offset, carve and extract: "
                "dd if={firmware_path} bs=1 skip=<decimal_offset> of={workdir}/carved.bin && "
                "binwalk -e -C {workdir}/carved_out {workdir}/carved.bin\n"
                "Report the EXACT path to the extracted rootfs directory, or "
                "explain precisely why extraction is impossible."
            ).format(
                workdir=self.state.workdir,
                firmware_path=self.state.firmware_path,
            ),
        ]

        rootfs = None
        last_agent_output = ""

        for attempt in range(max_extract_retries):
            prompt = extraction_strategies[attempt]
            self.agent.inject_context(prompt)
            last_agent_output = self.agent.run_until_done(max_iterations=15)

            # --- Ground-truth validation (never trust the agent's summary) ---
            # 1. Check if the agent explicitly set the rootfs path
            rootfs = self.agent.get_context_var("rootfs_path")
            if rootfs and Path(rootfs).is_dir():
                # Verify it actually looks like a rootfs
                children = {c.name for c in Path(rootfs).iterdir() if c.is_dir()}
                if {"bin", "etc", "lib", "sbin"}.issubset(children):
                    break
                else:
                    logger.warning(
                        "Agent-reported rootfs '%s' missing core dirs (has: %s), "
                        "falling back to search",
                        rootfs, children,
                    )
                    rootfs = None

            # 2. Search multiple locations -- binwalk extracts relative to
            #    CWD or firmware parent, not necessarily our workdir.
            search_dirs = [
                self.state.workdir,
                str(Path(self.state.firmware_path).parent),
                self.state.firmware_path + ".extracted",
            ]
            # Also check <workdir>/<firmware_stem>.extracted
            fw_stem = Path(self.state.firmware_path).name
            search_dirs.append(
                str(Path(self.state.workdir) / (fw_stem + ".extracted"))
            )
            # And binwalk's _<name>.extracted pattern
            search_dirs.append(
                str(Path(self.state.workdir) / f"_{fw_stem}.extracted")
            )

            if not rootfs:
                for search_dir in search_dirs:
                    if Path(search_dir).is_dir():
                        rootfs = find_rootfs(search_dir)
                        if rootfs:
                            logger.info(
                                "Found rootfs via expanded search in %s",
                                search_dir,
                            )
                            break

            if rootfs:
                break

            searched_str = ", ".join(
                d for d in search_dirs if Path(d).is_dir()
            )
            logger.warning(
                "Extraction attempt %d/%d: no valid rootfs found in [%s], retrying",
                attempt + 1, max_extract_retries, searched_str,
            )
            # Feed the failure back so the agent knows it failed
            self.agent.inject_context(
                "[SYSTEM] The previous extraction attempt FAILED -- no directory "
                "containing /bin, /etc, /lib, /sbin was found on disk. "
                f"Searched: {searched_str}. "
                "Your summary was incorrect. Try the next strategy."
            )

        if not rootfs:
            searched_str = ", ".join(
                d for d in [
                    self.state.workdir,
                    str(Path(self.state.firmware_path).parent),
                ] if Path(d).is_dir()
            )
            raise RuntimeError(
                f"Extraction failed after {max_extract_retries} attempts: "
                "could not locate root filesystem. "
                f"Searched: [{searched_str}]"
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
            try:
                actions = deploy_nvram(
                    self.state.rootfs_path,
                    self.state.nvram_config,
                    device_ip=self.config.device_ip,
                )
                for action in actions:
                    logger.info("NVRAM: %s", action)
            except FileExistsError as exc:
                logger.warning("NVRAM deploy hit existing path (%s), continuing", exc)

        # Let agent do additional preparation
        prompt = PREPARE_PROMPT.format(
            rootfs_path=self.state.rootfs_path,
            architecture=self.state.architecture,
            needs_nvram=self.state.needs_nvram,
        )
        self.agent.inject_context(prompt)

        try:
            self.agent.run_until_done(max_iterations=15)
        except FileExistsError as exc:
            # Agent's shell commands may trigger FileExistsError via
            # Python shutil/pathlib calls.  Inject recovery guidance
            # and let the agent try once more with idempotent commands.
            logger.warning("Prepare phase hit FileExistsError: %s -- retrying", exc)
            recovery = (
                f"[SYSTEM] A FileExistsError occurred: {exc}\n"
                "The target path already exists from the firmware extraction.\n"
                "Resume preparation using ONLY idempotent commands:\n"
                "  mkdir -p, cp -af, ln -sf, mknod only after 'test -e'.\n"
                "Do NOT re-create directories that already exist."
            )
            self.agent.inject_context(recovery)
            self.agent.run_until_done(max_iterations=10)

        # Create rootfs disk image for QEMU
        image_path = str(Path(self.state.workdir) / "rootfs.img")
        try:
            self.state.rootfs_image = create_rootfs_image(
                self.state.rootfs_path,
                image_path,
                size_mb=256,
            )
        except subprocess.CalledProcessError as exc:
            cmd_str = " ".join(str(a) for a in (exc.cmd or []))
            stderr = (exc.stderr or b"").decode(errors="replace")[:300]
            raise RuntimeError(
                f"Failed to create rootfs image (exit {exc.returncode}).\n"
                f"Command: {cmd_str}\n"
                f"Stderr: {stderr}\n\n"
                "Fix: install genext2fs (apt install genext2fs) or run with sudo.\n"
                "If you don't need network TAP mode, try: --network user"
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"Permission denied creating rootfs image: {exc}\n\n"
                "Fix: install genext2fs (apt install genext2fs) or run with sudo."
            ) from exc

        logger.info("Preparation complete: image=%s", self.state.rootfs_image)

    @staticmethod
    def _detect_arch_from_rootfs(rootfs_path: str) -> str:
        """Fallback architecture detection by running 'file' on rootfs binaries."""
        arch_keywords = {
            "MIPS": "mipsel",
            "MIPS, MIPS-I": "mips",
            "MIPS, MIPS32": "mipsel",
            "ARM,": "arm",
            "ARM aarch64": "aarch64",
            "x86-64": "x86_64",
            "Intel 80386": "i386",
            "PowerPC": "ppc",
        }
        # Check common binaries in rootfs
        candidates = ["bin/busybox", "bin/sh", "sbin/init", "lib/libc.so.0"]
        rootfs = Path(rootfs_path)
        for candidate in candidates:
            target = rootfs / candidate
            if target.exists():
                try:
                    result = subprocess.run(
                        ["file", str(target)],
                        capture_output=True, text=True, timeout=5,
                    )
                    output = result.stdout
                    for keyword, arch in arch_keywords.items():
                        if keyword in output:
                            logger.info(
                                "Fallback arch detection: found '%s' in %s -> %s",
                                keyword, candidate, arch,
                            )
                            return arch
                except (subprocess.SubprocessError, OSError):
                    continue
        return "unknown"

    def _preflight_checks(self) -> list[str]:
        """Run deterministic pre-flight checks before QEMU launch.

        Verifies kernel/rootfs compatibility and fixes common issues
        automatically without wasting agent iterations.

        Returns list of issues found and auto-fixed.
        """
        issues_fixed: list[str] = []
        state = self.state
        rootfs_path = str(state.rootfs_path) if state.rootfs_path else None

        if not rootfs_path or not Path(rootfs_path).exists():
            logger.warning("Preflight: rootfs path not available, skipping checks")
            return issues_fixed

        rootfs = Path(rootfs_path)

        # 1. Check init path exists
        init_paths = [
            rootfs / "sbin" / "init",
            rootfs / "etc" / "init.d" / "rcS",
            rootfs / "init",
            rootfs / "bin" / "sh",
        ]
        init_found = any(p.exists() for p in init_paths)
        if not init_found:
            # Create a minimal init script
            init_dir = rootfs / "etc" / "init.d"
            init_dir.mkdir(parents=True, exist_ok=True)
            rcs = init_dir / "rcS"
            rcs.write_text(
                "#!/bin/sh\nmount -t proc proc /proc\n"
                "mount -t sysfs sysfs /sys\nexec /bin/sh\n"
            )
            rcs.chmod(0o755)
            # Also create /sbin/init -> /etc/init.d/rcS
            sbin = rootfs / "sbin"
            sbin.mkdir(parents=True, exist_ok=True)
            sbin_init = sbin / "init"
            if not sbin_init.exists():
                sbin_init.symlink_to("/etc/init.d/rcS")
            issues_fixed.append("Created missing /etc/init.d/rcS and /sbin/init")
            logger.info("Preflight: created missing init scripts")

        # 2. Check critical device nodes
        dev_dir = rootfs / "dev"
        dev_dir.mkdir(parents=True, exist_ok=True)
        required_nodes = {
            "console": (5, 1),    # char major 5, minor 1
            "null": (1, 3),       # char major 1, minor 3
            "ttyS0": (4, 64),     # char major 4, minor 64
            "ttyAMA0": (204, 64), # char major 204, minor 64
            "zero": (1, 5),       # char major 1, minor 5
        }
        for name, (major, minor) in required_nodes.items():
            node_path = dev_dir / name
            if not node_path.exists():
                try:
                    os.mknod(
                        str(node_path),
                        0o666 | 0o020000,
                        os.makedev(major, minor),
                    )
                    issues_fixed.append(f"Created /dev/{name}")
                except (OSError, PermissionError) as e:
                    # mknod may require root -- log but don't fail
                    logger.debug("Preflight: cannot create /dev/%s: %s", name, e)

        # 3. Check /proc and /sys mountpoints exist
        for mp in ["proc", "sys", "tmp"]:
            mp_path = rootfs / mp
            if not mp_path.exists():
                mp_path.mkdir(parents=True, exist_ok=True)
                issues_fixed.append(f"Created /{mp} mountpoint")

        # 4. Verify NVRAM setup if needed
        if state.needs_nvram:
            nvram_lib_paths = [
                rootfs / "usr" / "lib" / "libnvram.so",
                rootfs / "lib" / "libnvram.so",
                rootfs / "usr" / "lib" / "libnvram-0.3.so",
            ]
            nvram_found = any(p.exists() for p in nvram_lib_paths)
            if not nvram_found:
                issues_fixed.append(
                    "WARNING: NVRAM needed but libnvram.so not found in rootfs"
                )
                logger.warning(
                    "Preflight: libnvram.so not found -- NVRAM emulation may fail"
                )

            nvram_ini = rootfs / "etc" / "nvram.ini"
            if not nvram_ini.exists():
                issues_fixed.append(
                    "WARNING: NVRAM needed but /etc/nvram.ini not found"
                )
                logger.warning("Preflight: nvram.ini missing")

        # 5. Check rootfs image file (if separate from directory)
        if state.rootfs_image and Path(state.rootfs_image).exists():
            img_size = Path(state.rootfs_image).stat().st_size
            if img_size < 1024:
                issues_fixed.append(
                    f"WARNING: rootfs image is very small ({img_size} bytes)"
                )
                logger.warning(
                    "Preflight: rootfs image is only %d bytes", img_size
                )

        if issues_fixed:
            logger.info(
                "Preflight checks: %d issues found/fixed", len(issues_fixed)
            )
        else:
            logger.info("Preflight checks: all clear")

        return issues_fixed

    def _run_emulate_loop(self) -> None:
        """Phase 4+5: Emulate with fix loop.

        Tries to boot in QEMU. If boot fails, enters FIX phase
        to diagnose and repair, then retries. Loops up to max_fix_attempts.
        """
        self._set_phase(PipelinePhase.EMULATE)

        # --- Guard: architecture must be known ---
        arch = self.state.architecture
        if not arch or arch == "unknown":
            # Try fallback detection from rootfs binaries
            rootfs = self.state.rootfs_path
            if rootfs:
                arch = self._detect_arch_from_rootfs(rootfs)
                self.state.architecture = arch

        if not arch or arch == "unknown":
            raise RuntimeError(
                "Cannot emulate: firmware architecture is 'unknown'.\n"
                "The analysis phase failed to detect CPU architecture.\n"
                "Please specify it manually with --arch (e.g. --arch mipsel)"
            )

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

        # Setup network (only for tap mode; user mode needs no host setup)
        if self.config.network_mode == "tap":
            try:
                host_iface = detect_host_interface() or "eth0"
                net_config = NetworkConfig(
                    device_ip=self.config.device_ip,
                    nat_interface=host_iface,
                    qemu_mode="tap",
                )
                self.network = NetworkManager(net_config)
                self.network.setup()
            except (subprocess.CalledProcessError, PermissionError, OSError) as exc:
                logger.warning(
                    "TAP network setup failed (%s), falling back to user mode", exc
                )
                qemu_config.network_mode = "user"
                self.network = None
        else:
            logger.info("Using QEMU user-mode networking (no host setup needed)")

        # --- Pre-flight checks ---
        preflight_issues = self._preflight_checks()
        if preflight_issues:
            logger.info(
                "Preflight fixed %d issues: %s",
                len(preflight_issues), preflight_issues,
            )

        # Detect vendor for vendor-specific recovery
        vendor = (
            _detect_vendor(str(self.state.rootfs_path))
            if self.state.rootfs_path
            else None
        )
        if vendor:
            logger.info("Detected vendor: %s", vendor)
            self.state.vendor = vendor  # Store for later use

        previous_fixes: list[str] = []  # Track what we've tried
        max_fix_attempts = self.config.max_fix_attempts
        serial_path = Path(self.state.workdir) / "serial.log"
        diagnosis: BootDiagnosis | None = None  # last diagnosis for degraded boot

        while self.state.fix_attempts <= max_fix_attempts:
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

            if self.state.fix_attempts > max_fix_attempts:
                logger.error(
                    "Max fix attempts (%d) reached. Giving up.",
                    max_fix_attempts,
                )
                break  # Fall through to degraded boot

            # Stop current QEMU instance
            self.qemu.stop()

            # Run fix phase with structured diagnosis
            self._set_phase(PipelinePhase.FIX)

            # Read FULL serial log for diagnosis
            serial_log = ""
            if serial_path.exists():
                try:
                    serial_log = serial_path.read_text(errors="replace")
                except OSError:
                    serial_log = ""

            # Get QEMU stderr if available
            qemu_stderr = ""
            if (
                hasattr(self.qemu, '_process')
                and self.qemu._process
                and self.qemu._process.stderr
            ):
                try:
                    qemu_stderr = self.qemu._process.stderr.read() or ""
                except Exception:
                    pass

            # Structured diagnosis
            diagnosis = _diagnose_boot_failure(serial_log, qemu_stderr)
            logger.info(
                "Boot diagnosis: type=%s cause=%s",
                diagnosis.failure_type, diagnosis.root_cause[:100],
            )

            # Select context-specific fix prompt
            fix_prompt = _select_fix_prompt(
                diagnosis=diagnosis,
                rootfs_path=str(self.state.rootfs_path or ""),
                arch=str(self.state.architecture or ""),
                vendor=getattr(self.state, 'vendor', None),
                previous_fixes="\n".join(
                    f"  Attempt {i+1}: {f}"
                    for i, f in enumerate(previous_fixes)
                ),
            )

            self.agent.inject_context(fix_prompt)
            fix_result = self.agent.run_until_done(max_iterations=10)
            self.agent.state.attempted_fixes.append(fix_result[:200])

            # Record what was tried
            previous_fixes.append(
                f"[{diagnosis.failure_type}] Agent attempted fix for: "
                f"{diagnosis.root_cause[:80]}"
            )

            # Back to emulate phase for retry
            self._set_phase(PipelinePhase.EMULATE)
            logger.info(
                "Fix attempt %d complete, retrying emulation",
                self.state.fix_attempts,
            )

        # --- Degraded mode boot attempt ---
        logger.warning(
            "All %d fix attempts exhausted. Trying degraded boot "
            "(init=/bin/sh)...",
            max_fix_attempts,
        )

        # Stop QEMU if still running
        if self.qemu and self.qemu.is_running():
            self.qemu.stop()

        # Modify kernel append to use init=/bin/sh
        original_append = qemu_config.kernel_append or ""
        degraded_append = (
            re.sub(r'init=\S+', '', original_append).strip()
            + " init=/bin/sh"
        )
        qemu_config.kernel_append = degraded_append

        try:
            self.qemu = QemuManager(
                config=qemu_config,
                on_serial_line=lambda line: logger.debug("SERIAL: %s", line),
            )
            self.qemu.start()
            time.sleep(10)  # Give it time to boot to shell

            # Check if we got a shell prompt
            degraded_serial = ""
            if serial_path.exists():
                degraded_serial = serial_path.read_text(errors="replace")

            if (
                "/ #" in degraded_serial
                or "~ #" in degraded_serial
                or "/bin/sh" in degraded_serial
            ):
                logger.info(
                    "Degraded boot SUCCESS: kernel + rootfs basic stack works"
                )
                self.state.emulation_status = "PARTIAL"
                self.state.degraded_boot = True
                last_cause = (
                    diagnosis.root_cause if diagnosis else "unknown"
                )
                self.state.degraded_diagnosis = (
                    "Kernel and rootfs are functional (shell accessible), "
                    "but userspace services failed to start. "
                    f"Original failure: {last_cause}"
                )
                # Restore original append for potential retry
                qemu_config.kernel_append = original_append
                return  # Continue to VERIFY with partial status
            else:
                degraded_diag = _diagnose_boot_failure(degraded_serial)
                logger.warning(
                    "Degraded boot also failed: %s",
                    degraded_diag.root_cause,
                )
                self.state.emulation_status = "FAILED"
                self.state.degraded_diagnosis = (
                    "Both normal and degraded boot failed. "
                    f"Fundamental issue: {degraded_diag.root_cause}"
                )
        except Exception as e:
            logger.error("Degraded boot exception: %s", e)
            self.state.degraded_diagnosis = f"Degraded boot crashed: {e}"
        finally:
            try:
                if self.qemu:
                    self.qemu.stop()
            except Exception:
                pass
            # Restore original append
            qemu_config.kernel_append = original_append

        self._set_phase(PipelinePhase.FAILED)
        self.state.final_status = "failed_max_retries"

    def _run_verify(self) -> None:
        """Phase 6: Verify the firmware is running and services are accessible."""
        self._set_phase(PipelinePhase.VERIFY)

        # Include degraded boot info in verify prompt if applicable
        extra_context = ""
        if self.state.degraded_boot:
            extra_context = (
                "\n\nNOTE: Firmware booted in DEGRADED MODE (init=/bin/sh). "
                "Full userspace services are NOT running. "
                f"Diagnosis: {self.state.degraded_diagnosis}\n"
                "Verify what IS accessible: check if kernel booted, "
                "if basic shell commands work, and report PARTIAL status."
            )

        prompt = VERIFY_PROMPT.format(
            device_ip=self.config.device_ip,
        )
        if extra_context:
            prompt += extra_context

        self.agent.inject_context(prompt)
        result = self.agent.run_until_done(max_iterations=10)

        # Parse verification results
        if self.state.degraded_boot:
            self.state.final_status = "partial"
        elif "RUNNING" in result.upper():
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

        if self.state.degraded_boot:
            status["degraded_boot"] = True
            status["degraded_diagnosis"] = self.state.degraded_diagnosis
        if self.state.vendor:
            status["vendor"] = self.state.vendor

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
