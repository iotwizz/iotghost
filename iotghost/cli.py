"""Click CLI entry point for IoTGhost.

Usage:
    iotghost run firmware.bin --ollama --model glm4:latest
    iotghost run firmware.bin --ollama --model llama3.1:8b --verbose
    iotghost run firmware.bin --arch mipsel --network user
    iotghost info firmware.bin
    iotghost check-deps
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from iotghost import __version__

console = Console()


def _setup_logging(verbose: bool, debug: bool) -> None:
    """Configure logging based on verbosity flags."""
    if debug:
        level = logging.DEBUG
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    elif verbose:
        level = logging.INFO
        fmt = "%(levelname)s %(message)s"
    else:
        level = logging.WARNING
        fmt = "%(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__, prog_name="iotghost")
def main() -> None:
    """IoTGhost -- AI-powered IoT firmware emulator.

    Feed it a firmware binary, it handles extraction, architecture detection,
    NVRAM emulation, kernel configuration, network setup, and iterative error
    correction -- all driven autonomously by an LLM agent.
    """


# ---------------------------------------------------------------------------
# run command -- the main entrypoint
# ---------------------------------------------------------------------------

@main.command()
@click.argument("firmware", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--ollama", "provider", flag_value="ollama", default=True,
    help="Use Ollama as LLM provider (default).",
)
@click.option(
    "--model", default="glm4:latest",
    help="LLM model name [default: glm4:latest].",
    show_default=True,
)
@click.option(
    "--base-url", default="http://localhost:11434",
    help="Ollama API base URL.",
    show_default=True,
)
@click.option(
    "--arch",
    type=click.Choice(["mipsel", "mipsbe", "armel", "arm64", "x86", "x86_64", "auto"]),
    default="auto",
    help="Force architecture (auto-detect by default).",
    show_default=True,
)
@click.option(
    "--network",
    type=click.Choice(["tap", "user", "none"]),
    default="user",
    help="Network mode: tap (requires root), user (unprivileged), none.",
    show_default=True,
)
@click.option(
    "--device-ip", default="192.168.1.1",
    help="IP address for the emulated device.",
    show_default=True,
)
@click.option(
    "--timeout", default=120, type=int,
    help="Boot timeout in seconds.",
    show_default=True,
)
@click.option(
    "--max-retries", default=5, type=int,
    help="Maximum fix-and-retry attempts.",
    show_default=True,
)
@click.option(
    "--max-iterations", default=50, type=int,
    help="Maximum LLM iterations per phase.",
    show_default=True,
)
@click.option(
    "--workdir", default=None, type=click.Path(),
    help="Working directory (default: auto-created next to firmware).",
)
@click.option(
    "--kernels-dir", default=None, type=click.Path(exists=True),
    help="Directory containing pre-built QEMU kernels.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Show detailed progress output.",
)
@click.option(
    "--debug", is_flag=True, default=False,
    help="Enable debug logging (very verbose).",
)
@click.option(
    "--no-tui", is_flag=True, default=False,
    help="Disable Rich TUI, use plain text output.",
)
@click.option(
    "--temperature", default=0.1, type=float,
    help="LLM temperature (lower = more deterministic).",
    show_default=True,
)
@click.option(
    "--nollama", "--no-ollama", is_flag=True, default=False,
    help="Skip local Ollama -- use --base-url for a remote/cloud LLM endpoint.",
)
def run(
    firmware: str,
    provider: str,
    model: str,
    base_url: str,
    arch: str,
    network: str,
    device_ip: str,
    timeout: int,
    max_retries: int,
    max_iterations: int,
    workdir: str | None,
    kernels_dir: str | None,
    verbose: bool,
    debug: bool,
    no_tui: bool,
    temperature: float,
    nollama: bool,
) -> None:
    """Emulate an IoT firmware image using AI-driven QEMU emulation.

    FIRMWARE is the path to the firmware binary (.bin, .img, .trx, etc.)

    \b
    Examples:
        iotghost run firmware.bin --ollama --model glm4:latest
        iotghost run DIR-850L_fw.bin --arch mipsel --verbose
        iotghost run camera.bin --network user --no-tui
    """
    _setup_logging(verbose, debug)

    from iotghost.agent import AgentConfig
    from iotghost.pipeline import EmulationPipeline, PipelineConfig

    # Print banner
    if not no_tui:
        _print_banner(firmware, model, arch, network)

    # When --nollama is set, treat the provider as a generic OpenAI-compatible
    # endpoint (no Ollama health check).  The user must supply --base-url and
    # --model pointing at their remote/cloud LLM.
    if nollama:
        provider = "openai_compatible"
        if base_url == "http://localhost:11434":
            console.print(
                "[bold yellow]Warning:[/] --nollama used but --base-url is still "
                "the default Ollama URL.  Pass --base-url <your-endpoint> to "
                "point at your cloud LLM."
            )

    # Build configuration
    agent_config = AgentConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    pipeline_config = PipelineConfig(
        firmware_path=firmware,
        workdir=workdir or "",
        agent_config=agent_config,
        network_mode=network,
        device_ip=device_ip,
        max_fix_attempts=max_retries,
        boot_timeout=timeout,
        kernels_dir=kernels_dir,
        force_arch=arch if arch != "auto" else None,
    )

    # Initialize TUI or plain output
    tui = None
    if not no_tui:
        from iotghost.tui import EmulationTUI
        tui = EmulationTUI(verbose=verbose)

    # Build pipeline with callbacks
    # Use getattr for tool callbacks in case TUI class hasn't been updated yet
    pipeline = EmulationPipeline(
        config=pipeline_config,
        on_phase_change=tui.on_phase_change if tui else _plain_phase_change,
        on_agent_message=tui.on_agent_message if tui else _plain_message,
        on_tool_call=getattr(tui, "on_tool_call", _plain_tool_call) if tui else _plain_tool_call,
        on_tool_result=getattr(tui, "on_tool_result", _plain_tool_result) if tui else _plain_tool_result,
        on_status_update=tui.on_status_update if tui else None,
    )

    # Run
    try:
        if tui:
            tui.start()

        result = pipeline.run()

        if tui:
            tui.stop()
            tui.print_summary(pipeline.get_status())
        else:
            _print_plain_summary(pipeline.get_status())

        # Exit code based on result
        if result.final_status == "running":
            console.print("\n[bold green]Firmware is running! Access it at {device_ip}[/]")
            # Keep running until Ctrl+C
            if pipeline.qemu and pipeline.qemu.is_running():
                console.print("[dim]Press Ctrl+C to stop emulation[/]")
                try:
                    pipeline.qemu.state.process.wait()
                except KeyboardInterrupt:
                    console.print("\n[yellow]Stopping emulation...[/]")
                    pipeline.stop()
        elif result.final_status == "partial":
            console.print("\n[bold yellow]Partial success -- some services running[/]")
            sys.exit(0)
        else:
            console.print(f"\n[bold red]Emulation failed: {result.final_status}[/]")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Cleaning up...[/]")
        if tui:
            tui.stop()
        pipeline.stop()
        sys.exit(130)
    except ConnectionError as exc:
        console.print(f"\n[bold red]Connection error:[/] {exc}")
        console.print("[dim]Make sure Ollama is running: ollama serve[/]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# info command -- analyze without emulating
# ---------------------------------------------------------------------------

@main.command()
@click.argument("firmware", type=click.Path(exists=True, dir_okay=False))
@click.option("--verbose", "-v", is_flag=True)
def info(firmware: str, verbose: bool) -> None:
    """Analyze a firmware image without emulating it.

    Extracts and reports: architecture, filesystem type, NVRAM requirements,
    kernel version, and recommended QEMU configuration.
    """
    _setup_logging(verbose, False)

    from iotghost.extractors import detect_architecture, parse_binwalk_output
    from iotghost.tools.shell import execute_command

    console.print(f"[bold]Analyzing:[/] {firmware}\n")

    # Run binwalk scan
    console.print("[dim]Running binwalk scan...[/]")
    result = execute_command(f"binwalk {firmware}")
    if result.success:
        entries = parse_binwalk_output(result.output)
        console.print(f"[bold]Binwalk results:[/] {len(entries)} entries found")
        for entry in entries[:10]:
            fmt_str = f" [{entry.format_type.value}]" if entry.format_type.value != "unknown" else ""
            console.print(f"  {entry.offset_hex}: {entry.description[:80]}{fmt_str}")
    else:
        console.print(f"[red]Binwalk failed:[/] {result.output[:200]}")

    # Detect architecture
    console.print("\n[dim]Detecting architecture...[/]")
    result = execute_command(f"file {firmware}")
    if result.success:
        arch, endian = detect_architecture(result.output)
        console.print(f"[bold]Architecture:[/] {arch.value} ({endian}-endian)")
    else:
        console.print("[yellow]Could not detect architecture from file header[/]")

    # Check entropy (encryption detection)
    console.print("\n[dim]Checking entropy...[/]")
    result = execute_command(f"binwalk -E {firmware}")
    if result.success:
        from iotghost.extractors import check_encryption
        encrypted = check_encryption(result.output)
        if encrypted:
            console.print("[bold red]WARNING: Firmware appears to be ENCRYPTED[/]")
        else:
            console.print("[green]Firmware does not appear encrypted[/]")

    console.print(f"\n[dim]Use 'iotghost run {firmware}' to start emulation[/]")


# ---------------------------------------------------------------------------
# check-deps command -- verify system dependencies
# ---------------------------------------------------------------------------

@main.command("check-deps")
def check_deps() -> None:
    """Check if required system dependencies are installed."""
    from iotghost.tools.shell import execute_command

    deps = [
        ("binwalk", "binwalk --help", "Firmware extraction"),
        ("qemu-system-mipsel", "qemu-system-mipsel --version", "MIPS emulation"),
        ("qemu-system-arm", "qemu-system-arm --version", "ARM emulation"),
        ("qemu-system-aarch64", "qemu-system-aarch64 --version", "ARM64 emulation"),
        ("sasquatch", "sasquatch --help", "Vendor SquashFS extraction"),
        ("jefferson", "jefferson --help", "JFFS2 extraction"),
        ("unsquashfs", "unsquashfs --help", "SquashFS extraction"),
        ("cpio", "cpio --version", "CPIO extraction"),
        ("brctl", "brctl --version", "Network bridge control"),
    ]

    console.print("[bold]IoTGhost Dependency Check[/]\n")

    found = 0
    missing = 0

    for name, check_cmd, description in deps:
        result = execute_command(check_cmd, timeout=5)
        if result.success:
            console.print(f"  [green]OK[/]  {name:24s} {description}")
            found += 1
        else:
            console.print(f"  [red]--[/]  {name:24s} {description}")
            missing += 1

    # Check Ollama
    console.print()
    result = execute_command("curl -s http://localhost:11434/api/tags", timeout=5)
    if result.success and "models" in result.output:
        console.print(f"  [green]OK[/]  {'ollama':24s} LLM provider (running)")
        found += 1
    else:
        console.print(f"  [red]--[/]  {'ollama':24s} LLM provider (not running)")
        missing += 1

    console.print(f"\n[bold]{found} found, {missing} missing[/]")
    if missing > 0:
        console.print(
            "\n[dim]Install missing deps:\n"
            "  sudo apt install qemu-system binwalk squashfs-tools\n"
            "  pip install jefferson ubi_reader\n"
            "  # For sasquatch: https://github.com/devttys0/sasquatch\n"
            "  # For Ollama: https://ollama.ai[/]"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _print_banner(firmware: str, model: str, arch: str, network: str) -> None:
    """Print the IoTGhost startup banner."""
    console.print()
    console.print("[bold blue]  _       _    ____  _               _   [/]")
    console.print("[bold blue] (_) ___ | |_ / ___|| |__   ___  ___| |_ [/]")
    console.print("[bold blue] | |/ _ \\| __| |  _ | '_ \\ / _ \\/ __| __|[/]")
    console.print("[bold blue] | | (_) | |_| |_| || | | | (_) \\__ \\ |_ [/]")
    console.print("[bold blue] |_|\\___/ \\__|\\____||_| |_|\\___/|___/\\__|[/]")
    console.print()
    console.print(f"[bold]v{__version__}[/] -- AI-Powered IoT Firmware Emulator")
    console.print()
    console.print(f"  Firmware:  {firmware}")
    console.print(f"  Model:     {model}")
    console.print(f"  Arch:      {arch}")
    console.print(f"  Network:   {network}")
    console.print()


# ---------------------------------------------------------------------------
# Plain-mode (non-TUI) live progress callbacks
# ---------------------------------------------------------------------------

_plain_state: dict[str, Any] = {
    "phase": "init",
    "cmd_count": 0,
    "start_time": 0.0,
}


def _plain_phase_change(old_phase: Any, new_phase: Any) -> None:
    """Show phase transitions in non-TUI mode."""
    phase_str = new_phase.value if hasattr(new_phase, "value") else str(new_phase)
    _plain_state["phase"] = phase_str
    _plain_state["cmd_count"] = 0
    _plain_state["start_time"] = time.time()
    console.print(f"\n[bold cyan]{'='*50}[/]")
    console.print(f"[bold cyan]  Phase: {phase_str.upper()}[/]")
    console.print(f"[bold cyan]{'='*50}[/]")


def _plain_tool_call(tool_call: Any) -> None:
    """Show live command execution in non-TUI mode."""
    _plain_state["cmd_count"] += 1
    elapsed = time.time() - _plain_state.get("start_time", time.time())
    phase = _plain_state["phase"]
    count = _plain_state["cmd_count"]

    # Build compact argument preview
    args_preview = ""
    if hasattr(tool_call, "arguments"):
        for k, v in tool_call.arguments.items():
            val_str = str(v)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            args_preview += f" {k}={val_str}"
            if len(args_preview) > 80:
                args_preview = args_preview[:77] + "..."
                break

    name = tool_call.name if hasattr(tool_call, "name") else str(tool_call)
    console.print(
        f"  [yellow]>[/] [dim]{phase}[/] #{count} "
        f"[bold yellow]{name}[/][dim]{args_preview}[/] "
        f"[dim]({elapsed:.0f}s)[/]"
    )


def _plain_tool_result(result: Any) -> None:
    """Show compact command result in non-TUI mode."""
    if not hasattr(result, "success"):
        return

    status_icon = "[green]OK[/]" if result.success else "[red]FAIL[/]"
    duration = f" {result.duration:.1f}s" if hasattr(result, "duration") and result.duration else ""

    # Show first meaningful line of output, truncated
    output = getattr(result, "output", "")
    preview = ""
    if output:
        for line in output.split("\n"):
            line = line.strip()
            if line:
                preview = line[:100]
                if len(line) > 100:
                    preview += "..."
                break

    console.print(f"    [{status_icon}]{duration} {preview}")

    if not result.success and output:
        # Show up to 3 lines of error detail
        err_lines = [l.strip() for l in output.split("\n") if l.strip()][:3]
        for el in err_lines[1:]:
            console.print(f"    [dim red]{el[:120]}[/]")


def _plain_message(message: str) -> None:
    """Simple stdout callback for non-TUI mode."""
    if message:
        for line in message.split("\n")[:5]:
            if line.strip():
                console.print(f"  [dim]| {line[:150]}[/]")


def _print_plain_summary(status: dict) -> None:
    """Print a plain-text summary."""
    console.print("\n" + "=" * 50)
    console.print("[bold]Emulation Result[/]")
    console.print("=" * 50)
    console.print(f"Status:       {status.get('final_status', 'unknown')}")
    console.print(f"Architecture: {status.get('architecture', 'unknown')}")
    console.print(f"NVRAM needed: {status.get('needs_nvram', 'unknown')}")
    console.print(f"Fix attempts: {status.get('fix_attempts', 0)}")
    console.print(f"Time elapsed: {status.get('elapsed', 0):.1f}s")

    agent = status.get("agent", {})
    console.print(f"Iterations:   {agent.get('iterations', 0)}")
    console.print(f"Commands:     {agent.get('commands_run', 0)}")
    console.print("=" * 50)


if __name__ == "__main__":
    main()
