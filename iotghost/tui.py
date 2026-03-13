"""Rich TUI dashboard -- live display of the emulation pipeline.

Provides a terminal user interface showing:
- Current pipeline phase and progress
- AI agent reasoning and decisions
- Shell commands being executed and their output
- QEMU serial console output
- Network status and accessible services
- Error log and fix attempts
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from iotghost.agent import Message, ToolCall, ToolResult
from iotghost.emulator import BootEvent, EmulationStatus

logger = logging.getLogger(__name__)

# Maximum lines to keep in scrollable buffers
MAX_LOG_LINES = 200
MAX_SERIAL_LINES = 100


@dataclass
class TuiState:
    """State backing the TUI display."""
    phase: str = "init"
    phase_icon: str = "..."
    agent_lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    command_lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    serial_lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_SERIAL_LINES))
    error_lines: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    services: list[tuple[str, int]] = field(default_factory=list)
    network_info: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    boot_events: list[str] = field(default_factory=list)


# Phase display configuration
_PHASE_DISPLAY: dict[str, tuple[str, str]] = {
    "init":     ("[*]", "dim"),
    "extract":  ("[E]", "cyan"),
    "analyze":  ("[A]", "blue"),
    "prepare":  ("[P]", "yellow"),
    "emulate":  ("[Q]", "green"),
    "fix":      ("[F]", "red"),
    "verify":   ("[V]", "magenta"),
    "done":     ("[+]", "bold green"),
    "failed":   ("[!]", "bold red"),
}


class EmulationTUI:
    """Rich-based TUI for monitoring the emulation pipeline.

    Integrates with the pipeline via callbacks -- the pipeline calls
    TUI methods when events occur, and the TUI updates the display.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.console = Console()
        self.state = TuiState()
        self.verbose = verbose
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # Pipeline callbacks -- called by the pipeline/agent
    # ------------------------------------------------------------------

    def on_phase_change(self, old_phase: Any, new_phase: Any) -> None:
        """Called when the pipeline transitions between phases."""
        phase_str = new_phase.value if hasattr(new_phase, "value") else str(new_phase)
        self.state.phase = phase_str
        icon, _ = _PHASE_DISPLAY.get(phase_str, ("[?]", "white"))
        self.state.phase_icon = icon
        self.state.agent_lines.append(f"--- Phase: {phase_str.upper()} ---")
        self._refresh()

    def on_agent_message(self, message: str) -> None:
        """Called when the AI agent produces a message."""
        if not message:
            return
        # Truncate long messages for display
        for line in message.split("\n")[:10]:
            if line.strip():
                self.state.agent_lines.append(line[:200])
        self._refresh()

    def on_tool_call(self, tool_call: ToolCall) -> None:
        """Called when the agent invokes a tool."""
        cmd_display = f">> {tool_call.name}("
        args_str = ", ".join(
            f"{k}={repr(v)[:60]}" for k, v in tool_call.arguments.items()
        )
        cmd_display += args_str + ")"
        self.state.command_lines.append(cmd_display)
        self._refresh()

    def on_tool_result(self, result: ToolResult) -> None:
        """Called when a tool returns a result."""
        status = "OK" if result.success else "FAIL"
        # Show first few lines of output
        output_preview = result.output.split("\n")[0][:120]
        self.state.command_lines.append(f"   [{status}] {output_preview}")

        if not result.success:
            self.state.error_lines.append(result.output[:200])

        self._refresh()

    def on_serial_line(self, line: str) -> None:
        """Called for each line of QEMU serial output."""
        self.state.serial_lines.append(line[:200])
        self._refresh()

    def on_boot_event(self, event: BootEvent) -> None:
        """Called when a boot event is detected."""
        self.state.boot_events.append(f"{event.event_type}: {event.message[:80]}")
        self._refresh()

    def on_status_update(self, status: dict[str, Any]) -> None:
        """Called with comprehensive pipeline status."""
        self.state.stats = status
        if "network" in status:
            self.state.network_info = status["network"]
        self._refresh()

    # ------------------------------------------------------------------
    # Display rendering
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the live TUI display."""
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and print final summary."""
        if self._live:
            self._live.stop()
            self._live = None

    def _refresh(self) -> None:
        """Update the live display."""
        if self._live:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Layout:
        """Build the full TUI layout."""
        layout = Layout()

        # Top: header with phase and stats
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self._build_header())

        # Body: split into left (agent + commands) and right (serial + network)
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )

        # Left column: agent reasoning on top, commands below
        layout["left"].split_column(
            Layout(name="agent", ratio=2),
            Layout(name="commands", ratio=1),
        )

        layout["agent"].update(self._build_agent_panel())
        layout["commands"].update(self._build_commands_panel())

        # Right column: serial output on top, network/services below
        layout["right"].split_column(
            Layout(name="serial", ratio=2),
            Layout(name="status", ratio=1),
        )

        layout["serial"].update(self._build_serial_panel())
        layout["status"].update(self._build_status_panel())

        layout["footer"].update(self._build_footer())

        return layout

    def _build_header(self) -> Panel:
        """Build the header panel with phase indicator."""
        icon, style = _PHASE_DISPLAY.get(self.state.phase, ("[?]", "white"))
        elapsed = time.time() - self.state.start_time

        stats = self.state.stats
        iterations = stats.get("agent", {}).get("iterations", 0)
        commands = stats.get("agent", {}).get("commands_run", 0)

        header_text = Text()
        header_text.append(" IoTGhost ", style="bold white on blue")
        header_text.append(f"  {icon} ", style=style)
        header_text.append(f"{self.state.phase.upper()}", style=f"bold {style}")
        header_text.append(f"  |  {elapsed:.0f}s elapsed", style="dim")
        header_text.append(f"  |  {iterations} iterations", style="dim")
        header_text.append(f"  |  {commands} commands", style="dim")

        return Panel(header_text, style="blue")

    def _build_agent_panel(self) -> Panel:
        """Build the AI agent reasoning panel."""
        lines = list(self.state.agent_lines)[-20:]
        content = Text()

        for line in lines:
            if line.startswith("---"):
                content.append(line + "\n", style="bold cyan")
            elif "error" in line.lower() or "fail" in line.lower():
                content.append(line + "\n", style="red")
            elif "success" in line.lower() or "found" in line.lower():
                content.append(line + "\n", style="green")
            else:
                content.append(line + "\n")

        return Panel(
            content,
            title="[bold]AI Agent[/bold]",
            border_style="cyan",
            subtitle="reasoning + decisions",
        )

    def _build_commands_panel(self) -> Panel:
        """Build the shell commands panel."""
        lines = list(self.state.command_lines)[-15:]
        content = Text()

        for line in lines:
            if line.startswith(">>"):
                content.append(line + "\n", style="bold yellow")
            elif "[OK]" in line:
                content.append(line + "\n", style="green")
            elif "[FAIL]" in line:
                content.append(line + "\n", style="red")
            else:
                content.append(line + "\n", style="dim")

        return Panel(
            content,
            title="[bold]Shell Commands[/bold]",
            border_style="yellow",
        )

    def _build_serial_panel(self) -> Panel:
        """Build the QEMU serial output panel."""
        lines = list(self.state.serial_lines)[-20:]
        content = Text()

        for line in lines:
            if "panic" in line.lower() or "error" in line.lower():
                content.append(line + "\n", style="bold red")
            elif "login" in line.lower() or "ready" in line.lower():
                content.append(line + "\n", style="bold green")
            elif "starting" in line.lower():
                content.append(line + "\n", style="yellow")
            else:
                content.append(line + "\n", style="dim white")

        return Panel(
            content,
            title="[bold]QEMU Serial Console[/bold]",
            border_style="green",
        )

    def _build_status_panel(self) -> Panel:
        """Build the network and services status panel."""
        table = Table(show_header=True, header_style="bold", expand=True, box=None)
        table.add_column("Property", style="dim", width=14)
        table.add_column("Value")

        # Architecture
        arch = self.state.stats.get("architecture", "detecting...")
        table.add_row("Architecture", arch)

        # NVRAM
        nvram = self.state.stats.get("needs_nvram", "unknown")
        table.add_row("NVRAM", str(nvram))

        # Network
        net = self.state.network_info
        if net:
            table.add_row("Device IP", net.get("device_ip", "n/a"))
            table.add_row("Host IP", net.get("host_ip", "n/a"))
            table.add_row("NAT", "yes" if net.get("nat_enabled") else "no")
        else:
            table.add_row("Network", "not configured")

        # Boot events
        if self.state.boot_events:
            last_event = self.state.boot_events[-1]
            table.add_row("Last Event", last_event[:40])

        # Fix attempts
        fixes = self.state.stats.get("fix_attempts", 0)
        if fixes > 0:
            table.add_row("Fix Attempts", str(fixes))

        # Services
        services = self.state.services
        if services:
            svc_str = ", ".join(f"{name}:{port}" for name, port in services[:5])
            table.add_row("Services", svc_str)

        return Panel(
            table,
            title="[bold]Status[/bold]",
            border_style="magenta",
        )

    def _build_footer(self) -> Panel:
        """Build the footer with controls hint."""
        footer = Text()
        footer.append(" Ctrl+C ", style="bold white on red")
        footer.append(" Stop  ", style="dim")

        errors = self.state.stats.get("last_error")
        if errors:
            footer.append(" | ", style="dim")
            footer.append(f"Last error: {str(errors)[:80]}", style="red")

        return Panel(footer, style="dim")

    # ------------------------------------------------------------------
    # Static output (non-live) for simple mode
    # ------------------------------------------------------------------

    def print_phase(self, phase: str, message: str) -> None:
        """Print a phase update in non-live mode."""
        icon, style = _PHASE_DISPLAY.get(phase, ("[?]", "white"))
        self.console.print(f"{icon} [bold {style}]{phase.upper()}[/]: {message}")

    def print_summary(self, status: dict[str, Any]) -> None:
        """Print a final summary of the emulation result."""
        self.console.print()
        self.console.print("[bold]=" * 60)
        self.console.print("[bold blue] IoTGhost Emulation Summary[/]")
        self.console.print("=" * 60)

        final = status.get("final_status", "unknown")
        if final == "running":
            self.console.print("[bold green]Status: RUNNING -- Firmware is operational![/]")
        elif final == "partial":
            self.console.print("[bold yellow]Status: PARTIAL -- Some services running[/]")
        else:
            self.console.print(f"[bold red]Status: {final.upper()}[/]")

        self.console.print(f"Architecture: {status.get('architecture', 'unknown')}")
        self.console.print(f"NVRAM needed: {status.get('needs_nvram', 'unknown')}")
        self.console.print(f"Fix attempts: {status.get('fix_attempts', 0)}")
        self.console.print(f"Time elapsed: {status.get('elapsed', 0):.1f}s")

        agent = status.get("agent", {})
        self.console.print(f"Agent iterations: {agent.get('iterations', 0)}")
        self.console.print(f"Commands executed: {agent.get('commands_run', 0)}")

        if status.get("last_error"):
            self.console.print(f"\n[red]Last error: {status['last_error']}[/]")

        network = status.get("network", {})
        if network.get("device_ip"):
            self.console.print(f"\n[bold]Network:[/]")
            self.console.print(f"  Device: {network['device_ip']}")
            self.console.print(f"  Host:   {network.get('host_ip', 'n/a')}")

        self.console.print("=" * 60)
