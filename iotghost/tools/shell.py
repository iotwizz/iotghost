"""Shell tool definitions for the IoTGhost AI agent.

Each tool is a callable that the LLM agent can invoke during firmware
emulation. Tools wrap subprocess calls with timeout handling, output
capture, and error normalization.

Every tool function has a `_schema` attribute containing its OpenAI-compatible
function schema, which the agent module uses to build tool descriptions for
the LLM.

Self-Healing Layer 1 -- Auto-Fix Interceptor:
    When execute_command() detects a known recoverable error pattern in stderr
    (e.g. 'File exists' from mkdir, 'No such file or directory' from cp/mv),
    it automatically rewrites the command and retries ONCE before returning
    the result to the LLM.  This avoids wasting an LLM iteration on trivially
    fixable shell errors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from iotghost.agent import ToolResult, register_tool

logger = logging.getLogger(__name__)

# Maximum output size to prevent context window overflow
MAX_OUTPUT_CHARS = 8000
TRUNCATION_MSG = "\n... [output truncated to {limit} chars, {total} total]"

# ---------------------------------------------------------------------------
# Auto-Fix Interceptor -- Self-Healing Layer 1
# ---------------------------------------------------------------------------

@dataclass
class AutoFixResult:
    """Tracks what the auto-fixer did to recover a command."""
    original_cmd: str
    fixed_cmd: str
    error_pattern: str          # which pattern matched
    strategy: str               # human-readable fix description
    success: bool = False       # did the retry succeed?


# Global state for auto-fix tracking (reset per phase by pipeline)
_auto_fix_history: list[AutoFixResult] = []
_auto_fix_count: int = 0
_max_auto_fixes: int = 10      # cap per phase, configurable via CLI
_self_heal_enabled: bool = True
_on_auto_fix: Callable[[AutoFixResult], None] | None = None


def configure_auto_fix(
    enabled: bool = True,
    max_fixes: int = 10,
    on_auto_fix: Callable[[AutoFixResult], None] | None = None,
) -> None:
    """Configure the auto-fix interceptor (called by pipeline at setup)."""
    global _self_heal_enabled, _max_auto_fixes, _on_auto_fix
    _self_heal_enabled = enabled
    _max_auto_fixes = max_fixes
    _on_auto_fix = on_auto_fix


def reset_auto_fix_state() -> None:
    """Reset auto-fix counters (called at the start of each phase)."""
    global _auto_fix_count
    _auto_fix_count = 0
    _auto_fix_history.clear()


def get_auto_fix_history() -> list[AutoFixResult]:
    """Return auto-fix history for the current phase."""
    return list(_auto_fix_history)


# --- Error pattern matchers + fix generators ---

_FIX_PATTERNS: list[tuple[str, re.Pattern, Callable[[str, re.Match], str | None]]] = []


def _register_fix(name: str, pattern: str):
    """Decorator to register an auto-fix pattern."""
    compiled = re.compile(pattern, re.IGNORECASE)
    def decorator(func: Callable[[str, re.Match], str | None]):
        _FIX_PATTERNS.append((name, compiled, func))
        return func
    return decorator


@_register_fix("mkdir_exists", r"mkdir:\s.*(?:File exists|cannot create directory).*['\"]?([^'\"\n]+)['\"]?")
def _fix_mkdir_exists(cmd: str, match: re.Match) -> str | None:
    """mkdir fails with 'File exists' -> retry with mkdir -p."""
    if "mkdir" in cmd and "-p" not in cmd:
        return cmd.replace("mkdir ", "mkdir -p ", 1)
    return None


@_register_fix("missing_parent_dir", r"(?:cp|mv|install):\s.*(?:No such file or directory|cannot (?:stat|create|move)).*['\"]?([^'\"\n]+)['\"]?")
def _fix_missing_parent(cmd: str, match: re.Match) -> str | None:
    """cp/mv/install fails with 'No such file or directory' -> create parent dirs first."""
    # Extract the destination path (last argument)
    parts = cmd.strip().split()
    if len(parts) < 3:
        return None
    dest = parts[-1]
    # If dest looks like a path (has a /), create its parent
    if "/" in dest:
        parent = str(Path(dest).parent)
        return f"mkdir -p {parent} && {cmd}"
    return None


@_register_fix("permission_denied", r"(?:Permission denied|EACCES|Operation not permitted).*['\"]?([^'\"\n]*)['\"]?")
def _fix_permission_denied(cmd: str, match: re.Match) -> str | None:
    """Permission denied -> try chmod +x or sudo depending on the command."""
    # For file copy/move operations, try making the target writable
    parts = cmd.strip().split()
    if not parts:
        return None
    verb = parts[0]
    if verb in ("cp", "mv", "install", "ln") and len(parts) >= 3:
        dest = parts[-1]
        if Path(dest).exists():
            return f"chmod -R u+w {dest} 2>/dev/null; {cmd}"
        parent = str(Path(dest).parent)
        return f"chmod -R u+w {parent} 2>/dev/null; {cmd}"
    # For execution failures, try chmod +x
    if verb in ("./", "sh", "bash") or verb.startswith("./") or verb.startswith("/"):
        target = parts[0]
        return f"chmod +x {target} && {cmd}"
    return None


@_register_fix("dir_not_empty", r"(?:mv|rm):\s.*(?:Directory not empty|cannot (?:move|remove)).*['\"]?([^'\"\n]+)['\"]?")
def _fix_dir_not_empty(cmd: str, match: re.Match) -> str | None:
    """mv fails with 'Directory not empty' -> use cp -af then rm."""
    parts = cmd.strip().split()
    if len(parts) >= 3 and parts[0] == "mv":
        # Extract source and dest, handling flags
        args = [p for p in parts[1:] if not p.startswith("-")]
        if len(args) >= 2:
            src, dst = args[-2], args[-1]
            return f"cp -af {src} {dst} && rm -rf {src}"
    return None


@_register_fix("device_busy", r"(?:Device or resource busy|EBUSY)")
def _fix_device_busy(cmd: str, match: re.Match) -> str | None:
    """Device busy on umount/rm -> lazy umount or wait-retry."""
    parts = cmd.strip().split()
    if parts and parts[0] == "umount":
        if "-l" not in parts:
            return cmd.replace("umount ", "umount -l ", 1)
    return None


@_register_fix("symlink_exists", r"ln:\s.*(?:File exists|failed to create symbolic link)")
def _fix_symlink_exists(cmd: str, match: re.Match) -> str | None:
    """ln fails with 'File exists' -> retry with ln -sf."""
    if "ln" in cmd and "-sf" not in cmd and "-f" not in cmd:
        return cmd.replace("ln -s ", "ln -sf ", 1).replace("ln ", "ln -sf ", 1)
    return None


def _try_auto_fix(
    cmd: str,
    stderr: str,
    exit_code: int,
    timeout: int,
    workdir: str | None,
) -> ToolResult | None:
    """Attempt to auto-fix a failed command. Returns new ToolResult or None.

    Called by execute_command() when a command fails. Checks stderr against
    known error patterns, generates a fixed command, and runs it once.
    Returns None if no fix was applicable or auto-fix is disabled/exhausted.
    """
    global _auto_fix_count

    if not _self_heal_enabled:
        return None
    if _auto_fix_count >= _max_auto_fixes:
        logger.debug("Auto-fix budget exhausted (%d/%d)", _auto_fix_count, _max_auto_fixes)
        return None

    for pattern_name, pattern_re, fix_func in _FIX_PATTERNS:
        match = pattern_re.search(stderr)
        if not match:
            continue

        fixed_cmd = fix_func(cmd, match)
        if not fixed_cmd or fixed_cmd == cmd:
            continue

        # Execute the fixed command
        logger.info("AUTO-FIX [%s]: %s -> %s", pattern_name, cmd[:80], fixed_cmd[:80])
        _auto_fix_count += 1

        fix_record = AutoFixResult(
            original_cmd=cmd,
            fixed_cmd=fixed_cmd,
            error_pattern=pattern_name,
            strategy=f"{pattern_name}: rewritten command",
        )

        try:
            result = subprocess.run(
                fixed_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env={**os.environ, "TERM": "dumb", "LANG": "C"},
            )

            fix_record.success = result.returncode == 0
            _auto_fix_history.append(fix_record)

            # Notify TUI
            if _on_auto_fix:
                _on_auto_fix(fix_record)

            stdout = result.stdout or ""
            stderr_new = result.stderr or ""

            output_parts = []
            output_parts.append(
                f"[AUTO-FIXED] Original command failed ({pattern_name}). "
                f"Retried with: {fixed_cmd}"
            )
            if stdout:
                output_parts.append(f"[stdout]\n{_truncate(stdout)}")
            if stderr_new:
                output_parts.append(f"[stderr]\n{_truncate(stderr_new)}")
            output_parts.append(f"[exit_code: {result.returncode}]")

            return ToolResult(
                tool_call_id="",
                output="\n".join(output_parts),
                success=result.returncode == 0,
                duration=0.0,
            )

        except subprocess.TimeoutExpired:
            fix_record.success = False
            _auto_fix_history.append(fix_record)
            logger.warning("Auto-fix command also timed out: %s", fixed_cmd[:80])
            return None
        except Exception as exc:
            fix_record.success = False
            _auto_fix_history.append(fix_record)
            logger.warning("Auto-fix command failed: %s -- %s", fixed_cmd[:80], exc)
            return None

    return None  # No pattern matched


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate output to prevent LLM context overflow."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + TRUNCATION_MSG.format(limit=limit, total=len(text))
        + text[-half:]
    )


# ---------------------------------------------------------------------------
# execute_command -- the primary tool
# ---------------------------------------------------------------------------

def execute_command(cmd: str, timeout: int = 120, workdir: str | None = None) -> ToolResult:
    """Execute a shell command and return stdout + stderr.

    This is the agent's primary way of interacting with the system.
    All binwalk, QEMU, file manipulation, and network commands go through here.
    """
    logger.info("EXEC: %s (timeout=%ds)", cmd, timeout)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env={**os.environ, "TERM": "dumb", "LANG": "C"},
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.returncode

        output_parts = []
        if stdout:
            output_parts.append(f"[stdout]\n{_truncate(stdout)}")
        if stderr:
            output_parts.append(f"[stderr]\n{_truncate(stderr)}")
        output_parts.append(f"[exit_code: {exit_code}]")

        output = "\n".join(output_parts)
        success = exit_code == 0

        if not success:
            logger.warning("Command failed (exit %d): %s", exit_code, cmd[:100])
            # --- Self-Healing Layer 1: try auto-fix before returning ---
            auto_result = _try_auto_fix(cmd, stderr, exit_code, timeout, workdir)
            if auto_result is not None:
                return auto_result

        return ToolResult(tool_call_id="", output=output, success=success)

    except subprocess.TimeoutExpired:
        logger.error("Command timed out after %ds: %s", timeout, cmd[:100])
        return ToolResult(
            tool_call_id="",
            output=f"[TIMEOUT] Command exceeded {timeout}s limit. "
                   "Consider: increase timeout, run in background, or use a different approach.",
            success=False,
        )
    except Exception as exc:
        logger.error("Command error: %s -- %s", cmd[:100], exc)
        return ToolResult(
            tool_call_id="",
            output=f"[ERROR] Failed to execute command: {type(exc).__name__}: {exc}",
            success=False,
        )


execute_command._schema = {
    "description": (
        "Execute a shell command on the host system. Use for running binwalk, "
        "QEMU, file operations, network configuration, and any other system commands. "
        "Returns stdout, stderr, and exit code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Shell command to execute (passed to /bin/sh -c)",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default 120). Use higher for QEMU boot.",
                "default": 120,
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the command (optional).",
            },
        },
        "required": ["cmd"],
    },
}


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def read_file(path: str, max_bytes: int = 32768) -> ToolResult:
    """Read a file's contents (text or hex dump for binary)."""
    logger.info("READ: %s", path)

    p = Path(path)
    if not p.exists():
        return ToolResult(
            tool_call_id="", output=f"Error: File not found: {path}", success=False
        )

    try:
        size = p.stat().st_size
        if size > max_bytes:
            # Read partial
            with open(p, "rb") as f:
                raw = f.read(max_bytes)
            try:
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                content = raw.hex()
            return ToolResult(
                tool_call_id="",
                output=f"[partial read: {max_bytes}/{size} bytes]\n{_truncate(content)}",
                success=True,
            )

        # Try text first
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            return ToolResult(tool_call_id="", output=content, success=True)
        except Exception:
            raw = p.read_bytes()
            return ToolResult(
                tool_call_id="",
                output=f"[binary file, {size} bytes, hex dump]\n{raw[:1024].hex()}",
                success=True,
            )

    except PermissionError:
        return ToolResult(
            tool_call_id="", output=f"Error: Permission denied: {path}", success=False
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            output=f"Error reading {path}: {type(exc).__name__}: {exc}",
            success=False,
        )


read_file._schema = {
    "description": (
        "Read the contents of a file. Returns text content for text files, "
        "hex dump for binary files. Large files are truncated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file to read.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum bytes to read (default 32KB).",
                "default": 32768,
            },
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def write_file(path: str, content: str, mode: str = "w") -> ToolResult:
    """Write content to a file. Creates parent directories if needed."""
    logger.info("WRITE: %s (%d chars, mode=%s)", path, len(content), mode)

    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if mode == "a":
            with open(p, "a") as f:
                f.write(content)
        else:
            p.write_text(content)

        size = p.stat().st_size
        return ToolResult(
            tool_call_id="",
            output=f"Written {size} bytes to {path}",
            success=True,
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            output=f"Error writing {path}: {type(exc).__name__}: {exc}",
            success=False,
        )


write_file._schema = {
    "description": (
        "Write content to a file. Creates parent directories automatically. "
        "Use for creating config files, init scripts, NVRAM defaults, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to write to.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
            "mode": {
                "type": "string",
                "description": "'w' to overwrite (default), 'a' to append.",
                "default": "w",
                "enum": ["w", "a"],
            },
        },
        "required": ["path", "content"],
    },
}


# ---------------------------------------------------------------------------
# patch_binary
# ---------------------------------------------------------------------------

def patch_binary(path: str, offset: str, data: str) -> ToolResult:
    """Write hex data at a specific offset in a binary file.

    Used for binary patching -- disabling checks, fixing magic bytes, etc.
    """
    logger.info("PATCH: %s at offset %s with %d hex chars", path, offset, len(data))

    try:
        p = Path(path)
        if not p.exists():
            return ToolResult(
                tool_call_id="", output=f"Error: File not found: {path}", success=False
            )

        offset_int = int(offset, 16) if offset.startswith("0x") else int(offset)
        patch_bytes = bytes.fromhex(data)

        with open(p, "r+b") as f:
            f.seek(offset_int)
            original = f.read(len(patch_bytes))
            f.seek(offset_int)
            f.write(patch_bytes)

        return ToolResult(
            tool_call_id="",
            output=(
                f"Patched {len(patch_bytes)} bytes at offset {offset} in {path}\n"
                f"Original: {original.hex()}\n"
                f"New:      {data}"
            ),
            success=True,
        )
    except ValueError as exc:
        return ToolResult(
            tool_call_id="",
            output=f"Error: Invalid offset or hex data: {exc}",
            success=False,
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            output=f"Error patching {path}: {type(exc).__name__}: {exc}",
            success=False,
        )


patch_binary._schema = {
    "description": (
        "Write hex data at a specific offset in a binary file. "
        "Use for patching firmware binaries -- disabling license checks, "
        "fixing magic bytes, NOP-ing out hardware-specific code, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the binary file.",
            },
            "offset": {
                "type": "string",
                "description": "Byte offset (decimal or 0x hex) where to write.",
            },
            "data": {
                "type": "string",
                "description": "Hex string of bytes to write (e.g., '90909090' for NOPs).",
            },
        },
        "required": ["path", "offset", "data"],
    },
}


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def list_directory(path: str, recursive: bool = False, max_entries: int = 200) -> ToolResult:
    """List contents of a directory with file types and sizes."""
    logger.info("LIST: %s (recursive=%s)", path, recursive)

    p = Path(path)
    if not p.exists():
        return ToolResult(
            tool_call_id="", output=f"Error: Directory not found: {path}", success=False
        )
    if not p.is_dir():
        return ToolResult(
            tool_call_id="", output=f"Error: Not a directory: {path}", success=False
        )

    try:
        entries = []
        iterator = p.rglob("*") if recursive else p.iterdir()

        for i, entry in enumerate(sorted(iterator)):
            if i >= max_entries:
                entries.append(f"... and more ({max_entries} limit reached)")
                break

            rel = entry.relative_to(p)
            if entry.is_dir():
                entries.append(f"  {rel}/")
            elif entry.is_symlink():
                target = entry.resolve() if entry.exists() else "broken"
                entries.append(f"  {rel} -> {target}")
            else:
                try:
                    size = entry.stat().st_size
                    if size > 1_000_000:
                        size_str = f"{size / 1_000_000:.1f}M"
                    elif size > 1000:
                        size_str = f"{size / 1000:.1f}K"
                    else:
                        size_str = f"{size}B"
                    entries.append(f"  {rel} ({size_str})")
                except OSError:
                    entries.append(f"  {rel} (?)")

        header = f"Directory: {path} ({len(entries)} entries)\n"
        return ToolResult(
            tool_call_id="",
            output=header + "\n".join(entries),
            success=True,
        )
    except PermissionError:
        return ToolResult(
            tool_call_id="",
            output=f"Error: Permission denied listing {path}",
            success=False,
        )


list_directory._schema = {
    "description": (
        "List contents of a directory showing file names, types, and sizes. "
        "Use to explore extracted firmware filesystems."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list.",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, list all files recursively (default false).",
                "default": False,
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum entries to return (default 200).",
                "default": 200,
            },
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# check_network
# ---------------------------------------------------------------------------

def check_network(host: str, port: int, timeout: float = 5.0) -> ToolResult:
    """Test TCP connectivity to a host:port.

    Used to verify if emulated firmware services are reachable.
    """
    logger.info("NETCHECK: %s:%d", host, port)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        start = time.time()
        result_code = sock.connect_ex((host, port))
        elapsed = time.time() - start
        sock.close()

        if result_code == 0:
            return ToolResult(
                tool_call_id="",
                output=f"OPEN: {host}:{port} (connected in {elapsed:.2f}s)",
                success=True,
            )
        else:
            return ToolResult(
                tool_call_id="",
                output=f"CLOSED: {host}:{port} (error code {result_code}, {elapsed:.2f}s)",
                success=False,
            )
    except socket.timeout:
        return ToolResult(
            tool_call_id="",
            output=f"TIMEOUT: {host}:{port} (no response in {timeout}s)",
            success=False,
        )
    except socket.gaierror:
        return ToolResult(
            tool_call_id="",
            output=f"DNS_ERROR: Cannot resolve host '{host}'",
            success=False,
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            output=f"ERROR: {host}:{port} -- {type(exc).__name__}: {exc}",
            success=False,
        )


check_network._schema = {
    "description": (
        "Test TCP connectivity to a host and port. Use to verify if the "
        "emulated firmware's web interface, SSH, telnet, or other services "
        "are reachable after boot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "host": {
                "type": "string",
                "description": "Target IP address or hostname.",
            },
            "port": {
                "type": "integer",
                "description": "TCP port number to test.",
            },
            "timeout": {
                "type": "number",
                "description": "Connection timeout in seconds (default 5).",
                "default": 5.0,
            },
        },
        "required": ["host", "port"],
    },
}


# ---------------------------------------------------------------------------
# Registration -- called when this module is imported
# ---------------------------------------------------------------------------

def register_all_tools() -> None:
    """Register all shell tools with the agent's tool registry."""
    register_tool("execute_command", execute_command)
    register_tool("read_file", read_file)
    register_tool("write_file", write_file)
    register_tool("patch_binary", patch_binary)
    register_tool("list_directory", list_directory)
    register_tool("check_network", check_network)
    logger.info("Registered %d shell tools", 6)
