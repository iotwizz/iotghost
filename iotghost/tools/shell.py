"""Shell tool definitions for the IoTGhost AI agent.

Each tool is a callable that the LLM agent can invoke during firmware
emulation. Tools wrap subprocess calls with timeout handling, output
capture, and error normalization.

Every tool function has a `_schema` attribute containing its OpenAI-compatible
function schema, which the agent module uses to build tool descriptions for
the LLM.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

from iotghost.agent import ToolResult, register_tool

logger = logging.getLogger(__name__)

# Maximum output size to prevent context window overflow
MAX_OUTPUT_CHARS = 8000
TRUNCATION_MSG = "\n... [output truncated to {limit} chars, {total} total]"


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
