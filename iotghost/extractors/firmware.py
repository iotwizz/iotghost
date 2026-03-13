"""Firmware extraction module -- unpacks firmware images to get root filesystems.

Wraps binwalk, sasquatch, jefferson, ubi_reader, and other extraction tools
with automatic format detection and fallback chains. The AI agent calls
execute_command to run these tools, but this module provides the structured
logic for the pipeline's extract phase.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class FirmwareFormat(str, Enum):
    """Detected firmware container/filesystem formats."""
    SQUASHFS = "squashfs"
    CRAMFS = "cramfs"
    JFFS2 = "jffs2"
    UBIFS = "ubifs"
    YAFFS2 = "yaffs2"
    CPIO = "cpio"
    EXT2 = "ext2"
    EXT4 = "ext4"
    TAR = "tar"
    ZIP = "zip"
    UIMAGE = "uimage"
    ELF = "elf"
    RAW = "raw"
    ENCRYPTED = "encrypted"
    UNKNOWN = "unknown"


class Architecture(str, Enum):
    """CPU architecture of the firmware."""
    MIPSEL = "mipsel"
    MIPSBE = "mipsbe"
    ARMEL = "armel"
    ARM64 = "arm64"
    X86 = "x86"
    X86_64 = "x86_64"
    PPC = "ppc"
    UNKNOWN = "unknown"


@dataclass
class BinwalkEntry:
    """Single entry from binwalk scan output."""
    offset: int
    offset_hex: str
    description: str
    format_type: FirmwareFormat = FirmwareFormat.UNKNOWN


@dataclass
class ExtractionResult:
    """Result of firmware extraction attempt."""
    success: bool
    rootfs_path: str | None = None
    architecture: Architecture = Architecture.UNKNOWN
    endianness: str = "unknown"
    formats_found: list[FirmwareFormat] = field(default_factory=list)
    kernel_offset: int | None = None
    kernel_version: str | None = None
    binwalk_entries: list[BinwalkEntry] = field(default_factory=list)
    extraction_method: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Binwalk output parsing
# ---------------------------------------------------------------------------

# Maps binwalk description keywords to firmware formats
_FORMAT_KEYWORDS: dict[str, FirmwareFormat] = {
    "squashfs": FirmwareFormat.SQUASHFS,
    "cramfs": FirmwareFormat.CRAMFS,
    "jffs2": FirmwareFormat.JFFS2,
    "ubi image": FirmwareFormat.UBIFS,
    "ubifs": FirmwareFormat.UBIFS,
    "yaffs": FirmwareFormat.YAFFS2,
    "cpio": FirmwareFormat.CPIO,
    "ext2": FirmwareFormat.EXT2,
    "ext4": FirmwareFormat.EXT4,
    "tar archive": FirmwareFormat.TAR,
    "zip archive": FirmwareFormat.ZIP,
    "uimage": FirmwareFormat.UIMAGE,
    "elf,": FirmwareFormat.ELF,
}


def parse_binwalk_output(output: str) -> list[BinwalkEntry]:
    """Parse binwalk scan output into structured entries.

    Binwalk output format:
    DECIMAL       HEXADECIMAL     DESCRIPTION
    -------------------------------------------------------
    0             0x0             uImage header, ...
    262144        0x40000         Squashfs filesystem, ...
    """
    entries = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("DECIMAL") or line.startswith("---"):
            continue

        # Parse: decimal_offset  hex_offset  description
        match = re.match(r"(\d+)\s+(0x[0-9A-Fa-f]+)\s+(.+)", line)
        if not match:
            continue

        offset = int(match.group(1))
        offset_hex = match.group(2)
        description = match.group(3)

        # Detect format type
        fmt = FirmwareFormat.UNKNOWN
        desc_lower = description.lower()
        for keyword, format_type in _FORMAT_KEYWORDS.items():
            if keyword in desc_lower:
                fmt = format_type
                break

        entries.append(BinwalkEntry(
            offset=offset,
            offset_hex=offset_hex,
            description=description,
            format_type=fmt,
        ))

    return entries


# ---------------------------------------------------------------------------
# Architecture detection from ELF / file command output
# ---------------------------------------------------------------------------

# Maps `file` command output keywords to architectures
_ARCH_PATTERNS: list[tuple[str, Architecture, str]] = [
    (r"MIPS.*MSB", Architecture.MIPSBE, "big"),
    (r"MIPS.*LSB", Architecture.MIPSEL, "little"),
    (r"MIPS", Architecture.MIPSEL, "little"),  # default MIPS to little
    (r"ARM aarch64", Architecture.ARM64, "little"),
    (r"ARM.*EABI", Architecture.ARMEL, "little"),
    (r"ARM", Architecture.ARMEL, "little"),
    (r"x86-64|x86_64|AMD64", Architecture.X86_64, "little"),
    (r"Intel 80386|i386|x86", Architecture.X86, "little"),
    (r"PowerPC|PPC", Architecture.PPC, "big"),
]


def detect_architecture(file_output: str) -> tuple[Architecture, str]:
    """Detect CPU architecture from `file` command output.

    Returns (Architecture, endianness) tuple.
    """
    for pattern, arch, endian in _ARCH_PATTERNS:
        if re.search(pattern, file_output, re.IGNORECASE):
            # Refine endianness from explicit markers
            if "MSB" in file_output:
                endian = "big"
            elif "LSB" in file_output:
                endian = "little"
            return arch, endian

    return Architecture.UNKNOWN, "unknown"


def detect_kernel_version(strings_output: str) -> str | None:
    """Extract Linux kernel version from strings output.

    Looks for patterns like 'Linux version 2.6.31' or '3.10.14'.
    """
    match = re.search(r"Linux version (\d+\.\d+\.\d+)", strings_output)
    if match:
        return match.group(1)

    # Fallback: look for kernel version string
    match = re.search(r"(\d+\.\d+\.\d+)(?:-\w+)?\s+\(", strings_output)
    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def get_extraction_commands(
    firmware_path: str,
    output_dir: str,
    formats: list[FirmwareFormat],
) -> list[tuple[str, str]]:
    """Generate extraction commands based on detected formats.

    Returns list of (command, description) tuples in priority order.
    The AI agent should try them in order until one succeeds.
    """
    commands = []
    fw = firmware_path
    out = output_dir

    # Always try binwalk first -- handles most formats
    commands.append((
        f"binwalk -e -C {out} {fw}",
        "binwalk standard extraction",
    ))

    for fmt in formats:
        if fmt == FirmwareFormat.SQUASHFS:
            # Try sasquatch for vendor-modified SquashFS
            commands.append((
                f"sasquatch -d {out}/squashfs-root {fw}",
                "sasquatch (handles vendor-modified SquashFS)",
            ))
            # unsquashfs as fallback
            commands.append((
                f"unsquashfs -d {out}/squashfs-root {fw}",
                "unsquashfs standard extraction",
            ))

        elif fmt == FirmwareFormat.JFFS2:
            commands.append((
                f"jefferson -d {out}/jffs2-root {fw}",
                "jefferson JFFS2 extraction",
            ))

        elif fmt == FirmwareFormat.UBIFS:
            commands.append((
                f"ubireader_extract_files -o {out}/ubi-root {fw}",
                "ubi_reader UBI extraction",
            ))

        elif fmt == FirmwareFormat.YAFFS2:
            commands.append((
                f"unyaffs {fw} {out}/yaffs2-root",
                "unyaffs YAFFS2 extraction",
            ))

        elif fmt == FirmwareFormat.CPIO:
            cpio_dir = f"{out}/cpio-root"
            commands.append((
                f"mkdir -p {cpio_dir} && cd {cpio_dir} && cpio -idm < {fw}",
                "cpio extraction",
            ))

        elif fmt == FirmwareFormat.EXT2 or fmt == FirmwareFormat.EXT4:
            commands.append((
                f"mkdir -p {out}/ext-root && mount -o loop,ro {fw} {out}/ext-root",
                "mount ext filesystem (requires root)",
            ))

    # Last resort: recursive binwalk with matryoshka
    commands.append((
        f"binwalk -Me -C {out} {fw}",
        "binwalk recursive matryoshka extraction",
    ))

    return commands


def find_rootfs(extraction_dir: str) -> str | None:
    """Search extracted files for the root filesystem directory.

    Handles common extraction layouts including:
    - Direct rootfs in extraction dir
    - binwalk _*.extracted/ offset directories (e.g. _40000.extracted/)
    - TRX/ASUS nested layouts with multiple extraction layers
    - squashfs-root/ and similar named directories
    - Recursive matryoshka extraction trees

    Returns the path to the best rootfs candidate, or None.
    """
    extraction_path = Path(extraction_dir)
    if not extraction_path.exists():
        return None

    rootfs_markers = {"bin", "etc", "lib", "sbin"}
    strong_markers = {"usr", "var", "tmp", "dev", "proc"}

    # Collect all candidates with their scores for ranked selection
    candidates: list[tuple[int, int, str]] = []  # (strong_count, -depth, path)

    # --- Strategy 1: Check well-known directory names first (fast path) ---
    well_known = [
        "squashfs-root", "rootfs", "rootfs-root",
        "cpio-root", "jffs2-root", "ubi-root",
        "yaffs2-root", "ext-root",
    ]
    for name in well_known:
        for match in extraction_path.rglob(name):
            if not match.is_dir():
                continue
            children = {c.name for c in match.iterdir() if c.is_dir()}
            if rootfs_markers.issubset(children):
                strong_count = len(strong_markers & children)
                depth = len(match.relative_to(extraction_path).parts)
                candidates.append((strong_count, -depth, str(match)))
                logger.debug("rootfs candidate (well-known): %s score=%d", match, strong_count)

    # --- Strategy 2: Scan binwalk offset directories ---
    # binwalk creates dirs like _firmware.bin.extracted/ containing
    # offset-named subdirs (e.g., 40000/) or further _*.extracted/ trees.
    # Newer binwalk versions also create bin-N-extracted/ directories.
    for extracted_dir in extraction_path.rglob("*extracted*"):
        if not extracted_dir.is_dir():
            continue
        # Check the extracted dir itself
        children = {c.name for c in extracted_dir.iterdir() if c.is_dir()}
        if rootfs_markers.issubset(children):
            strong_count = len(strong_markers & children)
            depth = len(extracted_dir.relative_to(extraction_path).parts)
            candidates.append((strong_count, -depth, str(extracted_dir)))

        # Check immediate subdirs (offset dirs like 40000/, squashfs-root/, etc.)
        for sub in extracted_dir.iterdir():
            if not sub.is_dir():
                continue
            sub_children = {c.name for c in sub.iterdir() if c.is_dir()}
            if rootfs_markers.issubset(sub_children):
                strong_count = len(strong_markers & sub_children)
                depth = len(sub.relative_to(extraction_path).parts)
                candidates.append((strong_count, -depth, str(sub)))
                logger.debug("rootfs candidate (offset-dir): %s score=%d", sub, strong_count)

    # --- Strategy 2b: Scan binwalk v3 bin-N-extracted directories ---
    # binwalk v3+ uses naming like bin-0-extracted/, bin-1-extracted/
    for extracted_dir in extraction_path.rglob("bin-*-extracted"):
        if not extracted_dir.is_dir():
            continue
        # Check the extracted dir itself
        children = {c.name for c in extracted_dir.iterdir() if c.is_dir()}
        if rootfs_markers.issubset(children):
            strong_count = len(strong_markers & children)
            depth = len(extracted_dir.relative_to(extraction_path).parts)
            candidates.append((strong_count, -depth, str(extracted_dir)))

        # Check immediate subdirs (offset dirs like 40000/, squashfs-root/, etc.)
        for sub in extracted_dir.iterdir():
            if not sub.is_dir():
                continue
            sub_children = {c.name for c in sub.iterdir() if c.is_dir()}
            if rootfs_markers.issubset(sub_children):
                strong_count = len(strong_markers & sub_children)
                depth = len(sub.relative_to(extraction_path).parts)
                candidates.append((strong_count, -depth, str(sub)))
                logger.debug("rootfs candidate (offset-dir): %s score=%d", sub, strong_count)

    # --- Strategy 3: Breadth-first scan up to 6 levels deep ---
    # Walk level by level so shallower matches rank higher
    for depth_limit in range(7):
        for candidate in extraction_path.rglob("*"):
            if not candidate.is_dir():
                continue
            rel_depth = len(candidate.relative_to(extraction_path).parts)
            if rel_depth != depth_limit:
                continue

            try:
                children = {c.name for c in candidate.iterdir() if c.is_dir()}
            except PermissionError:
                continue

            if rootfs_markers.issubset(children):
                strong_count = len(strong_markers & children)
                candidates.append((strong_count, -rel_depth, str(candidate)))

    # --- Select best candidate ---
    if candidates:
        # Sort by: most strong markers first, then shallowest depth
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_score, best_neg_depth, best_path = candidates[0]
        # Any directory with all 4 core markers (bin, etc, lib, sbin) is a
        # valid rootfs.  Strong markers are a ranking tiebreaker, not a gate.
        logger.info(
            "Found rootfs at: %s (score=%d, depth=%d, candidates=%d)",
            best_path, best_score, -best_neg_depth, len(candidates),
        )
        return best_path

    # --- Fallback: relaxed matching (3 of 4 core markers) ---
    relaxed: list[tuple[int, int, str]] = []
    for candidate in extraction_path.rglob("*"):
        if not candidate.is_dir():
            continue
        try:
            children = {c.name for c in candidate.iterdir() if c.is_dir()}
        except PermissionError:
            continue
        core_count = len(rootfs_markers & children)
        if core_count >= 3:
            strong_count = len(strong_markers & children)
            rel_depth = len(candidate.relative_to(extraction_path).parts)
            relaxed.append((core_count + strong_count, -rel_depth, str(candidate)))

    if relaxed:
        relaxed.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, best_path = relaxed[0]
        logger.info("Found rootfs (relaxed) at: %s", best_path)
        return best_path

    logger.warning("No rootfs found in %s", extraction_dir)
    return None


def check_encryption(binwalk_entropy_output: str) -> bool:
    """Check if firmware appears encrypted based on entropy analysis.

    High entropy (>0.95) across the entire image suggests encryption.
    """
    # Look for entropy values in binwalk -E output
    high_entropy_blocks = 0
    total_blocks = 0

    for line in binwalk_entropy_output.split("\n"):
        match = re.search(r"(\d+\.\d+)", line)
        if match:
            entropy = float(match.group(1))
            total_blocks += 1
            if entropy > 0.95:
                high_entropy_blocks += 1

    if total_blocks == 0:
        return False

    # If >80% of blocks have high entropy, likely encrypted
    return (high_entropy_blocks / total_blocks) > 0.8
