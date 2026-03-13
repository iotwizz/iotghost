"""NVRAM emulation -- intercept libnvram calls for firmware that needs it.

~53% of IoT firmware images depend on NVRAM for configuration. Without
emulation, web servers and most services crash immediately. This module:

1. Generates an NVRAM defaults INI file from vendor templates + extracted configs
2. Deploys a libnvram.so intercept library into the rootfs via LD_PRELOAD
3. Scans firmware binaries to detect NVRAM dependency and required keys
4. Provides utilities to manage NVRAM key-value stores
"""

from __future__ import annotations

import configparser
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from iotghost.prompts import VENDOR_NVRAM_DEFAULTS

logger = logging.getLogger(__name__)

# Path where libnvram.so intercept library is stored in the package
_LIBNVRAM_SRC = Path(__file__).parent / "lib" / "libnvram.so"

# Default NVRAM paths inside the emulated rootfs
NVRAM_INI_PATH = "/etc/nvram.ini"
NVRAM_LIB_PATH = "/usr/lib/libnvram.so"
NVRAM_OVERRIDE_ENV = "LD_PRELOAD=/usr/lib/libnvram.so"


@dataclass
class NvramConfig:
    """NVRAM configuration for a firmware image."""
    needed: bool = False
    vendor_family: str = "broadcom_generic"
    defaults: dict[str, str] = field(default_factory=dict)
    extracted_keys: dict[str, str] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    binary_refs: list[str] = field(default_factory=list)  # binaries that reference nvram


@dataclass
class NvramScanResult:
    """Result of scanning a rootfs for NVRAM dependencies."""
    needs_nvram: bool = False
    nvram_binaries: list[str] = field(default_factory=list)
    nvram_functions: list[str] = field(default_factory=list)  # nvram_get, nvram_set, etc.
    extracted_defaults: dict[str, str] = field(default_factory=dict)
    default_files_found: list[str] = field(default_factory=list)
    recommended_vendor: str = "broadcom_generic"


# ---------------------------------------------------------------------------
# NVRAM dependency detection
# ---------------------------------------------------------------------------

# Functions that indicate NVRAM dependency
_NVRAM_FUNCTIONS = [
    "nvram_get",
    "nvram_set",
    "nvram_unset",
    "nvram_commit",
    "nvram_safe_get",
    "nvram_safe_set",
    "nvram_bufget",
    "nvram_bufset",
    "acosNvramConfig_get",
    "acosNvramConfig_set",
    "acos_nvram_get",
    "acos_nvram_set",
]

# Known NVRAM default file locations in firmware
_NVRAM_DEFAULT_PATHS = [
    "etc/nvram.default",
    "etc/nvram_default",
    "etc/nvram.ini",
    "tmp/nvram_default",
    "etc/default/nvram",
    "usr/etc/default",
    "etc/config.default",
    "rom/etc/nvram.default",
]

# Vendor detection patterns in firmware files
_VENDOR_PATTERNS: list[tuple[str, str]] = [
    (r"D-Link|dlink|DIR-\d+|DCS-\d+", "dlink"),
    (r"TP-LINK|tplink|TL-WR|TL-MR|Archer", "tplink"),
    (r"NETGEAR|netgear|R6\d+|R7\d+|RAX", "netgear"),
    (r"ASUS|RT-AC|RT-AX|GT-AX", "asus"),
    (r"OpenWrt|LEDE|openwrt", "openwrt"),
    (r"Hikvision|hikvision|HIKVISION|DS-\d+", "hikvision"),
    (r"Dahua|dahua|DH-IPC|DH-NVR", "dahua"),
]


def scan_rootfs_for_nvram(rootfs_path: str) -> NvramScanResult:
    """Scan an extracted rootfs to determine NVRAM requirements.

    Checks for:
    1. Binaries that reference nvram_get/nvram_set functions
    2. Existing NVRAM default files
    3. Vendor identification for default selection
    """
    result = NvramScanResult()
    root = Path(rootfs_path)

    if not root.exists():
        logger.error("Rootfs path does not exist: %s", rootfs_path)
        return result

    # --- Scan binaries for NVRAM function references ---
    binary_dirs = ["bin", "sbin", "usr/bin", "usr/sbin", "usr/lib", "lib"]
    for bdir in binary_dirs:
        dirpath = root / bdir
        if not dirpath.exists():
            continue

        for binfile in dirpath.iterdir():
            if not binfile.is_file():
                continue
            try:
                content = binfile.read_bytes()
                for func_name in _NVRAM_FUNCTIONS:
                    if func_name.encode() in content:
                        result.needs_nvram = True
                        rel_path = str(binfile.relative_to(root))
                        if rel_path not in result.nvram_binaries:
                            result.nvram_binaries.append(rel_path)
                        if func_name not in result.nvram_functions:
                            result.nvram_functions.append(func_name)
            except (PermissionError, OSError):
                continue

    # --- Look for existing NVRAM default files ---
    for nvram_path in _NVRAM_DEFAULT_PATHS:
        full_path = root / nvram_path
        if full_path.exists():
            result.default_files_found.append(nvram_path)
            try:
                content = full_path.read_text(errors="replace")
                parsed = parse_nvram_defaults(content)
                result.extracted_defaults.update(parsed)
                logger.info(
                    "Found NVRAM defaults at %s (%d keys)",
                    nvram_path, len(parsed),
                )
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", nvram_path, exc)

    # --- Vendor detection ---
    result.recommended_vendor = detect_vendor(rootfs_path)

    logger.info(
        "NVRAM scan: needed=%s, binaries=%d, functions=%s, defaults=%d keys, vendor=%s",
        result.needs_nvram,
        len(result.nvram_binaries),
        result.nvram_functions,
        len(result.extracted_defaults),
        result.recommended_vendor,
    )

    return result


def detect_vendor(rootfs_path: str) -> str:
    """Detect firmware vendor from rootfs contents.

    Checks banner files, binary strings, and filesystem layout.
    """
    root = Path(rootfs_path)

    # Check common files for vendor strings
    check_files = [
        "etc/banner",
        "etc/issue",
        "etc/hostname",
        "etc/version",
        "etc/model",
        "www/login.html",
        "www/index.html",
        "web/login.html",
        "usr/sbin/httpd",
    ]

    all_text = ""
    for fpath in check_files:
        full_path = root / fpath
        if full_path.exists():
            try:
                all_text += full_path.read_text(errors="replace")[:4096]
            except (PermissionError, OSError, UnicodeDecodeError):
                try:
                    all_text += full_path.read_bytes()[:4096].decode("ascii", errors="replace")
                except Exception:
                    continue

    # Match against vendor patterns
    for pattern, vendor in _VENDOR_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            logger.info("Detected vendor: %s", vendor)
            return vendor

    return "broadcom_generic"


# ---------------------------------------------------------------------------
# NVRAM defaults parsing and generation
# ---------------------------------------------------------------------------

def parse_nvram_defaults(content: str) -> dict[str, str]:
    """Parse an NVRAM defaults file into key-value pairs.

    Handles multiple formats:
    - key=value (most common)
    - key value (space-separated)
    - INI format [section] + key=value
    - Shell-style export KEY=value
    """
    defaults: dict[str, str] = {}

    for line in content.split("\n"):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Skip section headers
        if line.startswith("[") and line.endswith("]"):
            continue

        # Remove 'export' prefix
        if line.startswith("export "):
            line = line[7:]

        # Parse key=value
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not key.startswith("("):
                defaults[key] = value
        # Parse key<space>value (less common)
        elif "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                defaults[parts[0].strip()] = parts[1].strip()

    return defaults


def generate_nvram_ini(
    vendor: str = "broadcom_generic",
    extracted_defaults: dict[str, str] | None = None,
    extra_overrides: dict[str, str] | None = None,
    device_ip: str = "192.168.1.1",
) -> str:
    """Generate an NVRAM INI file for the libnvram intercept library.

    Priority (highest to lowest):
    1. extra_overrides (user-specified)
    2. extracted_defaults (from firmware's own default files)
    3. Vendor template defaults
    4. Generic Broadcom defaults (base)

    The IP address is injected into lan_ipaddr and related keys.
    """
    # Start with generic Broadcom base
    merged: dict[str, str] = dict(VENDOR_NVRAM_DEFAULTS.get("broadcom_generic", {}))

    # Layer vendor-specific defaults
    if vendor != "broadcom_generic" and vendor in VENDOR_NVRAM_DEFAULTS:
        merged.update(VENDOR_NVRAM_DEFAULTS[vendor])

    # Layer extracted defaults from firmware
    if extracted_defaults:
        merged.update(extracted_defaults)

    # Layer user overrides
    if extra_overrides:
        merged.update(extra_overrides)

    # Ensure IP configuration is consistent
    merged["lan_ipaddr"] = device_ip
    if "lan_gateway" in merged and merged["lan_gateway"] == "0.0.0.0":
        # Set gateway to .1 if on same subnet
        parts = device_ip.rsplit(".", 1)
        if len(parts) == 2:
            merged["lan_gateway"] = f"{parts[0]}.1"

    # Generate INI format
    lines = [
        "# IoTGhost NVRAM defaults",
        f"# Vendor family: {vendor}",
        f"# Generated keys: {len(merged)}",
        "",
    ]

    # Group by category for readability
    categories = {
        "Network": ["lan_", "wan_", "dns_", "dhcp_"],
        "HTTP": ["http_", "httpd_", "admin_", "remote_"],
        "WiFi": ["wl", "wlan_", "wifi_"],
        "System": ["os_", "model", "product", "vendor", "device_", "sys_",
                    "friendly_", "hostname", "firmver", "productid", "time_"],
        "Services": ["upnp_", "ntp_", "syslog_", "rtsp_", "server_",
                      "tcp_", "udp_"],
    }

    categorized: dict[str, list[tuple[str, str]]] = {cat: [] for cat in categories}
    uncategorized: list[tuple[str, str]] = []

    for key, value in sorted(merged.items()):
        placed = False
        for cat, prefixes in categories.items():
            if any(key.startswith(p) or key == p.rstrip("_") for p in prefixes):
                categorized[cat].append((key, value))
                placed = True
                break
        if not placed:
            uncategorized.append((key, value))

    for cat, items in categorized.items():
        if items:
            lines.append(f"# -- {cat} --")
            for key, value in items:
                lines.append(f"{key}={value}")
            lines.append("")

    if uncategorized:
        lines.append("# -- Other --")
        for key, value in uncategorized:
            lines.append(f"{key}={value}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NVRAM deployment into rootfs
# ---------------------------------------------------------------------------

def deploy_nvram(
    rootfs_path: str,
    nvram_config: NvramConfig,
    device_ip: str = "192.168.1.1",
) -> list[str]:
    """Deploy NVRAM emulation into the extracted rootfs.

    Steps:
    1. Write nvram.ini defaults file
    2. Copy libnvram.so intercept library (if available)
    3. Patch init scripts to set LD_PRELOAD

    Returns list of actions taken (for logging/display).
    """
    root = Path(rootfs_path)
    actions: list[str] = []

    # --- 1. Generate and write NVRAM defaults ---
    ini_content = generate_nvram_ini(
        vendor=nvram_config.vendor_family,
        extracted_defaults=nvram_config.extracted_keys,
        device_ip=device_ip,
    )

    ini_path = root / NVRAM_INI_PATH.lstrip("/")
    ini_path.parent.mkdir(parents=True, exist_ok=True)
    ini_path.write_text(ini_content)
    actions.append(f"Written NVRAM defaults to {NVRAM_INI_PATH} ({len(nvram_config.defaults)} keys)")
    logger.info("Deployed nvram.ini: %s", ini_path)

    # --- 2. Deploy libnvram.so ---
    lib_dest = root / NVRAM_LIB_PATH.lstrip("/")
    lib_dest.parent.mkdir(parents=True, exist_ok=True)

    if _LIBNVRAM_SRC.exists():
        shutil.copy2(_LIBNVRAM_SRC, lib_dest)
        actions.append(f"Deployed libnvram.so to {NVRAM_LIB_PATH}")
        logger.info("Deployed libnvram.so: %s", lib_dest)
    else:
        # Create a placeholder script that the AI agent can replace
        # with the correct architecture-specific library
        placeholder = (
            "#!/bin/sh\n"
            "# Placeholder: libnvram.so not found in package.\n"
            "# The AI agent should build or download the correct version\n"
            f"# for the firmware architecture and place it at {NVRAM_LIB_PATH}\n"
        )
        lib_dest.write_text(placeholder)
        actions.append(
            f"WARNING: libnvram.so not found in package. "
            f"Placeholder created at {NVRAM_LIB_PATH}. "
            "AI agent should build/download the correct library."
        )
        logger.warning("libnvram.so not found, created placeholder")

    # --- 3. Patch init scripts for LD_PRELOAD ---
    init_patched = _patch_init_for_nvram(root)
    if init_patched:
        actions.append(f"Patched init script: {init_patched}")
    else:
        # Create a wrapper init script
        _create_nvram_init_wrapper(root)
        actions.append("Created /etc/init.d/S00nvram wrapper for LD_PRELOAD")

    return actions


def _patch_init_for_nvram(rootfs: Path) -> str | None:
    """Patch existing init scripts to include LD_PRELOAD for libnvram.

    Returns the path of the patched script, or None if no suitable script found.
    """
    init_scripts = [
        "etc/init.d/rcS",
        "etc/rc.d/rc.sysinit",
        "etc/init.d/rc.local",
        "etc/preinit",
    ]

    for script_path in init_scripts:
        full_path = rootfs / script_path
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(errors="replace")

            # Skip if already patched
            if "LD_PRELOAD" in content and "libnvram" in content:
                logger.info("Init script already has LD_PRELOAD: %s", script_path)
                return script_path

            # Insert LD_PRELOAD after shebang line
            lines = content.split("\n")
            insert_idx = 1 if lines and lines[0].startswith("#!") else 0

            nvram_lines = [
                "",
                "# IoTGhost NVRAM emulation",
                f"export LD_PRELOAD={NVRAM_LIB_PATH}",
                f"export NVRAM_DEFAULTS_PATH={NVRAM_INI_PATH}",
                "",
            ]

            for i, nvram_line in enumerate(nvram_lines):
                lines.insert(insert_idx + i, nvram_line)

            full_path.write_text("\n".join(lines))
            logger.info("Patched init script: %s", script_path)
            return script_path

        except (PermissionError, OSError) as exc:
            logger.warning("Cannot patch %s: %s", script_path, exc)
            continue

    return None


def _create_nvram_init_wrapper(rootfs: Path) -> None:
    """Create a new init script that sets up NVRAM emulation early in boot."""
    wrapper_content = f"""#!/bin/sh
# IoTGhost NVRAM emulation init script
# This runs early in boot to ensure all services have NVRAM access

export LD_PRELOAD={NVRAM_LIB_PATH}
export NVRAM_DEFAULTS_PATH={NVRAM_INI_PATH}

# Create /dev/nvram if firmware expects it
if [ ! -e /dev/nvram ]; then
    mknod /dev/nvram c 228 0 2>/dev/null || true
fi

# Create /tmp/nvram for runtime storage
mkdir -p /tmp/nvram

# Copy defaults to runtime location
cp {NVRAM_INI_PATH} /tmp/nvram/nvram.ini 2>/dev/null || true

echo "[IoTGhost] NVRAM emulation active"
"""

    wrapper_path = rootfs / "etc/init.d/S00nvram"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)
    logger.info("Created NVRAM init wrapper: %s", wrapper_path)


# ---------------------------------------------------------------------------
# NVRAM key extraction from binaries
# ---------------------------------------------------------------------------

def extract_nvram_keys_from_binary(binary_path: str) -> list[str]:
    """Extract likely NVRAM key names from a firmware binary.

    Uses string analysis to find patterns like:
    - nvram_get("key_name")
    - Strings that look like NVRAM keys (lowercase_with_underscores)
    """
    keys: list[str] = []

    try:
        content = Path(binary_path).read_bytes()
        text = content.decode("ascii", errors="replace")

        # Pattern 1: nvram_get("key") / nvram_safe_get("key")
        for match in re.finditer(r'nvram_(?:safe_)?get\s*\(\s*"([^"]+)"', text):
            key = match.group(1)
            if _is_valid_nvram_key(key):
                keys.append(key)

        # Pattern 2: Common NVRAM key patterns in strings
        # Look for strings that match typical key naming conventions
        for match in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,5})\b', text):
            candidate = match.group(1)
            if candidate in _KNOWN_NVRAM_KEYS and candidate not in keys:
                keys.append(candidate)

    except Exception as exc:
        logger.warning("Failed to extract NVRAM keys from %s: %s", binary_path, exc)

    return sorted(set(keys))


def _is_valid_nvram_key(key: str) -> bool:
    """Check if a string looks like a valid NVRAM key."""
    if len(key) < 2 or len(key) > 64:
        return False
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_.-]*$', key):
        return False
    # Filter out common false positives
    if key in {"if", "do", "done", "then", "else", "fi", "for", "while", "in"}:
        return False
    return True


# Common NVRAM keys found across many firmware images
_KNOWN_NVRAM_KEYS = {
    "lan_ipaddr", "lan_netmask", "lan_gateway", "lan_proto", "lan_hwaddr",
    "wan_ipaddr", "wan_netmask", "wan_gateway", "wan_proto", "wan_hwaddr",
    "wan_dns", "wan_hostname",
    "http_lanport", "http_enable", "http_username", "http_passwd",
    "wl0_ssid", "wl0_mode", "wl0_radio", "wl0_security_mode",
    "wl0_wpa_psk", "wl0_crypto", "wl0_channel",
    "os_name", "os_version", "model_name", "device_name",
    "time_zone", "ntp_server", "ntp_enable",
    "remote_management", "remote_port",
    "upnp_enable", "syslog_enable",
    "admin_user", "admin_passwd",
    "hostname", "firmver", "productid",
}
