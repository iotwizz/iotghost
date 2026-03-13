"""Network setup -- TAP devices, bridges, and IP configuration for QEMU.

Creates the host-side network infrastructure needed for the emulated
firmware to communicate. Handles TAP device creation, bridge setup,
IP assignment, and cleanup. Also provides utilities for the AI agent
to diagnose network issues inside the emulated environment.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network configuration for the emulation session."""
    # Host-side config
    tap_name: str = "tap0"
    bridge_name: str = "iotbr0"
    host_ip: str = "192.168.1.100"
    host_netmask: str = "255.255.255.0"
    host_cidr: int = 24

    # Emulated device config
    device_ip: str = "192.168.1.1"
    device_netmask: str = "255.255.255.0"
    device_gateway: str = "192.168.1.100"  # host acts as gateway

    # DHCP (optional dnsmasq for the emulated device)
    enable_dhcp: bool = False
    dhcp_range_start: str = "192.168.1.50"
    dhcp_range_end: str = "192.168.1.150"

    # NAT for internet access from emulated device
    enable_nat: bool = True
    nat_interface: str = "eth0"  # host's internet-facing interface

    # QEMU network mode
    qemu_mode: str = "tap"  # tap, user, none


@dataclass
class NetworkState:
    """Current state of network setup."""
    tap_created: bool = False
    bridge_created: bool = False
    ip_assigned: bool = False
    nat_enabled: bool = False
    dhcp_running: bool = False
    dhcp_pid: int | None = None
    errors: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)  # commands executed


# ---------------------------------------------------------------------------
# Network setup and teardown
# ---------------------------------------------------------------------------

class NetworkManager:
    """Manages host-side network configuration for QEMU emulation.

    Creates TAP device, bridge, assigns IPs, sets up NAT, and optionally
    runs a DHCP server. All operations require root privileges.
    """

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self.state = NetworkState()

    def _run(self, cmd: str, check: bool = True) -> tuple[bool, str]:
        """Execute a network configuration command.

        Returns (success, output) tuple.
        """
        logger.debug("NET CMD: %s", cmd)
        self.state.setup_commands.append(cmd)

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10,
            )
            output = (result.stdout + result.stderr).strip()

            if result.returncode != 0 and check:
                logger.warning("Command failed: %s -> %s", cmd, output)
                self.state.errors.append(f"{cmd}: {output}")
                return False, output

            return True, output
        except subprocess.TimeoutExpired:
            err = f"Timeout: {cmd}"
            self.state.errors.append(err)
            return False, err
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self.state.errors.append(err)
            return False, err

    def setup(self) -> bool:
        """Perform full network setup. Returns True if successful."""
        logger.info("Setting up network: tap=%s, bridge=%s", 
                     self.config.tap_name, self.config.bridge_name)

        if not self._check_root():
            self.state.errors.append(
                "Root privileges required for network setup. "
                "Run with sudo or use --network user for unprivileged mode."
            )
            return False

        steps = [
            ("Creating TAP device", self._create_tap),
            ("Creating bridge", self._create_bridge),
            ("Assigning IP addresses", self._assign_ips),
        ]

        if self.config.enable_nat:
            steps.append(("Setting up NAT", self._setup_nat))

        if self.config.enable_dhcp:
            steps.append(("Starting DHCP server", self._start_dhcp))

        for desc, func in steps:
            logger.info("Network setup: %s", desc)
            if not func():
                logger.error("Network setup failed at: %s", desc)
                return False

        logger.info("Network setup complete")
        return True

    def teardown(self) -> None:
        """Clean up all network resources."""
        logger.info("Tearing down network configuration")

        # Stop DHCP if running
        if self.state.dhcp_running and self.state.dhcp_pid:
            self._run(f"kill {self.state.dhcp_pid}", check=False)
            self.state.dhcp_running = False

        # Remove NAT rules
        if self.state.nat_enabled:
            self._run(
                f"iptables -t nat -D POSTROUTING -o {self.config.nat_interface} "
                f"-j MASQUERADE",
                check=False,
            )
            self._run(
                f"iptables -D FORWARD -i {self.config.bridge_name} -j ACCEPT",
                check=False,
            )
            self.state.nat_enabled = False

        # Remove bridge
        if self.state.bridge_created:
            self._run(f"ip link set {self.config.bridge_name} down", check=False)
            self._run(f"brctl delbr {self.config.bridge_name}", check=False)
            self.state.bridge_created = False

        # Remove TAP device
        if self.state.tap_created:
            self._run(f"ip link set {self.config.tap_name} down", check=False)
            self._run(f"ip tuntap del dev {self.config.tap_name} mode tap", check=False)
            self.state.tap_created = False

        logger.info("Network teardown complete")

    def _check_root(self) -> bool:
        """Check if we have root privileges."""
        return os.geteuid() == 0

    def _create_tap(self) -> bool:
        """Create a TAP network device."""
        tap = self.config.tap_name

        # Check if TAP already exists
        ok, _ = self._run(f"ip link show {tap}", check=False)
        if ok:
            logger.info("TAP device %s already exists", tap)
            self.state.tap_created = True
            return True

        ok, _ = self._run(f"ip tuntap add dev {tap} mode tap")
        if not ok:
            return False

        ok, _ = self._run(f"ip link set {tap} up")
        if not ok:
            return False

        self.state.tap_created = True
        return True

    def _create_bridge(self) -> bool:
        """Create a network bridge and add the TAP device to it."""
        br = self.config.bridge_name
        tap = self.config.tap_name

        # Check if bridge already exists
        ok, _ = self._run(f"ip link show {br}", check=False)
        if ok:
            logger.info("Bridge %s already exists", br)
            self.state.bridge_created = True
        else:
            ok, _ = self._run(f"brctl addbr {br}")
            if not ok:
                # Try ip command as fallback
                ok, _ = self._run(f"ip link add name {br} type bridge")
                if not ok:
                    return False
            self.state.bridge_created = True

        # Add TAP to bridge
        self._run(f"brctl addif {br} {tap}", check=False)
        self._run(f"ip link set {br} up")

        return True

    def _assign_ips(self) -> bool:
        """Assign IP addresses to the bridge interface."""
        br = self.config.bridge_name
        ip_cidr = f"{self.config.host_ip}/{self.config.host_cidr}"

        # Flush existing IPs on bridge
        self._run(f"ip addr flush dev {br}", check=False)

        ok, _ = self._run(f"ip addr add {ip_cidr} dev {br}")
        if not ok:
            return False

        self.state.ip_assigned = True
        return True

    def _setup_nat(self) -> bool:
        """Configure NAT for internet access from the emulated device."""
        nat_if = self.config.nat_interface
        br = self.config.bridge_name

        # Enable IP forwarding
        self._run("sysctl -w net.ipv4.ip_forward=1")

        # NAT masquerade rule
        ok, _ = self._run(
            f"iptables -t nat -A POSTROUTING -o {nat_if} -j MASQUERADE"
        )
        if not ok:
            return False

        # Allow forwarding from bridge
        self._run(f"iptables -A FORWARD -i {br} -j ACCEPT")
        self._run(f"iptables -A FORWARD -o {br} -m state --state RELATED,ESTABLISHED -j ACCEPT")

        self.state.nat_enabled = True
        return True

    def _start_dhcp(self) -> bool:
        """Start a DHCP server for the emulated device."""
        br = self.config.bridge_name
        start = self.config.dhcp_range_start
        end = self.config.dhcp_range_end

        # Use dnsmasq for DHCP
        cmd = (
            f"dnsmasq --interface={br} "
            f"--dhcp-range={start},{end},12h "
            f"--bind-interfaces --no-daemon "
            f"--log-queries "
            f"--pid-file=/tmp/iotghost_dhcp.pid "
            f"&"
        )
        ok, _ = self._run(cmd, check=False)

        if ok:
            self.state.dhcp_running = True
            # Read PID
            try:
                pid_content = Path("/tmp/iotghost_dhcp.pid").read_text().strip()
                self.state.dhcp_pid = int(pid_content)
            except Exception:
                pass

        return True  # DHCP is optional, don't fail setup

    def get_qemu_network_args(self) -> list[str]:
        """Generate QEMU network command-line arguments."""
        if self.config.qemu_mode == "tap":
            return [
                "-netdev",
                f"tap,id=net0,ifname={self.config.tap_name},script=no,downscript=no",
                "-device",
                "e1000,netdev=net0",
            ]
        elif self.config.qemu_mode == "user":
            return [
                "-netdev",
                "user,id=net0,hostfwd=tcp::8080-:80,hostfwd=tcp::2222-:22,"
                "hostfwd=tcp::4443-:443",
                "-device",
                "e1000,netdev=net0",
            ]
        return []  # no network

    def get_status(self) -> dict[str, Any]:
        """Return network status summary."""
        return {
            "tap_device": self.config.tap_name,
            "tap_created": self.state.tap_created,
            "bridge": self.config.bridge_name,
            "bridge_created": self.state.bridge_created,
            "host_ip": self.config.host_ip,
            "device_ip": self.config.device_ip,
            "nat_enabled": self.state.nat_enabled,
            "dhcp_running": self.state.dhcp_running,
            "errors": self.state.errors,
            "mode": self.config.qemu_mode,
        }


# ---------------------------------------------------------------------------
# Network diagnostic utilities for the AI agent
# ---------------------------------------------------------------------------

def detect_host_interface() -> str | None:
    """Auto-detect the host's primary internet-facing interface.

    Used to configure NAT correctly.
    """
    try:
        result = subprocess.run(
            "ip route | grep default | awk '{print $5}' | head -1",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        iface = result.stdout.strip()
        if iface:
            logger.info("Detected host interface: %s", iface)
            return iface
    except Exception:
        pass
    return None


def check_port_accessible(ip: str, port: int, timeout: float = 3.0) -> bool:
    """Quick TCP port check -- used by the AI agent to verify services."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def scan_common_ports(ip: str, timeout: float = 2.0) -> list[tuple[int, str]]:
    """Scan common IoT service ports. Returns list of (port, service_name)."""
    common_ports = [
        (22, "SSH"),
        (23, "Telnet"),
        (80, "HTTP"),
        (443, "HTTPS"),
        (554, "RTSP"),
        (8080, "HTTP-Alt"),
        (8443, "HTTPS-Alt"),
        (8000, "HTTP-Service"),
        (8888, "HTTP-Proxy"),
        (37777, "Dahua-TCP"),
        (37778, "Dahua-UDP"),
        (49152, "UPnP"),
    ]

    open_ports = []
    for port, service in common_ports:
        if check_port_accessible(ip, port, timeout):
            open_ports.append((port, service))
            logger.info("Port %d (%s) open on %s", port, service, ip)

    return open_ports


def generate_network_init_script(
    device_ip: str = "192.168.1.1",
    netmask: str = "255.255.255.0",
    gateway: str = "192.168.1.100",
    interface: str = "eth0",
) -> str:
    """Generate a network init script to inject into the firmware rootfs.

    This ensures the emulated device configures its network interface
    even if its normal config mechanism fails.
    """
    return f"""#!/bin/sh
# IoTGhost network initialization
# Ensures network comes up even if firmware's own config fails

# Wait for interface to appear
sleep 2

# Configure interface
ifconfig {interface} {device_ip} netmask {netmask} up 2>/dev/null || \\
  ip addr add {device_ip}/{_netmask_to_cidr(netmask)} dev {interface} 2>/dev/null

# Add default route
route add default gw {gateway} 2>/dev/null || \\
  ip route add default via {gateway} 2>/dev/null

# Set DNS
echo "nameserver 8.8.8.8" > /tmp/resolv.conf
cp /tmp/resolv.conf /etc/resolv.conf 2>/dev/null || true

echo "[IoTGhost] Network configured: {interface} = {device_ip}"
"""


def _netmask_to_cidr(netmask: str) -> int:
    """Convert subnet mask to CIDR notation."""
    try:
        return ipaddress.IPv4Network(f"0.0.0.0/{netmask}").prefixlen
    except ValueError:
        return 24  # default
