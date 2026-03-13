# IoTGhost

**AI-powered IoT firmware emulator** -- autonomous QEMU full-system emulation driven by LLM shell agents.

Feed it a firmware binary. It handles everything: extraction, architecture detection, NVRAM emulation, kernel configuration, network setup, and iterative error correction. **Zero prompts required** -- the AI agent works autonomously in the background.

## How It Works

```
firmware.bin --> [Extract] --> [Analyze] --> [Prepare] --> [Emulate] --> [Fix Loop] --> [Verify]
                    |             |             |             |              |              |
                 binwalk      detect arch    NVRAM setup   QEMU boot    AI diagnoses   services
                 unpack       find kernel    patch libs    serial mon   fixes errors    reachable
                 rootfs       scan NVRAM     device nodes  network up   retry boot      web UI up
```

The AI agent receives expert system prompts encoding deep IoT emulation knowledge -- the same techniques a senior firmware security researcher would apply manually:

- **NVRAM emulation** for the ~53% of firmware that depends on `libnvram`
- **Architecture detection** from ELF headers (MIPS, ARM, x86, AARCH64)
- **Kernel panic diagnosis** with targeted fix strategies
- **Vendor-specific tricks** for D-Link, TP-Link, Netgear, ASUS, Hikvision, Dahua, MikroTik, QNAP
- **Self-healing loop** -- when emulation fails, the AI reads errors, patches, and retries

## Quick Start

### Prerequisites

```bash
# System dependencies
sudo apt install qemu-system binwalk squashfs-tools

# Optional but recommended
pip install jefferson ubi_reader
git clone https://github.com/devttys0/sasquatch && cd sasquatch && sudo ./build.sh

# Ollama (LLM provider)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull glm4:latest   # or any model you prefer
```

### Install IoTGhost

```bash
pip install .
# or for development:
pip install -e ".[dev]"
```

### Run

```bash
# Basic usage -- AI handles everything
iotghost run firmware.bin --ollama --model glm4:latest

# Specify architecture if auto-detection fails
iotghost run DIR-850L_fw.bin --arch mipsel --verbose

# Unprivileged mode (no root needed for network)
iotghost run camera.bin --network user

# Plain text output (no TUI)
iotghost run firmware.bin --no-tui --verbose

# Analyze without emulating
iotghost info firmware.bin

# Check system dependencies
iotghost check-deps
```

## Architecture

```
iotghost/
|-- cli.py              # Click CLI entry point
|-- pipeline.py         # Main orchestration (extract -> verify loop)
|-- agent.py            # LLM shell agent (Ollama API, tool-use loop)
|-- prompts.py          # Expert system prompts + vendor NVRAM defaults
|-- emulator.py         # QEMU process manager + boot monitoring
|-- nvram.py            # NVRAM emulation (libnvram.so LD_PRELOAD)
|-- network.py          # TAP/bridge/NAT network setup
|-- tui.py              # Rich TUI dashboard
|-- tools/
|   |-- __init__.py
|   |-- shell.py        # Shell tools: execute_command, read_file, etc.
|-- extractors/
|   |-- __init__.py
|   |-- firmware.py     # Firmware extraction (binwalk, sasquatch, etc.)
|-- kernels/            # Pre-built QEMU kernels (user-supplied)
|-- nvram_defaults/     # Vendor NVRAM config templates
```

### Component Flow

```
CLI (cli.py)
  |
  v
Pipeline (pipeline.py) ---- orchestrates phases, manages state
  |
  |-- ShellAgent (agent.py) ---- LLM brain, calls tools autonomously
  |     |
  |     |-- Ollama API ---- sends prompts, receives tool calls
  |     |-- Tool Registry ---- execute_command, read_file, etc.
  |     |-- Context Window ---- accumulates knowledge, auto-trims
  |
  |-- QemuManager (emulator.py) ---- launches QEMU, monitors serial
  |-- NetworkManager (network.py) ---- TAP, bridge, NAT, DHCP
  |-- NVRAM (nvram.py) ---- scans dependencies, deploys libnvram.so
  |-- Extractors (extractors/) ---- binwalk, sasquatch, jefferson
  |
  v
TUI (tui.py) ---- Rich dashboard with live panels
```

## CLI Reference

### `iotghost run`

| Flag | Default | Description |
|------|---------|-------------|
| `--ollama` | `true` | Use Ollama as LLM provider |
| `--model` | `glm4:latest` | LLM model name |
| `--base-url` | `http://localhost:11434` | Ollama API URL |
| `--arch` | `auto` | Force architecture (mipsel/mipsbe/armel/arm64/x86/x86_64) |
| `--network` | `user` | Network mode: tap (root), user (unprivileged), none |
| `--device-ip` | `192.168.1.1` | Emulated device IP |
| `--timeout` | `120` | Boot timeout (seconds) |
| `--max-retries` | `5` | Max fix-and-retry attempts |
| `--max-iterations` | `50` | Max LLM iterations per phase |
| `--workdir` | auto | Working directory for extraction |
| `--kernels-dir` | built-in | Directory with pre-built QEMU kernels |
| `--temperature` | `0.1` | LLM temperature |
| `--verbose` / `-v` | `false` | Detailed output |
| `--debug` | `false` | Debug logging |
| `--no-tui` | `false` | Plain text instead of Rich TUI |

### `iotghost info`

Analyze a firmware image without emulating. Reports architecture, filesystem type, NVRAM requirements, and encryption status.

### `iotghost check-deps`

Verify that required system tools (QEMU, binwalk, etc.) and Ollama are available.

## Supported Architectures

| Architecture | QEMU Binary | Machine | Typical Devices |
|-------------|-------------|---------|------------------|
| MIPS LE | qemu-system-mipsel | malta | Most routers (D-Link, TP-Link, Netgear) |
| MIPS BE | qemu-system-mips | malta | Some routers, IP cameras |
| ARM LE | qemu-system-arm | virt | IP cameras, IoT hubs, newer routers |
| ARM64 | qemu-system-aarch64 | virt | High-end NAS, newer cameras |
| x86 | qemu-system-i386 | pc | MikroTik, some NAS devices |
| x86_64 | qemu-system-x86_64 | pc | QNAP, Synology NAS |

## How the AI Agent Works

The agent operates as an autonomous shell agent with a tool-use loop:

1. **Receives a phase prompt** with expert instructions (e.g., "Extract this firmware")
2. **Generates tool calls** to execute shell commands, read files, etc.
3. **Observes results** -- reads stdout/stderr from each command
4. **Decides next action** based on accumulated context
5. **Loops** until the phase objective is complete or it gets stuck

The agent has NO access to the internet and NO ability to ask the user questions. It must solve every problem using only:
- Shell commands (binwalk, QEMU, file operations, network tools)
- Its expert knowledge of IoT firmware internals
- Its reasoning ability to diagnose and fix errors

### Error Recovery

When emulation fails (kernel panic, service crash, network issue), the pipeline:

1. Captures the last 30 lines of QEMU serial output
2. Feeds the error + history of previous fix attempts to the AI
3. AI diagnoses the root cause and applies a targeted fix
4. Pipeline retries emulation
5. Repeats up to `--max-retries` times (default 5)

## NVRAM Emulation

~53% of IoT firmware depends on NVRAM for configuration. IoTGhost handles this automatically:

1. **Scans** binaries for `nvram_get`/`nvram_set` function references
2. **Detects** the vendor (D-Link, TP-Link, Netgear, ASUS, etc.)
3. **Extracts** existing NVRAM defaults from the firmware
4. **Generates** a merged NVRAM config with vendor-specific keys
5. **Deploys** `libnvram.so` via `LD_PRELOAD` to intercept calls
6. **Patches** init scripts to load NVRAM emulation early in boot

Built-in vendor profiles: Broadcom (generic), D-Link, TP-Link, Netgear, ASUS, OpenWrt, Hikvision, Dahua.

## Pre-built Kernels

IoTGhost needs pre-built QEMU kernels matching common firmware versions. Place them in the `kernels/` directory:

```
kernels/
|-- vmlinux.mipsel.2.6
|-- vmlinux.mipsel.3.2
|-- vmlinux.mipsel.4.1
|-- vmlinux.mipsbe.2.6
|-- vmlinux.mipsbe.4.1
|-- zImage.armel.3.10
|-- zImage.armel.4.1
|-- Image.arm64.4.9
|-- Image.arm64.5.4
|-- bzImage.x86.4.1
```

You can build these from the Linux kernel source or download from:
- [FirmAE kernels](https://github.com/pr0v3rbs/FirmAE)
- [Firmadyne kernels](https://github.com/firmadyne/firmadyne)

Or specify a custom directory: `--kernels-dir /path/to/kernels`

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check iotghost/

# Type check
mypy iotghost/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

IoTGhost builds on the shoulders of:
- [Firmadyne](https://github.com/firmadyne/firmadyne) -- pioneering firmware emulation research
- [FirmAE](https://github.com/pr0v3rbs/FirmAE) -- improved emulation techniques
- [binwalk](https://github.com/ReFirmLabs/binwalk) -- firmware extraction
- [QEMU](https://www.qemu.org/) -- system emulation
- [Ollama](https://ollama.ai/) -- local LLM inference
- [Rich](https://github.com/Textualize/rich) -- terminal UI
