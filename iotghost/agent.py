"""LLM shell agent core -- autonomous tool-use loop with Ollama integration.

The agent receives a system prompt with deep IoT emulation knowledge,
then iteratively executes shell commands, reads their output, and decides
the next action. It accumulates context across steps and implements
retry logic with backoff for failed approaches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import httpx

from iotghost.prompts import (
    BINARY_FIX_PROMPT,
    KERNEL_BUILD_PROMPT,
    NVRAM_RECOVERY_PROMPT,
    SELF_HEAL_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Single message in the agent's conversation history."""
    role: Role
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    output: str
    success: bool
    duration: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the LLM agent."""
    provider: str = "ollama"
    model: str = "glm4:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 4096
    max_iterations: int = 50
    max_retries_per_phase: int = 5
    command_timeout: int = 120
    context_window: int = 32000
    verbose: bool = False


@dataclass
class AgentState:
    """Mutable state tracked across the agent's execution."""
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    phase: str = "init"
    context_vars: dict[str, str] = field(default_factory=dict)
    attempted_fixes: list[str] = field(default_factory=list)
    errors_seen: list[str] = field(default_factory=list)
    phase_retries: int = 0
    total_commands_run: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


# ---------------------------------------------------------------------------
# Self-Healing Layer 2 -- Error Tracker + Stuck Detection
# ---------------------------------------------------------------------------

# Error categories for classification
ERROR_CATEGORIES = {
    "permission": re.compile(r"Permission denied|EACCES|Operation not permitted", re.I),
    "missing_file": re.compile(r"No such file or directory|ENOENT|cannot stat", re.I),
    "exists": re.compile(r"File exists|EEXIST|already exists|cannot create", re.I),
    "timeout": re.compile(r"TIMEOUT|timed? ?out|deadline exceeded", re.I),
    "crash": re.compile(r"Segmentation fault|SIGSEGV|core dump|SIGABRT|killed", re.I),
    "disk": re.compile(r"No space left|ENOSPC|Disk quota exceeded", re.I),
    "network": re.compile(r"Connection refused|ECONNREFUSED|Network unreachable", re.I),
    "busy": re.compile(r"Device or resource busy|EBUSY", re.I),
    "kernel_panic": re.compile(r"Kernel panic|not syncing|VFS:.*Unable to mount", re.I),
    "arch_mismatch": re.compile(r"Illegal instruction|SIGILL|Invalid ELF|wrong ELF class|exec format error", re.I),
    "nvram": re.compile(r"nvram|libnvram|nvram_get|nvram\.ini", re.I),
}


@dataclass
class ErrorRecord:
    """Single recorded error from a tool execution."""
    iteration: int
    command: str
    error_text: str
    category: str       # key from ERROR_CATEGORIES or 'unknown'
    cmd_hash: str       # hash of command for repetition detection
    timestamp: float = field(default_factory=time.time)


class ErrorTracker:
    """Tracks command failures with pattern detection for self-healing.

    Detects three kinds of stuck states:
    - 'repeater': same exact command failing repeatedly (3+ times)
    - 'looper': cycling between 2-3 failing commands
    - 'stalled': 5+ consecutive failures regardless of command

    When stuck is detected, the agent injects a corrective prompt
    forcing a different approach.
    """

    HISTORY_SIZE = 50
    REPEAT_THRESHOLD = 3    # same command N times = repeater
    LOOP_WINDOW = 8         # look at last N commands for loop detection
    STALL_THRESHOLD = 5     # N consecutive failures = stalled

    def __init__(self) -> None:
        self.error_history: deque[ErrorRecord] = deque(maxlen=self.HISTORY_SIZE)
        self.cmd_hashes: deque[str] = deque(maxlen=self.LOOP_WINDOW * 2)
        self.consecutive_failures: int = 0
        self.total_failures: int = 0
        self.errors_by_category: dict[str, int] = {}
        self._corrective_injections: int = 0
        self.auto_fix_failures: int = 0  # Layer 1 shell auto-fix failures

    def reset(self) -> None:
        """Reset all tracking (called at phase start)."""
        self.error_history.clear()
        self.cmd_hashes.clear()
        self.consecutive_failures = 0
        self.total_failures = 0
        self.errors_by_category.clear()
        self._corrective_injections = 0
        self.auto_fix_failures = 0

    def record_success(self) -> None:
        """Record a successful command -- reduces consecutive failure count."""
        self.consecutive_failures = max(0, self.consecutive_failures - 1)

    def record_error(self, iteration: int, command: str, error_text: str) -> None:
        """Record a command failure and classify it."""
        category = self._classify(error_text)
        cmd_hash = hashlib.md5(command.strip().encode()).hexdigest()[:12]

        record = ErrorRecord(
            iteration=iteration,
            command=command[:200],
            error_text=error_text[:500],
            category=category,
            cmd_hash=cmd_hash,
        )
        self.error_history.append(record)
        self.cmd_hashes.append(cmd_hash)
        self.consecutive_failures += 1
        self.total_failures += 1
        self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1

        logger.debug(
            "ErrorTracker: %s (cat=%s, consecutive=%d, total=%d)",
            command[:60], category, self.consecutive_failures, self.total_failures,
        )

    def detect_stuck(self) -> str | None:
        """Detect if the agent is stuck. Returns stuck type or None.

        Returns:
            'repeater' -- same command failing 3+ times
            'looper'   -- cycling between 2-3 failing commands
            'stalled'  -- 5+ consecutive failures of any kind
            None       -- not stuck
        """
        if self.consecutive_failures < 2:
            return None

        # Check repeater: last N hashes are all the same
        if len(self.cmd_hashes) >= self.REPEAT_THRESHOLD:
            recent = list(self.cmd_hashes)[-self.REPEAT_THRESHOLD:]
            if len(set(recent)) == 1:
                return "repeater"

        # Check looper: small set of hashes cycling in the window
        if len(self.cmd_hashes) >= self.LOOP_WINDOW:
            window = list(self.cmd_hashes)[-self.LOOP_WINDOW:]
            unique = set(window)
            if len(unique) <= 3 and len(window) >= self.LOOP_WINDOW:
                return "looper"

        # Check stalled: too many consecutive failures
        if self.consecutive_failures >= self.STALL_THRESHOLD:
            return "stalled"

        return None

    def build_diagnostic_context(self) -> str:
        """Build a structured error summary for corrective prompt injection.

        Returns a formatted string with:
        - Error category breakdown
        - Last N unique failed commands with their errors
        - Stuck type diagnosis
        """
        stuck_type = self.detect_stuck()
        lines = [
            "=== SELF-HEALING DIAGNOSTIC ===",
            f"Consecutive failures: {self.consecutive_failures}",
            f"Total failures this phase: {self.total_failures}",
            f"Stuck type: {stuck_type or 'none'}",
            "",
            "Error breakdown by category:",
        ]

        for cat, count in sorted(self.errors_by_category.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")

        # Show last 5 unique errors
        seen_hashes: set[str] = set()
        unique_errors: list[ErrorRecord] = []
        for rec in reversed(self.error_history):
            if rec.cmd_hash not in seen_hashes:
                seen_hashes.add(rec.cmd_hash)
                unique_errors.append(rec)
            if len(unique_errors) >= 5:
                break

        if unique_errors:
            lines.append("")
            lines.append("Recent unique failures:")
            for rec in unique_errors:
                lines.append(f"  [{rec.category}] $ {rec.command}")
                # Show first line of error
                first_line = rec.error_text.strip().split("\n")[0][:120]
                lines.append(f"    -> {first_line}")

        lines.append("=== END DIAGNOSTIC ===")
        return "\n".join(lines)

    def _classify(self, error_text: str) -> str:
        """Classify error text into a category."""
        for cat, pattern in ERROR_CATEGORIES.items():
            if pattern.search(error_text):
                return cat
        return "unknown"

    @property
    def has_errors(self) -> bool:
        return self.total_failures > 0

    def record_auto_fix_failure(self) -> None:
        """Record a failed auto-fix from shell Layer 1.

        This feeds Layer 1 failures into Layer 2 stuck detection,
        so the ErrorTracker escalates sooner when auto-fix keeps failing.
        """
        self.auto_fix_failures += 1
        # Every 3 auto-fix failures counts as 1 consecutive failure
        if self.auto_fix_failures % 3 == 0:
            self.consecutive_failures += 1

    def dominant_category(self) -> str | None:
        """Return the error category with the most occurrences, or None.

        Used to select the matching expert prompt for recurring errors.
        """
        if not self.errors_by_category:
            return None
        top = max(self.errors_by_category, key=self.errors_by_category.get)
        # Only return if category has 2+ occurrences (not a one-off)
        if self.errors_by_category[top] >= 2:
            return top
        return None

    def get_expert_prompt(self) -> str | None:
        """Select the best expert prompt based on recurring error categories.

        Maps dominant error categories to specialized recovery prompts.
        Returns None if no clear category dominates.
        """
        category = self.dominant_category()
        if category is None:
            return None

        # Map error categories to expert prompts
        category_to_prompt: dict[str, str] = {
            "kernel_panic": KERNEL_BUILD_PROMPT,
            "arch_mismatch": KERNEL_BUILD_PROMPT,
            "crash": BINARY_FIX_PROMPT,
            "nvram": NVRAM_RECOVERY_PROMPT,
            "missing_file": BINARY_FIX_PROMPT,  # often missing libs
        }

        return category_to_prompt.get(category)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Tools are registered by the pipeline before agent execution.
# Each tool is a callable: (name, args) -> ToolResult
ToolFunction = Callable[..., ToolResult]

# Global tool registry populated by tools/ module
_tool_registry: dict[str, ToolFunction] = {}


def register_tool(name: str, func: ToolFunction) -> None:
    """Register a tool function the agent can invoke."""
    _tool_registry[name] = func
    logger.debug("Registered tool: %s", name)


def get_tool(name: str) -> ToolFunction | None:
    """Look up a registered tool by name."""
    return _tool_registry.get(name)


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_tool_registry.keys())


def get_tool_schemas() -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool schemas from registered tools."""
    schemas = []
    for name, func in _tool_registry.items():
        # Extract schema from function's _schema attribute if present
        schema = getattr(func, "_schema", None)
        if schema:
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {}),
                },
            })
    return schemas


# ---------------------------------------------------------------------------
# Ollama API client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Minimal Ollama chat completions client with tool-use support."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to Ollama.

        Uses /api/chat endpoint which supports tool calling natively.
        Falls back to parsing JSON tool calls from plain text if the
        model doesn't support native tool use.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        logger.debug("Ollama request: model=%s, messages=%d", self.config.model, len(messages))

        try:
            resp = self.client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error: %s", exc.response.text)
            raise
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# Response parsing -- handles both native tool calls and text-based fallback
# ---------------------------------------------------------------------------

def parse_tool_calls(response: dict[str, Any]) -> tuple[str, list[ToolCall]]:
    """Extract assistant text and tool calls from Ollama response.

    Ollama models that support tool use return structured tool_calls.
    For models that don't, we parse JSON tool invocations from the text.
    """
    message = response.get("message", {})
    content = message.get("content", "")
    tool_calls: list[ToolCall] = []

    # --- Native tool calls (Ollama 0.4+ with capable models) ---
    raw_calls = message.get("tool_calls", [])
    for i, call in enumerate(raw_calls):
        func = call.get("function", {})
        tool_calls.append(ToolCall(
            id=f"call_{i}",
            name=func.get("name", ""),
            arguments=func.get("arguments", {}),
        ))

    # --- Fallback: parse tool calls from text ---
    if not tool_calls and content:
        tool_calls = _parse_text_tool_calls(content)

    return content, tool_calls


def _parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls embedded in assistant text.

    Supports formats:
    1. ```tool_call\\n{"name": "...", "arguments": {...}}\\n```
    2. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    3. Direct JSON: {"tool": "...", "args": {...}}
    """
    import re
    calls: list[ToolCall] = []

    # Pattern 1: fenced code blocks
    fenced = re.findall(r"```(?:tool_call|json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    for block in fenced:
        parsed = _try_parse_call(block.strip())
        if parsed:
            calls.append(parsed)

    # Pattern 2: XML-style tags
    if not calls:
        tagged = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        for block in tagged:
            parsed = _try_parse_call(block.strip())
            if parsed:
                calls.append(parsed)

    # Pattern 3: bare JSON (last resort)
    if not calls:
        json_blocks = re.findall(r"\{[^{}]*\"(?:name|tool)\"[^{}]*\}", text)
        for block in json_blocks:
            parsed = _try_parse_call(block)
            if parsed:
                calls.append(parsed)
                break  # Only take first to avoid false positives

    return calls


def _try_parse_call(text: str) -> ToolCall | None:
    """Attempt to parse a single tool call from JSON text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    # Normalize different key conventions
    name = data.get("name") or data.get("tool") or data.get("function", "")
    args = data.get("arguments") or data.get("args") or data.get("parameters", {})

    if not name:
        return None

    return ToolCall(
        id=f"call_parsed_{hash(text) % 10000}",
        name=str(name),
        arguments=args if isinstance(args, dict) else {},
    )


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

class ShellAgent:
    """Autonomous LLM agent that drives firmware emulation via shell tools.

    The agent maintains a conversation with the LLM, feeding it tool results
    and accumulating context across iterations. It implements a self-healing
    loop: when emulation fails, the agent diagnoses errors and applies fixes
    without human intervention.
    """

    def __init__(
        self,
        config: AgentConfig,
        on_message: Callable[[Message], None] | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_phase_change: Callable[[str, str], None] | None = None,
        on_self_heal: Callable[[str, str, bool], None] | None = None,
        self_heal_enabled: bool = True,
    ) -> None:
        self.config = config
        self.state = AgentState()
        self.client = OllamaClient(config)
        self.error_tracker = ErrorTracker()
        self._self_heal_enabled = self_heal_enabled

        # Callbacks for TUI/logging
        self._on_message = on_message
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._on_phase_change = on_phase_change
        self._on_self_heal = on_self_heal

    def initialize(self, system_prompt: str | None = None) -> None:
        """Set up the agent with its system prompt."""
        prompt = system_prompt or SYSTEM_PROMPT
        self.state.messages.append(Message(role=Role.SYSTEM, content=prompt))
        logger.info("Agent initialized with %d-char system prompt", len(prompt))

    def inject_context(self, phase_prompt: str) -> None:
        """Inject a phase-specific prompt as a user message.

        This is how the pipeline feeds instructions to the agent for each
        phase (extract, analyze, prepare, emulate, fix, verify).
        """
        old_phase = self.state.phase
        self.state.phase = phase_prompt.split("PHASE:")[1].split("\n")[0].strip() if "PHASE:" in phase_prompt else "unknown"

        if self._on_phase_change and old_phase != self.state.phase:
            self._on_phase_change(old_phase, self.state.phase)

        # Reset error tracker on phase change for fresh tracking
        if old_phase != self.state.phase:
            self.error_tracker.reset()

        msg = Message(role=Role.USER, content=phase_prompt)
        self.state.messages.append(msg)
        if self._on_message:
            self._on_message(msg)

    def run_until_done(self, max_iterations: int | None = None) -> str:
        """Execute the agent loop until the LLM stops calling tools.

        Returns the final assistant message (summary/status).
        """
        limit = max_iterations or self.config.max_iterations
        tool_schemas = get_tool_schemas()

        while self.state.iteration < limit:
            self.state.iteration += 1
            logger.info(
                "Iteration %d/%d (phase=%s)",
                self.state.iteration, limit, self.state.phase,
            )

            # --- LLM inference ---
            messages_payload = self._build_messages_payload()
            response = self.client.chat(messages_payload, tools=tool_schemas or None)

            content, tool_calls = parse_tool_calls(response)

            # Record assistant message
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )
            self.state.messages.append(assistant_msg)
            if self._on_message:
                self._on_message(assistant_msg)

            # --- No tool calls = agent is done (or stuck) ---
            if not tool_calls:
                logger.info("Agent completed phase (no more tool calls)")
                return content

            # --- Execute each tool call ---
            for tc in tool_calls:
                if self._on_tool_call:
                    self._on_tool_call(tc)

                result = self._execute_tool(tc)
                self.state.total_commands_run += 1

                if self._on_tool_result:
                    self._on_tool_result(result)

                # --- Self-Healing Layer 2: track errors ---
                if result.success:
                    self.error_tracker.record_success()
                else:
                    # Extract the command from tool call args
                    cmd_str = tc.arguments.get("cmd", tc.arguments.get("command", tc.name))
                    self.error_tracker.record_error(
                        iteration=self.state.iteration,
                        command=str(cmd_str),
                        error_text=result.output,
                    )

                # Feed result back to LLM
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.output,
                    tool_call_id=result.tool_call_id,
                )
                self.state.messages.append(tool_msg)

            # --- Self-Healing: inject corrective prompt if stuck ---
            if self._self_heal_enabled:
                stuck_type = self.error_tracker.detect_stuck()
                if stuck_type:
                    self._inject_corrective_prompt()

            # --- Context window management ---
            self._trim_context_if_needed()

        logger.warning("Agent hit iteration limit (%d)", limit)
        return f"[IoTGhost] Reached maximum iterations ({limit}). Last phase: {self.state.phase}"

    def _inject_corrective_prompt(self) -> None:
        """Inject a corrective prompt based on escalation tier.

        Tier 1 (1-3 failures): Targeted hint based on dominant error category
        Tier 2 (4-6 failures): Full SELF_HEAL_PROMPT with diagnostic context
        Tier 3 (7+ failures):  Force phase skip -- this issue cannot be resolved
        """
        failures = self.error_tracker.consecutive_failures
        stuck_type = self.error_tracker.detect_stuck()

        if failures >= 7 or (stuck_type and self.error_tracker._corrective_injections >= 3):
            # Tier 3: Force skip
            skip_msg = (
                "[SYSTEM] CRITICAL: This issue cannot be resolved with current approach. "
                "The agent has failed {} consecutive times (stuck pattern: {}). "
                "STOP trying to fix this specific issue. Instead:\n"
                "1. If in FIX phase: accept partial failure and proceed to VERIFY\n"
                "2. If in PREPARE phase: skip the failing step and continue\n"
                "3. If in EMULATE phase: try degraded mode with init=/bin/sh\n"
                "DO NOT repeat any previous commands."
            ).format(failures, stuck_type or "exhaustion")
            msg = Message(role=Role.USER, content=skip_msg)
            self.state.messages.append(msg)
            if self._on_message:
                self._on_message(msg)
            if self._on_self_heal:
                self._on_self_heal(self.state.phase, stuck_type or "exhaustion", True)
            self.error_tracker.consecutive_failures = max(0, failures - 2)
            self.error_tracker._corrective_injections += 1
            logger.warning("Tier 3 escalation: forcing phase skip after %d failures", failures)
            return

        if failures >= 4:
            # Tier 2: Full SELF_HEAL_PROMPT
            diag_context = self.error_tracker.build_diagnostic_context()
            heal_msg = SELF_HEAL_PROMPT.format(
                diagnostic_context=diag_context,
            )
            # Append expert prompt if we have a dominant category
            expert = self.error_tracker.get_expert_prompt()
            if expert:
                heal_msg += "\n\n## EXPERT RECOVERY GUIDANCE\n" + expert

            msg = Message(role=Role.USER, content=heal_msg)
            self.state.messages.append(msg)
            if self._on_message:
                self._on_message(msg)
            if self._on_self_heal:
                self._on_self_heal(self.state.phase, stuck_type or "stalled", True)
            self.error_tracker.consecutive_failures = max(0, failures - 2)
            self.error_tracker._corrective_injections += 1
            logger.warning("Tier 2 escalation: SELF_HEAL_PROMPT after %d failures", failures)
            return

        # Tier 1 (1-3 failures): Targeted hint
        expert = self.error_tracker.get_expert_prompt()
        if expert:
            hint_msg = (
                "[SYSTEM] The last {} commands failed. Error pattern suggests: {}\n\n"
                "Targeted guidance:\n{}"
            ).format(
                failures,
                self.error_tracker.dominant_category() or "unknown",
                expert[:800],  # Truncate to keep it focused
            )
        else:
            # Generic hint
            recent_errors = list(self.error_tracker.error_history)[-3:]
            error_summary = "\n".join(
                f"  - [{e.category}] {e.error_text[:100]}" for e in recent_errors
            )
            hint_msg = (
                "[SYSTEM] The last {} commands failed with these errors:\n{}\n\n"
                "Try a DIFFERENT approach. Do NOT repeat the same commands. "
                "Consider: different paths, different tools, or skip this step."
            ).format(failures, error_summary)

        msg = Message(role=Role.USER, content=hint_msg)
        self.state.messages.append(msg)
        if self._on_message:
            self._on_message(msg)
        if self._on_self_heal:
            self._on_self_heal(self.state.phase, stuck_type or "hint", True)
        self.error_tracker.consecutive_failures = max(0, failures - 2)
        self.error_tracker._corrective_injections += 1
        logger.info("Tier 1 hint after %d failures", failures)

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and return the result."""
        func = get_tool(tool_call.name)
        if not func:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=f"Error: Unknown tool '{tool_call.name}'. Available: {list_tools()}",
                success=False,
            )

        start = time.time()
        try:
            result = func(**tool_call.arguments)
            result.tool_call_id = tool_call.id
            result.duration = time.time() - start
            return result
        except TypeError as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=f"Error: Bad arguments for '{tool_call.name}': {exc}",
                success=False,
                duration=time.time() - start,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=f"Error executing '{tool_call.name}': {type(exc).__name__}: {exc}",
                success=False,
                duration=time.time() - start,
            )

    def _build_messages_payload(self) -> list[dict[str, Any]]:
        """Convert internal messages to Ollama API format."""
        payload = []
        for msg in self.state.messages:
            entry: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            # For tool results, Ollama expects them as 'tool' role messages
            if msg.role == Role.TOOL:
                entry["role"] = "tool"
            payload.append(entry)
        return payload

    def _trim_context_if_needed(self) -> None:
        """Trim older messages if approaching context window limit.

        Keeps system prompt and recent messages. Summarizes dropped
        messages to preserve critical context.
        """
        # Rough token estimate: 4 chars per token
        total_chars = sum(len(m.content) for m in self.state.messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens < self.config.context_window * 0.8:
            return

        logger.info("Trimming context: ~%d tokens (limit: %d)", estimated_tokens, self.config.context_window)

        # Keep system prompt (index 0) and last N messages
        keep_recent = 20
        if len(self.state.messages) <= keep_recent + 1:
            return

        system_msg = self.state.messages[0]
        dropped = self.state.messages[1:-keep_recent]
        kept = self.state.messages[-keep_recent:]

        # Create summary of dropped context
        summary_parts = []
        for msg in dropped:
            if msg.role == Role.ASSISTANT and msg.content:
                # Keep assistant reasoning summaries
                first_line = msg.content.split("\n")[0][:200]
                summary_parts.append(f"[{msg.role.value}] {first_line}")
            elif msg.role == Role.TOOL and "Error" in msg.content:
                # Keep error messages
                summary_parts.append(f"[tool_error] {msg.content[:200]}")

        summary = (
            "[Context trimmed. Summary of earlier work:\n"
            + "\n".join(summary_parts[-10:])  # Keep last 10 summaries
            + "\n]"
        )

        self.state.messages = [
            system_msg,
            Message(role=Role.USER, content=summary),
            *kept,
        ]
        logger.info("Context trimmed: dropped %d messages, kept %d", len(dropped), len(kept))

    def get_context_var(self, key: str) -> str | None:
        """Retrieve a context variable set during execution."""
        return self.state.context_vars.get(key)

    def set_context_var(self, key: str, value: str) -> None:
        """Set a context variable for cross-phase communication."""
        self.state.context_vars[key] = value
        logger.debug("Context var set: %s = %s", key, value[:100])

    def get_stats(self) -> dict[str, Any]:
        """Return execution statistics."""
        return {
            "iterations": self.state.iteration,
            "commands_run": self.state.total_commands_run,
            "phase": self.state.phase,
            "elapsed_seconds": round(self.state.elapsed, 1),
            "messages": len(self.state.messages),
            "context_vars": dict(self.state.context_vars),
            "errors_seen": len(self.state.errors_seen),
            "fixes_attempted": len(self.state.attempted_fixes),
            # Self-healing stats
            "self_heal_injections": self.error_tracker._corrective_injections,
            "tracked_failures": self.error_tracker.total_failures,
            "errors_by_category": dict(self.error_tracker.errors_by_category),
        }

    def shutdown(self) -> None:
        """Clean up resources."""
        self.client.close()
