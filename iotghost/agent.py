"""LLM shell agent core -- autonomous tool-use loop with Ollama integration.

The agent receives a system prompt with deep IoT emulation knowledge,
then iteratively executes shell commands, reads their output, and decides
the next action. It accumulates context across steps and implements
retry logic with backoff for failed approaches.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import httpx

from iotghost.prompts import SYSTEM_PROMPT

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
    ) -> None:
        self.config = config
        self.state = AgentState()
        self.client = OllamaClient(config)

        # Callbacks for TUI/logging
        self._on_message = on_message
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._on_phase_change = on_phase_change

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

                # Feed result back to LLM
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.output,
                    tool_call_id=result.tool_call_id,
                )
                self.state.messages.append(tool_msg)

            # --- Context window management ---
            self._trim_context_if_needed()

        logger.warning("Agent hit iteration limit (%d)", limit)
        return f"[IoTGhost] Reached maximum iterations ({limit}). Last phase: {self.state.phase}"

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
        }

    def shutdown(self) -> None:
        """Clean up resources."""
        self.client.close()
