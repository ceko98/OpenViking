# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""
Claude Code VLM Backend

Runs completions through the local ``claude -p`` CLI, using the user's own Claude Code
login. Like the Codex backend it subclasses OpenAIVLM and swaps the HTTP client for an
adapter that returns OpenAI chat-completion shaped objects.

Every spawned call is isolated from the user's Claude Code setup (``--safe-mode``, no
tools, no MCP, no slash commands, no session persistence, plugin kill-switch env vars,
empty cwd) so it cannot trigger the memory plugin's hooks and recurse into OpenViking.
``--bare`` is deliberately not used: it disables OAuth/keychain auth.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from openviking.models.model_error import ModelError, ModelErrorKind
from openviking_cli.utils import get_logger

from .openai_vlm import OpenAIVLM

logger = get_logger(__name__)

DEFAULT_MODEL = "haiku"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_TIMEOUT = 180.0
MAX_CONCURRENT_ENV = "OPENVIKING_CLAUDE_CODE_MAX_CONCURRENT"

# Same kill-switches as examples/claude-code-memory-plugin/scripts/lib/host-compressor.mjs
_GUARD_ENV = {
    "OPENVIKING_MEMORY_ENABLED": "0",
    "OPENVIKING_AUTO_RECALL": "0",
    "OPENVIKING_AUTO_CAPTURE": "0",
    "OPENVIKING_NO_AUTO_INJECT": "1",
    "OPENVIKING_RECALL_COMPRESS": "off",
    "OPENVIKING_RECALL_REWRITE": "off",
}

# ponytail: one process-wide semaphore sized by the first provider built; resize needs a restart.
_semaphore: Optional[threading.BoundedSemaphore] = None
_semaphore_lock = threading.Lock()


def _get_semaphore(limit: int) -> threading.BoundedSemaphore:
    global _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = threading.BoundedSemaphore(max(1, limit))
        return _semaphore


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI message content to Anthropic content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: List[Dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                header, data = url.split(",", 1)
                media_type = header[len("data:") :].split(";", 1)[0]
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
            else:
                raise ModelError(
                    ModelErrorKind.INVALID_REQUEST,
                    "claude-code provider only supports base64 data URL images",
                    provider="claude-code",
                )
        elif part.get("type") == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
    return blocks


def build_request(messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Return (system_prompt, stream-json user message) for an OpenAI messages list."""
    system_parts: List[str] = []
    rest = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            blocks = _content_blocks(msg.get("content"))
            system_parts.extend(b["text"] for b in blocks if b["type"] == "text")
        else:
            rest.append(msg)

    blocks: List[Dict[str, Any]] = []
    single_user = len(rest) == 1 and rest[0].get("role") == "user"
    for msg in rest:
        role = msg.get("role", "user")
        msg_blocks = _content_blocks(msg.get("content"))
        if not single_user and msg_blocks:
            msg_blocks = [{"type": "text", "text": f"[{role}]"}] + msg_blocks
        blocks.extend(msg_blocks)
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    user_message = {"type": "user", "message": {"role": "user", "content": blocks}}
    return "\n\n".join(system_parts), user_message


def _rate_limited(events: List[Dict[str, Any]], result: Dict[str, Any]) -> bool:
    if result.get("api_error_status") == 429:
        return True
    return any(
        e.get("type") == "rate_limit_event"
        and (e.get("rate_limit_info") or {}).get("status") == "rejected"
        for e in events
    )


def parse_output(stdout: str) -> SimpleNamespace:
    """Parse stream-json stdout into a chat-completion shaped object."""
    events: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None:
        raise ModelError(
            ModelErrorKind.TRANSIENT_SERVICE_ERROR,
            "claude -p produced no result event",
            provider="claude-code",
        )

    text = result.get("result") or ""
    if result.get("is_error") or result.get("subtype", "success") != "success":
        if _rate_limited(events, result):
            kind = ModelErrorKind.RATE_LIMIT_ERROR
        elif result.get("api_error_status") in (401, 403):
            kind = ModelErrorKind.AUTHENTICATION_ERROR
        else:
            kind = ModelErrorKind.TRANSIENT_SERVICE_ERROR
        raise ModelError(
            kind,
            f"claude -p failed: {text or result.get('subtype')}",
            provider="claude-code",
            status_code=result.get("api_error_status"),
        )

    usage = result.get("usage") or {}
    prompt_tokens = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    )
    completion_tokens = usage.get("output_tokens") or 0
    finish_reason = "length" if result.get("stop_reason") == "max_tokens" else "stop"
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=usage.get("cache_read_input_tokens") or 0
            ),
        ),
    )


class ClaudeCodeCompletionsAdapter:
    def __init__(
        self,
        command: str,
        model: str,
        effort: Optional[str],
        timeout: float,
        max_concurrent: int,
    ):
        self.command = command
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self._semaphore = _get_semaphore(max_concurrent)
        self._warned_tools = False
        self._cwd: Optional[str] = None

    def _empty_cwd(self) -> str:
        if self._cwd is None or not os.path.isdir(self._cwd):
            self._cwd = tempfile.mkdtemp(prefix="ov-claude-code-")
        return self._cwd

    def _build_args(self, system_prompt: str, model: Optional[str]) -> List[str]:
        args = [
            self.command,
            "-p",
            "--safe-mode",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--no-session-persistence",
            "--model",
            model or self.model,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if self.effort:
            args += ["--effort", self.effort]
        if system_prompt:
            args += ["--system-prompt", system_prompt]
        return args

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if kwargs.get("stream"):
            raise NotImplementedError("Streaming is not supported for claude-code.")
        if kwargs.get("tools") and not self._warned_tools:
            # ponytail: no tool emulation; add a JSON tool-call protocol if tool loops are needed.
            logger.warning("claude-code provider ignores tool definitions")
            self._warned_tools = True
        system_prompt, user_message = build_request(kwargs.get("messages") or [])
        args = self._build_args(system_prompt, kwargs.get("model"))
        env = {**os.environ, **_GUARD_ENV}
        payload = json.dumps(user_message) + "\n"
        with self._semaphore:
            try:
                proc = subprocess.run(
                    args,
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    cwd=self._empty_cwd(),
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelError(
                    ModelErrorKind.TRANSIENT_NETWORK_ERROR,
                    f"claude -p timed out after {self.timeout}s",
                    provider="claude-code",
                ) from exc
            except FileNotFoundError as exc:
                raise ModelError(
                    ModelErrorKind.UNSUPPORTED,
                    f"claude CLI not found: {self.command}",
                    provider="claude-code",
                ) from exc
        if not proc.stdout.strip():
            raise ModelError(
                ModelErrorKind.TRANSIENT_SERVICE_ERROR,
                f"claude -p exited {proc.returncode} with no output: {proc.stderr.strip()[:500]}",
                provider="claude-code",
            )
        return parse_output(proc.stdout)


class ClaudeCodeAsyncCompletionsAdapter:
    def __init__(self, sync_adapter: ClaudeCodeCompletionsAdapter):
        self._sync_adapter = sync_adapter

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        return await asyncio.to_thread(self._sync_adapter.create, **kwargs)


class ClaudeCodeVLM(OpenAIVLM):
    def __init__(self, config: Dict[str, Any]):
        normalized = dict(config)
        normalized["provider"] = "claude-code"
        normalized.setdefault("model", DEFAULT_MODEL)
        super().__init__(normalized)
        self.model = self.model or DEFAULT_MODEL
        self.reasoning_effort = config.get("reasoning_effort") or DEFAULT_EFFORT
        self._command = config.get("command") or "claude"
        self._adapter = ClaudeCodeCompletionsAdapter(
            self._command,
            self.model,
            self.reasoning_effort,
            self.timeout if config.get("timeout") else DEFAULT_TIMEOUT,
            self._max_concurrent(),
        )
        self._client = SimpleNamespace(chat=SimpleNamespace(completions=self._adapter))
        self._async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=ClaudeCodeAsyncCompletionsAdapter(self._adapter))
        )

    def _max_concurrent(self) -> int:
        # VLMConfig does not forward max_concurrent; also accept extra_request_body and env.
        value = (
            self.config.get("max_concurrent")
            or self.extra_request_body.get("max_concurrent")
            or os.environ.get(MAX_CONCURRENT_ENV)
            or DEFAULT_MAX_CONCURRENT
        )
        return int(value)

    def get_client(self):
        return self._client

    def get_async_client(self):
        return self._async_client

    def _apply_completion_params(self, kwargs, thinking, max_tokens=None) -> None:
        # claude -p takes no sampling/token params; keep the request kwargs minimal.
        return

    def is_available(self) -> bool:
        return shutil.which(self._command) is not None

    def close(self) -> None:
        return
