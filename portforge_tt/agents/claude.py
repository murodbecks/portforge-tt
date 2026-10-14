"""Claude Code CLI agent backend.

Ported from ``chia.models.claude`` (Copyright (c) 2026 The Regents of the
University of California; BSD-3-Clause, see LICENSE): the CLI (``claude --print``) execution path, the
stream-json event parser and the rate-limit parsers. Removed with the CHIA
layer: the Ray dispatch/decorators, the cross-worker session-transcript
carry (jobs now run on local threads in one process), and the experimental
Anthropic-API backend. Error types come from the shared :mod:`errors`
taxonomy. ``resume_session=True`` threads one conversation across calls
(``--session-id`` on the first run, ``--resume`` afterwards).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
from uuid import uuid4

from .errors import (RateLimitError, AuthenticationError, BillingError,
                     InvalidRequestError, ServerError, MaxOutputTokensError,
                     UnknownProviderError)
from .llm import QueryResult, LLMCallBase


@dataclass
class ClaudeCodeQueryResult(QueryResult):
    """``QueryResult`` specialised for the Claude Code CLI backend."""


# ``CLIResult`` is kept as a generic alias for any LLM result (back-compat
# with the CHIA-era module).
CLIResult = QueryResult


# ---------------------------------------------------------------------------
# Rate-limit text parser
# ---------------------------------------------------------------------------

_RATE_LIMIT_RE = re.compile(
    r"You've hit your limit\s*[·•\-—]\s*resets?\s+(\d{1,2})\s*(am|pm)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def parse_rate_limit_reset(text: str) -> Optional[datetime]:
    """Parse a Claude rate-limit message and return the UTC reset time.

    Expected format: ``"You've hit your limit · resets 4pm (America/Los_Angeles)"``

    Returns ``None`` when no rate-limit message is found.
    """
    m = _RATE_LIMIT_RE.search(text)
    if m is None:
        return None

    hour = int(m.group(1))
    ampm = m.group(2).lower()
    tz_str = m.group(3).strip()

    # Convert 12-hour → 24-hour
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    # Resolve timezone
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        # Fallback: common abbreviations
        _abbrev = {
            "PST": -8, "PDT": -7, "MST": -7, "MDT": -6,
            "CST": -6, "CDT": -5, "EST": -5, "EDT": -4,
            "UTC": 0, "GMT": 0,
        }
        offset_hours = _abbrev.get(tz_str.upper(), 0)
        tz = timezone(timedelta(hours=offset_hours))

    now_in_tz = datetime.now(tz)
    reset_local = now_in_tz.replace(hour=hour, minute=0, second=0, microsecond=0)

    # If reset hour is in the past, it means tomorrow
    if reset_local <= now_in_tz:
        reset_local += timedelta(days=1)

    return reset_local.astimezone(timezone.utc)


def parse_rate_limit_event(event: dict) -> Optional[datetime]:
    """Parse a ``rate_limit_event`` JSON object and return the UTC reset time.

    Only triggers when ``rate_limit_info.status`` is ``"rejected"`` — the event
    is also emitted with other statuses as an informational notice, which should
    NOT be treated as a rate limit.
    """
    info = event.get("rate_limit_info", {})
    if info.get("status") != "rejected":
        return None
    resets_at = info.get("resetsAt")
    if resets_at is None:
        return None
    try:
        return datetime.fromtimestamp(int(resets_at), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def _tool_url(tool: Any) -> str:
    return getattr(tool, "url", None) or \
        f"http://{tool.hostname}:{tool.port}/{tool.name}/mcp"


class ClaudeCodeLLM(LLMCallBase):
    """Wraps the Claude Code CLI (``claude --print``) as an agent backend.

    Each call to :meth:`prompt` spawns a ``claude`` subprocess that can
    connect to MCP tool servers. Honors ``--dangerously-skip-permissions``;
    claude's richer permission model lives in settings JSON passed through
    ``extra_cli_args`` (see ``agent.IsolatedClaude``).
    """

    supports_dangerously_skip_permissions = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        system_message: str = "",
        timeout_seconds: int = 600,
        retries: int = 3,
        logging_name: str = "claude_code",
        logging_level: int = logging.DEBUG,
        log_dir: Optional[str] = None,
        resume_session: bool = False,
        extra_cli_args: Optional[List[str]] = None,
        log_stream: bool = True,
        log_all: bool = False,
        dangerously_skip_permissions: bool = True,
    ):
        super().__init__(system_message=system_message,
                         dangerously_skip_permissions=dangerously_skip_permissions)
        self.logging_level = logging_level
        self.logging_name = logging_name
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.extra_cli_args = extra_cli_args or []
        self.logger = logging.getLogger(logging_name)
        self.log_stream = log_stream
        self.log_all = log_all

        self._call_counter = 0
        self._session_id = str(uuid4()) if resume_session else None
        self._last_metadata: dict = {}  # populated by _process_event_line
        self._rate_limit_event: Optional[dict] = None  # populated by _process_event_line

        self._log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            session_tag = f"_{self._session_id[:8]}" if self._session_id else ""
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_prefix = os.path.join(log_dir, f"{logging_name}_{run_id}{session_tag}")
        else:
            self._log_prefix = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prompt(
        self,
        user_message: str,
        tools: Optional[List[Any]] = [],
    ) -> ClaudeCodeQueryResult:
        """Send *user_message* to Claude Code CLI and return the response.

        Returns:
            :class:`ClaudeCodeQueryResult` with ``success=True`` when the CLI
            ran cleanly, or ``success=False`` when every retry attempt failed
            (in which case ``result`` is empty and ``returncode`` is ``-1``).

        Raises:
            RateLimitError: Usage limit hit — propagates immediately.
            AuthenticationError: Auth failure — propagates immediately.
            BillingError: Billing/payment issue — propagates immediately.
            InvalidRequestError: Malformed request — propagates immediately.
            ServerError: After all retries with exponential backoff.
            MaxOutputTokensError: After one retry attempt.
        """
        import time as _time

        for attempt in range(self.retries):
            try:
                self._last_metadata = {}
                self._rate_limit_event = None
                if self.log_stream:
                    cli = self._run_claude_streaming(user_message, tools)
                else:
                    cli = self._run_claude(user_message, tools)
                self._call_counter += 1
                self._last_metadata["model"] = self.model
                self._last_metadata["tools"] = [
                    {"name": t.name, "url": _tool_url(t)} for t in tools
                ]

                # Classify and raise typed errors
                self._classify_error(cli)

                cli.success = True
                return cli

            # -- Never retry: propagate immediately --
            except (RateLimitError, AuthenticationError, BillingError, InvalidRequestError):
                raise

            # -- Retry once: stochastic generation may produce shorter output --
            except MaxOutputTokensError:
                if attempt == 0:
                    self.logger.warning(
                        "Max output tokens on attempt %d/%d, retrying once",
                        attempt + 1, self.retries,
                    )
                    continue
                raise

            # -- Retry with exponential backoff: transient API issue --
            except ServerError:
                backoff = min(5 * 2 ** attempt, 60)
                self.logger.warning(
                    "Server error on attempt %d/%d, backing off %ds",
                    attempt + 1, self.retries, backoff,
                )
                _time.sleep(backoff)

            # -- Standard retry for unknown errors --
            except UnknownProviderError as exc:
                self.logger.warning(
                    "Unknown error on attempt %d/%d: %s",
                    attempt + 1, self.retries, exc,
                )

            except subprocess.TimeoutExpired:
                # A timeout means the session was likely created;
                # switch to --resume for subsequent attempts.
                if self._session_id is not None and self._call_counter == 0:
                    self._call_counter = 1
                self.logger.warning(
                    "Timeout on attempt %d/%d", attempt + 1, self.retries,
                )

            except Exception as exc:
                self.logger.warning(
                    "Unexpected error on attempt %d/%d: %s",
                    attempt + 1, self.retries, exc,
                )
        return ClaudeCodeQueryResult(result="", returncode=-1, stderr="", stream_result="", success=False)

    def _classify_error(self, cli: ClaudeCodeQueryResult) -> None:
        """Inspect *cli* and raise a typed error if something went wrong.

        Check order:
        1. Rate limit (text regex + streaming event) — highest priority
        2. Non-zero exit code — classify by stderr patterns
        3. Success — return without raising
        """
        # --- 1. Rate limit (can appear even with exit code 0) ---
        reset_time = parse_rate_limit_reset(cli.result)
        if reset_time is None and self._rate_limit_event is not None:
            reset_time = parse_rate_limit_event(self._rate_limit_event)

        if reset_time is not None:
            self.logger.warning(
                "Rate limit detected (resets %s). Response text:\n%s",
                reset_time.isoformat(), cli.result,
            )
            raise RateLimitError(
                reset_time=reset_time,
                raw_message=cli.result[:300],
                exit_code=cli.returncode,
            )

        # --- 2. Non-zero exit code → classify by stderr ---
        if cli.returncode != 0:
            stderr_lower = cli.stderr.lower()

            if any(kw in stderr_lower for kw in (
                "authentication", "unauthorized", "401", "not authenticated",
                "login", "auth token",
            )):
                raise AuthenticationError(cli.returncode, cli.stderr[:300])

            if any(kw in stderr_lower for kw in (
                "billing", "payment", "402", "overdue", "subscription",
                "plan expired",
            )):
                raise BillingError(cli.returncode, cli.stderr[:300])

            if any(kw in stderr_lower for kw in (
                "invalid request", "malformed", "400", "bad request",
                "invalid model",
            )):
                raise InvalidRequestError(cli.returncode, cli.stderr[:300])

            if any(kw in stderr_lower for kw in (
                "500", "503", "server error", "overloaded",
                "internal error", "service unavailable",
            )):
                raise ServerError(cli.returncode, cli.stderr[:300])

            if any(kw in stderr_lower for kw in (
                "max_output_tokens", "output token limit", "maximum output",
                "response too long",
            )):
                raise MaxOutputTokensError(
                    cli.returncode, cli.stderr[:300], partial_text=cli.result,
                )

            # Fallback: unknown error
            raise UnknownProviderError(
                cli.returncode, cli.stderr[:300], stderr=cli.stderr,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_mcp_config(self, tools: List[Any]) -> dict:
        """Build the JSON object expected by ``--mcp-config``."""
        servers = {}
        for tool in tools:
            servers[tool.name] = {
                "type": "http",
                "url": _tool_url(tool),
            }
        return {"mcpServers": servers}

    def _build_allowed_tools(self, tools: List[Any]) -> List[str]:
        """Return ``--allowedTools`` entries for every registered MCP tool."""
        allowed: list[str] = []
        for tool in tools:
            # FastMCP registers tools under server_name; the MCP tool ID
            # that Claude Code recognises is  mcp__<server>__<tool_name>.
            for fn_info in tool.mcp._tool_manager.list_tools():
                allowed.append(f"mcp__{tool.name}__{fn_info.name}")
        return allowed

    def _build_cmd(self, tools: Optional[List[Any]] = None) -> list:
        """Build the ``claude`` CLI command list.

        The user message is piped via stdin (``-p -``) to avoid OS
        argument-length limits with long prompts.
        """
        cmd = [
            "claude",
            "--print",
            "--model", self.model,
        ]
        if self.dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        if self.extra_cli_args:
            cmd += self.extra_cli_args

        if self.log_stream:
            cmd += ["--output-format", "stream-json", "--verbose"]

        if self.system_message:
            cmd += ["--system-prompt", self.system_message]

        if self._session_id is not None:
            if self._call_counter > 0:
                cmd += ["--resume", self._session_id]
            else:
                cmd += ["--session-id", self._session_id]

        if tools:
            cfg = self._build_mcp_config(tools)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(cfg, tmp)
            tmp.close()
            cmd += ["--mcp-config", tmp.name]

            allowed = self._build_allowed_tools(tools)
            if allowed:
                cmd += ["--allowedTools", ",".join(allowed)]

        cmd += ["-p", "-"]
        return cmd

    def _run_claude(
        self,
        user_message: str,
        tools: Optional[List[Any]] = None,
    ) -> ClaudeCodeQueryResult:
        """Run claude with simple capture (no event streaming)."""
        cmd = self._build_cmd(tools)
        self.logger.info("Running: %s", " ".join(cmd[:6]) + " ...")
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        result = subprocess.run(
            cmd,
            input=user_message,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )

        if result.returncode != 0:
            self.logger.warning("claude exited %d: %s", result.returncode, result.stderr[:500])

        if self._log_prefix is not None:
            truncated = user_message[:500] + ("..." if len(user_message) > 500 else "")
            with open(f"{self._log_prefix}.log", "a") as f:
                f.write("=" * 80 + "\n")
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prompt #{self._call_counter}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"[User Message]\n{truncated}\n\n")
                f.write(f"[Response]\n{result.stdout}\n\n")
                f.write("-" * 80 + "\n\n")

        return ClaudeCodeQueryResult(
            result=result.stdout,
            returncode=result.returncode,
            stderr=result.stderr,
            stream_result="",
        )

    # ------------------------------------------------------------------
    # Streaming log implementation
    # ------------------------------------------------------------------

    def _run_claude_streaming(
        self,
        user_message: str,
        tools: Optional[List[Any]] = None,
    ) -> ClaudeCodeQueryResult:
        """Run claude with ``--output-format stream-json``.

        Stdout (NDJSON events) is parsed by ``_process_event_line`` into
        structured entries.  Stderr lines are captured with a ``[stderr]``
        prefix.  A lock serialises writes from the two drain threads.

        Every parsed entry is appended to an in-memory ``stream_result``
        buffer (returned on ``ClaudeCodeQueryResult``) so callers can surface
        the event trace — tool calls, thinking, metadata — directly.  When
        ``log_dir`` was set on the constructor, the same entries are also
        mirrored to ``<prefix>.log`` on disk.
        """
        cmd = self._build_cmd(tools)
        self.logger.info("Running: %s", " ".join(cmd[:6]) + " ...")
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        result_text_parts: list[str] = []
        stderr_parts: list[str] = []
        stream_parts: list[str] = []
        lock = threading.Lock()

        class _TeeWriter:
            """File-like wrapper that always mirrors to *accumulator* and
            optionally writes through to *file* when one is provided."""
            def __init__(self, file, accumulator: list[str]):
                self._file = file
                self._accumulator = accumulator
            def write(self, s: str):
                if self._file is not None:
                    self._file.write(s)
                self._accumulator.append(s)
            def flush(self):
                if self._file is not None:
                    self._file.flush()
            def close(self):
                if self._file is not None:
                    self._file.close()

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        proc.stdin.write(user_message)
        proc.stdin.close()

        # Only mirror the live event stream to disk when log_all is set.
        # Without log_all we still accumulate stream_parts in memory but
        # write a compact result-only entry to the log after the run.
        file_handle = (
            open(f"{self._log_prefix}.log", "a")
            if self._log_prefix is not None and self.log_all else None
        )
        log_file = _TeeWriter(file_handle, stream_parts)

        log_file.write("=" * 80 + "\n")
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prompt #{self._call_counter}\n")
        log_file.write("=" * 80 + "\n\n")
        truncated = user_message[:500] + ("..." if len(user_message) > 500 else "")
        log_file.write(f"[User Message]\n{truncated}\n\n")
        log_file.flush()

        def drain_stdout():
            for line in proc.stdout:
                line = line.strip()
                if line:
                    with lock:
                        self._process_event_line(line, log_file, result_text_parts)
                        log_file.flush()

        def drain_stderr():
            for line in proc.stderr:
                with lock:
                    stderr_parts.append(line)
                    log_file.write(f"[stderr] {line}")
                    log_file.flush()

        t1 = threading.Thread(target=drain_stderr)
        t2 = threading.Thread(target=drain_stdout)
        t1.start()
        t2.start()

        proc.wait()
        t1.join()
        t2.join()

        if not result_text_parts:
            log_file.write("[DEBUG] No events parsed.\n")

        log_file.write("-" * 80 + "\n\n")
        log_file.flush()
        log_file.close()

        if self._log_prefix is not None and not self.log_all:
            with open(f"{self._log_prefix}.log", "a") as f:
                f.write("=" * 80 + "\n")
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prompt #{self._call_counter}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"[User Message]\n{truncated}\n\n")
                f.write(f"[Response]\n{''.join(result_text_parts)}\n\n")
                f.write("-" * 80 + "\n\n")

        if proc.returncode != 0:
            self.logger.warning("claude exited %d", proc.returncode)

        return ClaudeCodeQueryResult(
            result="".join(result_text_parts),
            returncode=proc.returncode,
            stderr="".join(stderr_parts),
            stream_result="".join(stream_parts),
        )

    def _process_event_line(self, line: str, f, result_text_parts: list) -> None:
        """Parse a single NDJSON event line and write to the log file."""
        if not line:
            return

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            f.write(f"[UNPARSED] {line[:200]}\n")
            f.flush()
            return

        event_type = event.get("type", "")

        if event_type == "assistant":
            msg = event.get("message", {})
            for block in msg.get("content", []):
                block_type = block.get("type", "")
                if block_type == "thinking":
                    f.write("[Thinking]\n")
                    f.write(block.get("thinking", ""))
                    f.write("\n\n")
                    f.flush()
                elif block_type == "text":
                    text = block.get("text", "")
                    result_text_parts.append(text)
                    f.write("[Response]\n")
                    f.write(text)
                    f.write("\n\n")
                    f.flush()
                elif block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = json.dumps(block.get("input", {}))
                    if len(tool_input) > 2000:
                        tool_input = tool_input[:2000] + "\n... [truncated]"
                    f.write(f"[Tool Call: {tool_name}]\n")
                    f.write(f"Args: {tool_input}\n\n")
                    f.flush()
                elif block_type == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict)
                        )
                    if len(content) > 2000:
                        content = content[:2000] + "\n... [truncated]"
                    f.write(f"[Tool Result]\n{content}\n\n")
                    f.flush()

        elif event_type == "user":
            # Tool results ride on user events — the CLI echoes each tool
            # reply back to the assistant as a user-turn ``tool_result``
            # block.  Capturing them here is the only way to see what the
            # assistant actually saw when it chose its next action.
            msg = event.get("message", {})
            content_blocks = msg.get("content", [])
            if isinstance(content_blocks, str):
                # Plain-text user message (initial prompt echo); skip.
                return
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict)
                    )
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"
                label = "Tool Result (error)" if block.get("is_error") else "Tool Result"
                f.write(f"[{label}]\n{content}\n\n")
                f.flush()

        elif event_type == "result":
            result_text = event.get("result", "")
            if result_text and not result_text_parts:
                result_text_parts.append(result_text)

            parts = []
            meta: dict = {}
            cost = event.get("total_cost_usd")
            if cost is not None:
                parts.append(f"Cost: ${cost:.4f}")
                meta["cost_usd"] = cost
            duration = event.get("duration_ms")
            if duration is not None:
                parts.append(f"Duration: {duration / 1000:.1f}s")
                meta["duration_s"] = round(duration / 1000, 2)
            turns = event.get("num_turns")
            if turns is not None:
                parts.append(f"Turns: {turns}")
                meta["num_turns"] = turns
            usage = event.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            cc_tok = usage.get("cache_creation_input_tokens", 0)
            cr_tok = usage.get("cache_read_input_tokens", 0)
            if in_tok:
                parts.append(f"Input tokens: {in_tok}")
                meta["input_tokens"] = in_tok
            if cc_tok:
                parts.append(f"Cache creation: {cc_tok}")
                meta["cache_creation_input_tokens"] = cc_tok
            if cr_tok:
                parts.append(f"Cache read: {cr_tok}")
                meta["cache_read_input_tokens"] = cr_tok
            if out_tok:
                parts.append(f"Output tokens: {out_tok}")
                meta["output_tokens"] = out_tok
            if parts:
                f.write(f"[Metadata]\n{' | '.join(parts)}\n")
                f.flush()
            if meta:
                self._last_metadata = meta

        elif event_type == "rate_limit_event":
            self._rate_limit_event = event

        # system — skip silently
