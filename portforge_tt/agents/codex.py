"""Codex CLI agent backend (new in the CHIA-free harness).

Wraps the OpenAI ``codex`` CLI (https://developers.openai.com/codex) the same
way the other backends wrap theirs: one ``codex exec`` subprocess per turn,
JSONL events on stdout for the transcript, and the final message read from
``--output-last-message`` so it never depends on event-shape quirks. With
``resume_session=True`` the session/thread id parsed from the event stream is
replayed via ``codex exec resume <id>`` on later calls.

Isolation model, mapped from the other backends' scoped configurations:

- ``-s read-only`` — the sandbox permits reads and rejects every write, so
  the only way to persist a source edit is the ``port_write`` MCP tool, and
  no build artifact or stray file can appear in the workspace.
- Model-generated commands run sandboxed with network access disabled by
  default; MCP tool calls are client-side and go through PortForge tools.
- ``--skip-git-repo-check`` (campaign workspaces are not git repos),
  ``--ignore-rules`` (no host execpolicy files).
- ``--ignore-user-config`` is added whenever a model is named explicitly, so
  no host ``config.toml`` influences the run. Without a model the user config
  must stay loadable, because it carries the default model selection.

The event stream is parsed defensively: codex has renamed its session
identifier (``session_id`` -> ``thread_id``) and its item types between
releases, so the parser accepts either shape and falls back to regex scans.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from .errors import (RateLimitError, AuthenticationError, BillingError,
                     InvalidRequestError, ServerError, MaxOutputTokensError,
                     UnknownProviderError)
from .llm import QueryResult, LLMCallBase

_SESSION_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


@dataclass
class CodexQueryResult(QueryResult):
    """:class:`QueryResult` specialised for the codex CLI.

    ``usage`` holds token counts when the event stream reported them;
    ``session_id`` is the conversation/thread id codex created (usable with
    ``codex exec resume``).
    """

    usage: Optional[dict] = None
    session_id: Optional[str] = None


def _tool_url(tool: Any) -> str:
    return getattr(tool, "url", None) or \
        f"http://{tool.hostname}:{tool.port}/{tool.name}/mcp"


def parse_session_id(stdout: str) -> Optional[str]:
    """Extract the codex session/thread id from ``exec --json`` stdout.

    Accepts both the older ``session_id`` field spelling and the newer
    ``thread_id`` one on any event; falls back to scanning for a UUID, which
    is the shape of every codex conversation id observed so far.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for key in ("session_id", "thread_id", "conversation_id"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        for key in ("session_id", "thread_id", "conversation_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    m = _SESSION_ID_RE.search(stdout)
    return m.group(0) if m else None


def parse_error_events(stdout: str) -> List[str]:
    """Collect human-readable error text from ``exec --json`` events.

    Codex reports failures as events whose type mentions ``error`` or
    ``failed`` (``turn.failed`` carrying ``error.message``, plain ``error``
    events, or older ``stream.error`` payloads). Returns every message found,
    oldest first; empty when the run had no structured failure.
    """
    messages: List[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = str(event.get("type", ""))
        if "error" not in kind and "failed" not in kind:
            continue
        error = event.get("error")
        if isinstance(error, dict):
            text = error.get("message") or error.get("type")
        elif isinstance(error, str) and error:
            text = error
        else:
            text = event.get("message")
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    return messages


def parse_usage(stdout: str) -> Optional[dict]:
    """Sum token usage from turn-completion events, when present."""
    usage: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or "usage" not in event:
            continue
        for source in (event["usage"], event["usage"].get("tokens") if isinstance(event["usage"], dict) else None):
            if not isinstance(source, dict):
                continue
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                value = source.get(key)
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
    return usage or None


def classify_text(text: str, exit_code: int) -> None:
    """Raise a typed error for a failure described by *text* (never returns on match)."""
    lowered = (text or "").lower()
    if any(kw in lowered for kw in (
            "usage limit", "rate limit", "rate_limit", "429", "too many requests",
            "quota exceeded", "you've hit your limit")):
        raise RateLimitError(
            reset_time=datetime.now(timezone.utc) + timedelta(seconds=60),
            raw_message=text[:300], exit_code=exit_code,
        )
    if any(kw in lowered for kw in (
            "authentication", "unauthorized", "401", "not logged in",
            "could not be refreshed", "please log in", "access token")):
        raise AuthenticationError(exit_code, text[:300])
    if any(kw in lowered for kw in (
            "billing", "payment", "402", "overdue", "subscription", "credit")):
        raise BillingError(exit_code, text[:300])
    if any(kw in lowered for kw in (
            "invalid request", "malformed", "400", "bad request", "invalid model",
            "model not found", "unknown model")):
        raise InvalidRequestError(exit_code, text[:300])
    if any(kw in lowered for kw in (
            "500", "503", "server error", "overloaded", "internal error",
            "service unavailable", "stream disconnected")):
        raise ServerError(exit_code, text[:300])
    if any(kw in lowered for kw in (
            "max_output_tokens", "output token limit", "maximum output",
            "response too long", "context window", "context_length_exceeded")):
        raise MaxOutputTokensError(exit_code, text[:300])
    raise UnknownProviderError(exit_code, text[:300], stderr=text)


class CodexLLM(LLMCallBase):
    """Wraps the ``codex`` CLI (``codex exec``) as an agent backend."""

    supports_dangerously_skip_permissions = True

    def __init__(
        self,
        model: Optional[str] = None,
        system_message: str = "",
        timeout_seconds: int = 600,
        retries: int = 3,
        logging_name: str = "codex",
        logging_level: int = logging.DEBUG,
        log_dir: Optional[str] = None,
        codex_bin: str = "codex",
        work_dir: Optional[str] = None,
        web: bool = False,
        sandbox: str = "read-only",
        extra_cli_args: Optional[List[str]] = None,
        dangerously_skip_permissions: bool = False,
        resume_session: bool = False,
    ):
        super().__init__(system_message=system_message,
                         dangerously_skip_permissions=dangerously_skip_permissions)
        self.logging_name = logging_name
        self.logging_level = logging_level
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.codex_bin = codex_bin
        self.work_dir = work_dir
        self.web = web
        if sandbox not in ("read-only", "workspace-write", "danger-full-access"):
            raise ValueError(f"unsupported codex sandbox mode {sandbox!r}")
        self.sandbox = sandbox
        self.extra_cli_args = extra_cli_args or []
        self.logger = logging.getLogger(logging_name)
        self.resume_session = resume_session
        self.session_id: Optional[str] = None
        self._last_metadata: dict = {}

        self._log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_prefix = os.path.join(log_dir, f"{logging_name}_{run_id}")
        else:
            self._log_prefix = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prompt(
        self,
        user_message: str,
        tools: Optional[List[Any]] = [],
    ) -> CodexQueryResult:
        """Send *user_message* to codex and return the response.

        Raises the shared typed errors (see :mod:`errors`) exactly like the
        other backends: capacity/auth/billing/invalid-request propagate
        immediately; transient failures retry with backoff first.
        """
        import time as _time

        for attempt in range(self.retries):
            try:
                self._last_metadata = {}
                cli = self._run_codex(user_message, tools)
                self._last_metadata["model"] = self.model or "<codex default>"
                self._last_metadata["tools"] = [
                    {"name": t.name, "url": _tool_url(t)} for t in tools
                ]
                self._classify_error(cli)
                cli.success = True
                return cli

            except (RateLimitError, AuthenticationError, BillingError, InvalidRequestError):
                raise

            except MaxOutputTokensError:
                if attempt == 0:
                    self.logger.warning(
                        "Max output tokens on attempt %d/%d, retrying once",
                        attempt + 1, self.retries,
                    )
                    continue
                raise

            except ServerError:
                backoff = min(5 * 2 ** attempt, 60)
                self.logger.warning(
                    "Server error on attempt %d/%d, backing off %ds",
                    attempt + 1, self.retries, backoff,
                )
                _time.sleep(backoff)

            except UnknownProviderError as exc:
                self.logger.warning(
                    "Unknown error on attempt %d/%d: %s",
                    attempt + 1, self.retries, exc,
                )

            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "Timeout on attempt %d/%d", attempt + 1, self.retries,
                )

            except Exception as exc:
                self.logger.warning(
                    "Unexpected error on attempt %d/%d: %s",
                    attempt + 1, self.retries, exc,
                )
        return CodexQueryResult(result="", returncode=-1, stderr="", stream_result="", success=False)

    def _classify_error(self, cli: CodexQueryResult) -> None:
        """Raise a typed error for a failed run.

        A clean run (exit 0 with a non-empty final message) returns. Anything
        else is classified from the structured error events first (preferred;
        codex reports provider failures there) with stderr as the fallback
        text — the same precedence the opencode backend uses for its export
        errors.
        """
        if cli.returncode == 0 and cli.result:
            return
        events = parse_error_events(cli.stream_result)
        text = "; ".join(events) or cli.stderr or "empty response"
        classify_text(text, cli.returncode)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_cmd(self, user_message: str, tools: Optional[List[Any]], last_message_path: str) -> list:
        """Build the ``codex exec`` command list (message arrives on stdin).

        Global options precede the optional ``resume <id>`` subcommand: ``-s``
        and ``-C`` exist only on ``exec`` itself, while the shared flags are
        accepted in either position.
        """
        cmd = [self.codex_bin, "exec",
               "--json",
               "--skip-git-repo-check",
               "--ignore-rules",
               "-s", self.sandbox,
               "-o", last_message_path]
        if self.model:  # omit -m so codex uses its configured default
            cmd += ["-m", self.model, "--ignore-user-config"]
        # The sandbox already blocks network access; web search is a separate
        # client-side tool the model may only use when explicitly enabled.
        cmd += ["-c", "tools.web_search=" + ("true" if self.web else "false")]
        if self.work_dir:
            cmd += ["-C", self.work_dir]
        for tool in tools or []:
            # Streamable-HTTP MCP server, configured inline as TOML overrides.
            cmd += ["-c", f"mcp_servers.{tool.name}.url=\"{_tool_url(tool)}\""]
        if self.extra_cli_args:
            cmd += self.extra_cli_args
        if self.resume_session and self.session_id:
            cmd += ["resume", self.session_id]
        cmd.append("-")  # read the prompt from stdin
        return cmd

    def _run_codex(
        self,
        user_message: str,
        tools: Optional[List[Any]] = None,
    ) -> CodexQueryResult:
        """Run ``codex exec`` and assemble a QueryResult.

        Stdout (the JSONL event stream) is captured to a file, matching the
        opencode backend: pipe buffers truncate long streams mid-JSON. The
        final assistant message is read from the ``--output-last-message``
        file, which codex writes even when the event vocabulary changes.
        """
        tools = tools or []
        with tempfile.TemporaryDirectory(prefix="portforge-codex-") as directory:
            last_message = os.path.join(directory, "last-message.txt")
            cmd = self._build_cmd(user_message, tools, last_message)
            self.logger.info("Running: %s ...", " ".join(cmd[:6]))
            env = {k: v for k, v in os.environ.items() if k != "CODEX_SANDBOX"}

            stream = tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codex_events_", delete=False
            )
            stream_path = stream.name
            try:
                with open(stream_path, "w") as out_fh:
                    proc = subprocess.run(
                        cmd,
                        input=user_message,
                        stdout=out_fh,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout_seconds,
                        env=env,
                    )
                with open(stream_path, "r") as in_fh:
                    stdout = in_fh.read()
            finally:
                try:
                    os.unlink(stream_path)
                except OSError:
                    pass

            session_id = parse_session_id(stdout)
            if session_id is not None:
                self.session_id = session_id
            usage = parse_usage(stdout)
            final = ""
            if os.path.exists(last_message):
                final = open(last_message, encoding="utf-8", errors="replace").read().strip()

            if self._log_prefix is not None:
                try:
                    truncated = user_message[:500] + ("..." if len(user_message) > 500 else "")
                    with open(f"{self._log_prefix}.log", "a") as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"session {session_id}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"[User Message]\n{truncated}\n\n")
                        f.write(stdout)
                        f.write("-" * 80 + "\n\n")
                except OSError as exc:
                    self.logger.warning(
                        "Could not write codex log %s.log: %s", self._log_prefix, exc
                    )

            return CodexQueryResult(
                result=final,
                returncode=proc.returncode,
                stderr=proc.stderr or "",
                stream_result=stdout,
                usage=usage,
                session_id=session_id,
            )
