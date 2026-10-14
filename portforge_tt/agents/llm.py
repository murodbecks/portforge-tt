"""Result type and polymorphic base for the agent CLI backends.

Ported from ``chia.base.llm_call`` (Copyright (c) 2026 The Regents of the
University of California; BSD-3-Clause, see LICENSE), minus the Ray/ChiaTool typing: tool servers are
now any object carrying ``name``, ``hostname`` and ``port`` (see
``mcpserver.ToolServer``).
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

# Sentinel for "argument not provided": distinguishes an explicit value (which
# warrants a warning on a backend that ignores it) from the unset default.
UNSET = object()


@dataclass
class QueryResult:
    """Structured result from prompting an LLM or agent CLI.

    result: the final assistant response
    returncode: the CLI process exit code (0 on clean runs)
    stderr: the CLI's stderr (CLIs only)
    stream_result: full transcript of all turns
    success: whether the prompt completed successfully
    """

    result: str
    returncode: int
    stderr: str
    stream_result: str
    success: bool = False


class LLMCallBase(ABC):
    """Polymorphic base for LLM/agent configuration traits and behavior."""

    # Capability flags — subclasses that honor these controls override them.
    supports_dangerously_skip_permissions: bool = False
    supports_config: bool = False

    def __init__(self, system_message: str,
                 dangerously_skip_permissions=UNSET, config=UNSET):
        self.system_message = system_message
        cls = type(self).__name__
        if (dangerously_skip_permissions is not UNSET
                and not self.supports_dangerously_skip_permissions):
            warnings.warn(
                f"{cls} does not support 'dangerously_skip_permissions'; the "
                f"argument is ignored (this backend has no permission gate).",
                stacklevel=2,
            )
        if config is not UNSET and not self.supports_config:
            warnings.warn(
                f"{cls} does not support a 'config' block; the argument is ignored.",
                stacklevel=2,
            )
        # Maps only to the backend's "dangerously skip permissions" CLI flag
        # (claude/opencode --dangerously-skip-permissions, codex
        # --dangerously-bypass-approvals-and-sandbox). Honored only where
        # supports_dangerously_skip_permissions is True.
        self.dangerously_skip_permissions = (
            True if dangerously_skip_permissions is UNSET else dangerously_skip_permissions
        )
        # The backend's config block (e.g. opencode's `permission` object).
        # ``None`` means "allow all". Honored only where supports_config is True.
        self.config = None if config is UNSET else config

    @abstractmethod
    def prompt(self, user_message: str,
               tools: Optional[List[Any]] = []) -> QueryResult:
        """Send a prompt to this agent backend."""
