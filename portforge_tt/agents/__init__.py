"""Agent roles and the CLI backends they run on.

``agent`` holds the two campaign roles (supervisor and worker) and their
turn protocol; ``opencode``/``claude``/``codex`` wrap one agent CLI each;
``errors`` and ``llm`` hold the shared result type and provider-error
taxonomy.
"""
from .agent import (BACKENDS, DEFAULT_MODELS, MODEL, SUPERVISOR_TIMEOUT_SECONDS,
                    WORKER_TIMEOUT_SECONDS, IsolatedClaude, IsolatedCodex,
                    IsolatedOpenCode, error_kind, parse_decision, query,
                    repository_root, role_selection, save, supervisor_node,
                    worker_node)

__all__ = [
    "BACKENDS", "DEFAULT_MODELS", "MODEL", "SUPERVISOR_TIMEOUT_SECONDS",
    "WORKER_TIMEOUT_SECONDS", "IsolatedClaude", "IsolatedCodex", "IsolatedOpenCode",
    "error_kind", "parse_decision", "query", "repository_root", "role_selection",
    "save", "supervisor_node", "worker_node",
]
