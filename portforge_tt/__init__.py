"""PortForge-TT: autonomous model porting to Tenstorrent hardware.

A supervisor agent plans one bounded unit of work at a time; a worker agent
researches the model, writes the TTNN backend, and runs isolated jobs on a TT
card through scoped MCP tools. A trusted evaluator compares every candidate
with an independent FP32 reference, first on small suites and finally, once,
on a frozen held-out corpus.

The package is self-contained (no CHIA/Ray dependency): ``runtime`` provides
local resource leases and background jobs, ``mcpserver`` serves the agent
tools over FastMCP, and ``agents`` wraps the ``opencode``, ``claude`` and
``codex`` CLIs. The top-level modules also run *flat* — a campaign freezes
the harness source into ``runs/<run>/harness/`` and re-executes those plain
files — which is why sibling imports there use the ``if __package__:`` shim
and ``agents`` keeps working as a top-level package inside the harness.
"""

__version__ = "2.0.0"
