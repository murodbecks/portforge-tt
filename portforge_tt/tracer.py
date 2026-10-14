"""Controller-side tracing: a small JSONL event log.

Replaces CHIA's Ray-actor profiler with a plain append-only file. The
controller (and the job runtime) append typed events with UTC timestamps;
nothing is sent anywhere, and the log is only written while a tracer is
active. Events are advisory diagnostics for the operator — job truth stays
in each job's request/result files.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_handle = None
_path: Path | None = None


def start(log_dir: str | os.PathLike) -> Path:
    """Start appending events to ``<log_dir>/events.jsonl`` (idempotent)."""
    global _handle, _path
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _lock:
        if _handle is not None:
            return _path
        _path = directory / "events.jsonl"
        _handle = _path.open("a", encoding="utf-8")
        return _path


def event(kind: str, **fields) -> None:
    """Append one event; a no-op until :func:`start` was called."""
    with _lock:
        if _handle is None:
            return
        record = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        _handle.write(json.dumps(record, default=str) + "\n")
        _handle.flush()


def stop() -> None:
    global _handle, _path
    with _lock:
        if _handle is not None:
            _handle.close()
        _handle, _path = None, None


def active() -> bool:
    return _handle is not None
