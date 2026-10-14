"""Local job runtime: named-resource leases and background execution.

This replaces the CHIA/Ray dispatch layer PortForge was originally built on.
Jobs are plain functions; ``@job`` declares the named resources (remote
hosts, agent credentials) a job consumes. Calling the decorated function
directly runs it in-process *without* acquiring anything (the historical
local-call semantics of ``ChiaFunction``); ``.submit()`` acquires the named
resources and runs the function on a background thread, returning a
``concurrent.futures.Future``.

All durable state is the files each job writes itself (``jobs/<id>/request.json``
and ``result.json``), so a controller restart loses nothing a future would
have held: futures are an in-process convenience, not storage. Resources are
acquired in sorted-name order, which keeps concurrent multi-resource jobs
deadlock-free without a central scheduler.
"""
from __future__ import annotations

import functools
import threading
import time
from concurrent.futures import Future

if __package__:  # installed as part of the portforge_tt package
    from . import tracer
else:  # flat: a frozen run harness or a direct script execution
    import tracer

_RESOURCES: dict[str, threading.BoundedSemaphore] = {}
_THREADS: list[threading.Thread] = []
_LOCK = threading.Lock()


def configure(**counts: int) -> None:
    """Define the available units of each named resource (default one each).

    Mirrors what the controller used to pass to ``ray.init(resources=...)``:
    ``configure(agent_creds=1, reference=1, tt=4)`` allows four concurrent TT
    jobs and one reference/agent job each. Calling this again replaces the
    capacity of the named resources; in-flight jobs keep their acquired units.
    """
    with _LOCK:
        for name, count in counts.items():
            if type(count) is not int or count < 1:
                raise ValueError(f"resource {name!r} needs a positive integer capacity")
            _RESOURCES[name] = threading.BoundedSemaphore(count)


def _semaphore(name: str) -> threading.BoundedSemaphore:
    with _LOCK:
        return _RESOURCES.setdefault(name, threading.BoundedSemaphore(1))


class _Leases:
    """Context manager acquiring declared units of every named resource."""

    def __init__(self, resources: dict):
        self.resources = {name: int(value) for name, value in (resources or {}).items()}
        self.held: list[str] = []

    def __enter__(self):
        for name in sorted(self.resources):
            for _ in range(self.resources[name]):
                _semaphore(name).acquire()
                self.held.append(name)
        return self

    def __exit__(self, *exception):
        for name in reversed(self.held):
            _semaphore(name).release()
        return False


def leases(**resources):
    """Acquire named resource units directly (``with leases(tt=1): ...``)."""
    return _Leases(resources)


def get(future):
    """Wait for a submitted job and return its result (re-raises its error)."""
    return future.result()


def shutdown(timeout: float = 30.0) -> None:
    """Wait for background job threads; they are daemons, so this is best-effort."""
    deadline = time.monotonic() + timeout
    for thread in list(_THREADS):
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)


def job(resources: dict | None = None, max_retries: int = 0):
    """Decorator marking a function as a lease-holding background job.

    Usage::

        @job(resources={"tt": 1})
        def experiment(run_dir, job_id, request, config):
            ...

        experiment(...)            # direct in-process call, acquires nothing
        future = experiment.submit(...)   # acquires leases, runs on a thread
        result = get(future)             # or future.result()
        experiment._original(...)        # the raw undecorated function
    """
    if max_retries != 0:
        raise ValueError("only max_retries=0 is supported; job durability is file-based")

    def decorate(function):
        return _Job(function, resources or {})

    return decorate


class _Job:
    """Callable wrapper returned by :func:`job` (never constructed directly)."""

    def __init__(self, function, resources: dict):
        self.fn = function
        self.resources = {name: int(value) for name, value in resources.items()}
        functools.update_wrapper(self, function)

    @property
    def _original(self):
        """The undecorated function (tests and direct in-process callers)."""
        return self.fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def submit(self, *args, **kwargs) -> Future:
        future: Future = Future()
        name = getattr(self.fn, "__name__", repr(self.fn))

        def run():
            if not future.set_running_or_notify_cancel():
                return
            started = time.monotonic()
            tracer.event("job.leased", job=name, resources=self.resources)
            try:
                with _Leases(self.resources):
                    future.set_result(self.fn(*args, **kwargs))
            except BaseException as exception:  # surfaced to the caller by get()
                future.set_exception(exception)
            finally:
                tracer.event("job.done", job=name, seconds=round(time.monotonic() - started, 3))

        thread = threading.Thread(target=run, name="portforge-" + name, daemon=True)
        with _LOCK:
            _THREADS.append(thread)
        thread.start()
        return future
