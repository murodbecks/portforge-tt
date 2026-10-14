"""MCP tool server base class: FastMCP served by a local uvicorn thread.

This is a local, dependency-light replacement for CHIA's ``ChiaTool`` (which
deployed the same FastMCP app inside a Ray actor). Subclasses register tool
functions in ``setup()`` exactly as before::

    class MyTools(ToolServer):
        def setup(self, work_dir):
            self.work_dir = Path(work_dir)
            self.mcp.add_tool(self.read, name="my_read")

        def read(self, path: str) -> str:
            ...

    tools = MyTools("my_tools", work_dir="/tmp")   # serves immediately
    tools.url   # http://127.0.0.1:<port>/my_tools/mcp
    tools.stop()

A subclass that defines ``setup`` but no ``__init__`` gets a constructor
automatically: ``ToolServer.__init__(name)`` first, then ``setup(*args,
**kwargs)``, then the server starts — the same contract the CHIA base class
offered, so existing subclasses keep working. Heavy imports (``mcp``,
``fastapi``, ``uvicorn``) happen lazily so that merely importing this module
(for example in tests that exercise tool logic through ``object.__new__``)
needs none of them.
"""
from __future__ import annotations

import logging
import threading
import time

_taken_ports: set[int] = set()
_registry: list["ToolServer"] = []


def active_servers() -> list["ToolServer"]:
    return list(_registry)


class ToolServer:
    """Base class for MCP tool servers served from the controller process."""

    def __init__(self, name: str, *, host: str = "127.0.0.1",
                 base_port: int = 8000, max_tries: int = 100,
                 logging_level: int = logging.DEBUG):
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings

        self.name = name
        self.hostname = host  # advertised to agent CLIs; agents are local too
        self.base_port, self.max_tries = base_port, max_tries
        # stateless HTTP: every request carries its own session, which is all
        # the scoped agent tools need (same settings the CHIA base used).
        self.mcp = FastMCP(
            name,
            stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            ),
        )
        self.port = None  # set by start()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging_level)
        self._server = None
        self._thread = None

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def setup(self, *args, **kwargs):
        """Register tools with ``self.mcp.add_tool(...)``; override this."""
        raise NotImplementedError(
            f"{type(self).__name__} defines neither __init__ nor setup(); "
            "implement one of them."
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Opt-in convenience, mirroring the CHIA base: a subclass that defines
        # setup() but no __init__ is constructed as
        # ChiaTool.__init__ -> setup(*args, **kwargs) -> start().
        if "setup" in cls.__dict__ and "__init__" not in cls.__dict__:
            def _auto_init(self, name, *args, base_port=8000, max_tries=100,
                           logging_level=logging.DEBUG, **kwargs):
                ToolServer.__init__(self, name, host="127.0.0.1", base_port=base_port,
                                    max_tries=max_tries, logging_level=logging_level)
                self.setup(*args, **kwargs)
                self.start()

            _auto_init.__name__ = "__init__"
            _auto_init.__qualname__ = f"{cls.__qualname__}.__init__"
            cls.__init__ = _auto_init

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Serve the FastMCP app on a free local port; returns the port."""
        import uvicorn
        from contextlib import AsyncExitStack, asynccontextmanager
        from fastapi import FastAPI

        mcp = self.mcp

        @asynccontextmanager
        async def lifespan(app):
            # Runs every mounted session manager concurrently.
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(mcp.session_manager.run())
                yield

        for port in range(self.base_port, self.base_port + self.max_tries):
            if port in _taken_ports:
                continue
            # A session manager can only .run() once per instance, so create a
            # fresh app per attempt (identical to the CHIA start_router).
            mcp._session_manager = None
            app = FastAPI(lifespan=lifespan)
            app.mount(f"/{self.name}", mcp.streamable_http_app())
            config = uvicorn.Config(app, host=self.hostname, port=port, log_level="warning")
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, daemon=True,
                                      name=f"portforge-mcp-{self.name}")
            thread.start()
            for _ in range(50):  # wait up to five seconds for a bind
                time.sleep(0.1)
                if server.started:
                    _taken_ports.add(port)
                    self.port, self._server, self._thread = port, server, thread
                    _registry.append(self)
                    self.logger.info("%s serving at %s", self.name, self.url)
                    return port
            server.should_exit = True
            thread.join(timeout=2)
        raise RuntimeError(
            f"no available port in range {self.base_port}-{self.base_port + self.max_tries - 1}"
        )

    @property
    def url(self) -> str:
        if self.port is None:
            raise RuntimeError(f"tool server {self.name!r} is not started")
        return f"http://{self.hostname}:{self.port}/{self.name}/mcp"

    def stop(self) -> None:
        """Stop the HTTP server thread (registered tools stay defined)."""
        if self in _registry:
            _registry.remove(self)
        server, thread, port = self._server, self._thread, self.port
        self._server = self._thread = None
        self.port = None
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                self.logger.warning("MCP server thread for %s still alive after 5s", self.name)
        if port is not None:
            _taken_ports.discard(port)
