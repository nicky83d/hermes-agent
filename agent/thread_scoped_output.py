"""Thread-scoped stdout/stderr silencing for background worker threads.

``contextlib.redirect_stdout`` reassigns the *process-global* stream, so a daemon worker
silencing itself also silences every other thread (gateway event loop included). This
module installs a per-thread routing proxy as ``sys.stdout``/``sys.stderr``: silenced
threads write to a sink, everyone else passes through to the original stream. Installed
once, idempotently, and never uninstalled (that would race other threads mid-write).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Iterator, TextIO

__all__ = ["thread_scoped_silence"]

_install_lock = threading.Lock()
# Proxy installed per attribute ("stdout"/"stderr"): never double-wrap.
_installed: dict[str, "_ThreadRoutingStream"] = {}
# One process-lifetime sink per stream: global redirects that displace and
# restore a proxy must not leak a new /dev/null descriptor each time.
_sinks: dict[str, TextIO] = {}
_routing_states: dict[str, "_RoutingState"] = {}


class _RoutingState:
    """Silencing registry shared by every proxy generation for one stream."""

    def __init__(self, sink: TextIO) -> None:
        self.sink = sink
        self.silenced: dict[int, int] = {}
        self.lock = threading.Lock()


class _ThreadRoutingStream:
    """``sys.stdout``/``sys.stderr`` stand-in routing writes per calling thread;
    unknown attributes delegate to the current thread's target."""

    def __init__(self, passthrough: TextIO, state: _RoutingState) -> None:
        self._passthrough = passthrough
        self._state = state

    def _target(self) -> TextIO:
        return self._state.sink if self._state.silenced.get(threading.get_ident(), 0) > 0 else self._passthrough

    def silence(self, ident: int) -> None:
        with self._state.lock:
            self._state.silenced[ident] = self._state.silenced.get(ident, 0) + 1

    def unsilence(self, ident: int) -> None:
        with self._state.lock:
            depth = self._state.silenced.get(ident, 0) - 1
            if depth > 0:
                self._state.silenced[ident] = depth
            else:
                self._state.silenced.pop(ident, None)

    def _forward(self, name: str, fallback, *args):  # type: ignore[no-untyped-def]
        """Call ``name`` on the current target; a dead target yields ``fallback(*args)`` instead of raising."""
        try:
            return getattr(self._target(), name)(*args)
        except Exception:
            return fallback(*args)

    def write(self, data):  # type: ignore[no-untyped-def]
        return self._forward("write", lambda d: len(d) if isinstance(d, str) else 0, data)

    def flush(self):  # type: ignore[no-untyped-def]
        return self._forward("flush", lambda: None)

    def writelines(self, lines):  # type: ignore[no-untyped-def]
        return self._forward("writelines", lambda _l: None, lines)

    def isatty(self) -> bool:
        try:
            return bool(self._target().isatty())
        except Exception:
            return False

    def fileno(self):  # type: ignore[no-untyped-def]
        return self._target().fileno()

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        target = self._target()
        # NEVER DELEGATE IN A CIRCLE. ``_SafeWriter`` and this proxy each forward unknown attributes
        # to what they wrap, so if each one is the other's target the two frames alternate until
        # Python's stack dies - which is exactly what happened live on the gateway: from 01:54 to
        # 09:19 every agent created in an api_server worker thread failed in ``setup_logging`` at
        # ``getattr(sys.stdout, "line_buffering")`` with ~500 alternating ``_SafeWriter.__getattr__``
        # / ``_ThreadRoutingStream.__getattr__`` frames, so every request Janet made for a choice came
        # back HTTP 500. A stream that leads back here has no such attribute; say so.
        if target is self or _reaches(target, self):
            raise AttributeError(name)
        return getattr(target, name)


def _reaches(stream, want, limit: int = 32) -> bool:
    """True if following ``stream``'s wrappers reaches ``want`` - i.e. walking it would loop.

    Follows the two delegation links in either order (``_passthrough`` for a routing proxy,
    ``_inner`` for a writer wrapper) using ``object.__getattribute__`` so a proxy's own
    ``__getattr__`` is never consulted while we are trying to decide whether to consult it.
    """
    seen = 0
    while stream is not None and seen < limit:
        if stream is want:
            return True
        try:
            stream = object.__getattribute__(stream, "_passthrough")
        except Exception:
            try:
                stream = object.__getattribute__(stream, "_inner")
            except Exception:
                return False
        seen += 1
    return False


def _installed_proxy_within(stream, limit: int = 32):
    """A routing proxy already installed somewhere in ``stream``'s chain, or None.

    ``_ensure_installed`` only RECOGNISED a proxy when it was ``sys.<attr>`` itself. Any wrapper in
    front of it (``_install_safe_stdio`` puts a ``_SafeWriter`` there, and every agent construction
    calls it) made the next install build yet another routing proxy whose passthrough was the wrapper
    around the previous one - one more layer per agent created, and the two ends of that stack
    delegating to each other. Finding the proxy through the wrapper stops the stack growing and
    keeps routing intact, because the wrapper forwards ``write``/``flush`` into it either way.
    """
    seen = 0
    while stream is not None and seen < limit:
        if isinstance(stream, _ThreadRoutingStream):
            return stream
        try:
            stream = object.__getattribute__(stream, "_inner")
        except Exception:
            try:
                stream = object.__getattribute__(stream, "_passthrough")
            except Exception:
                return None
        seen += 1
    return None


def _ensure_installed(attr: str, passthrough: TextIO) -> "_ThreadRoutingStream":
    """Install (idempotently) a routing proxy as ``sys.<attr>`` and return it."""
    with _install_lock:
        proxy = _installed.get(attr)
        current = getattr(sys, attr, None)
        if isinstance(current, _ThreadRoutingStream):
            # A redirect context may restore an older proxy; adopt it rather
            # than wrapping it into an unbounded chain.
            _installed[attr] = current
            _routing_states[attr] = current._state
            return current
        if proxy is not None and current is proxy:
            return proxy
        # A PROXY IN FRONT OF A WRAPPER IS STILL INSTALLED. Recognising only a bare proxy let the
        # chain grow one layer per agent construction (see _installed_proxy_within) until its two
        # ends delegated to each other.
        existing = _installed_proxy_within(current)
        if existing is not None:
            _installed[attr] = existing
            _routing_states[attr] = existing._state
            return existing
        # Route non-silenced threads to whatever is currently bound (an active
        # global redirect keeps its old behavior).
        passthrough = current if current is not None else passthrough
        sink = _sinks.get(attr)
        if sink is None or sink.closed:
            sink = _sinks[attr] = open(os.devnull, "w", encoding="utf-8")
        state = _routing_states.get(attr)
        if state is None or state.sink is not sink:
            state = _routing_states[attr] = _RoutingState(sink)
        proxy = _installed[attr] = _ThreadRoutingStream(passthrough, state)
        setattr(sys, attr, proxy)
        return proxy


@contextlib.contextmanager
def thread_scoped_silence() -> Iterator[None]:
    """Silence ``stdout``/``stderr`` for the *current thread only*."""
    ident = threading.get_ident()
    proxies = (_ensure_installed("stdout", sys.__stdout__ or sys.stdout), _ensure_installed("stderr", sys.__stderr__ or sys.stderr))
    for proxy in proxies:
        proxy.silence(ident)
    try:
        yield
    finally:
        for proxy in proxies:
            proxy.unsilence(ident)
