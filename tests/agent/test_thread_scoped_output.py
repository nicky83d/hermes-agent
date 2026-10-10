"""Tests for agent.thread_scoped_output.thread_scoped_silence.

Behaviour contract: a thread inside ``thread_scoped_silence()`` has its
stdout/stderr routed to devnull, while every OTHER thread keeps writing to the
real stream — even concurrently, while the first thread is still inside the
context.  This is the property the old process-global
``contextlib.redirect_stdout(devnull)`` violated (issue #55769 / #55925).
"""

import contextlib
import io
import sys
import threading
import time

import agent.thread_scoped_output as thread_output
from agent.thread_scoped_output import thread_scoped_silence

# IMPORTED AT MODULE SCOPE, like every other test that touches this module: importing
# ``agent.process_bootstrap`` pulls in ``hermes_bootstrap``, whose PM probe stats the real
# ``~/.hermes/manifest.json`` - legitimate for a real process, refused by tests/home_io_guard.py when
# it happens inside a test. Collection happens before that guard is armed.
from agent.process_bootstrap import _SafeWriter, _install_safe_stdio  # noqa: E402


def _run_with_real_stream(fn):
    """Bind a StringIO as the real stdout, run fn, return what reached it."""
    real_out = io.StringIO()
    orig = sys.stdout
    sys.stdout = real_out
    try:
        fn()
    finally:
        sys.stdout = orig
    return real_out.getvalue()






def test_stderr_is_also_routed_per_thread():
    real_err = io.StringIO()
    orig = sys.stderr
    sys.stderr = real_err
    try:
        with thread_scoped_silence():
            sys.stderr.write("err-dropped\n")
        sys.stderr.write("err-kept\n")
    finally:
        sys.stderr = orig
    out = real_err.getvalue()
    assert "err-dropped" not in out
    assert "err-kept" in out






def test_many_concurrent_silenced_and_loud_threads():
    """Stress: interleaved silenced/loud threads keep their respective fates."""
    start = threading.Event()
    results_lock = threading.Lock()

    def silenced(i):
        start.wait(timeout=2.0)
        with thread_scoped_silence():
            print(f"S{i}")
            time.sleep(0.05)

    def loud(i):
        start.wait(timeout=2.0)
        time.sleep(0.02)
        print(f"L{i}")

    def body():
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=silenced, args=(i,)))
            threads.append(threading.Thread(target=loud, args=(i,)))
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=15.0)
        assert not any(t.is_alive() for t in threads), "straggler thread would truncate captured output"

    captured = _run_with_real_stream(body)
    for i in range(5):
        assert f"S{i}" not in captured, f"silenced S{i} leaked"
        assert f"L{i}" in captured, f"loud L{i} swallowed"


def test_repeated_contexts_never_write_to_a_closed_sink():
    """The installed proxy must survive later silenced workers."""
    original = sys.stdout
    try:
        for _ in range(3):
            with thread_scoped_silence():
                sys.stdout.write("hidden\n")
            sys.stdout.fileno()
    finally:
        sys.stdout = original


def test_temporary_global_redirects_do_not_allocate_new_sinks(monkeypatch):
    """A displaced proxy is temporary, not a reason to leak another FD pair."""
    opened_sinks = []

    def fake_open(*_args, **_kwargs):
        sink = io.StringIO()
        opened_sinks.append(sink)
        return sink

    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "open", fake_open, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        with thread_scoped_silence():
            pass
        assert len(opened_sinks) == 2
        original_proxies = dict(thread_output._installed)

        for _ in range(20):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with thread_scoped_silence():
                    print("hidden")

        with thread_scoped_silence():
            pass
        assert len(opened_sinks) == 2
        assert thread_output._installed == original_proxies
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_a_wrapper_and_a_proxy_cannot_forward_in_a_circle():
    """The live failure, pinned: two stream proxies that are each other's target must not recurse.

    Measured on the gateway 01:54 -> 09:19: every agent created in an api_server worker thread died in
    ``setup_logging`` at ``getattr(sys.stdout, "line_buffering")`` with ~500 alternating
    ``_SafeWriter.__getattr__`` / ``_ThreadRoutingStream.__getattr__`` frames, so each request Janet
    made for a choice came back HTTP 500. A stream that leads back to itself has no such attribute.
    """
    state = thread_output._RoutingState(io.StringIO())
    proxy = thread_output._ThreadRoutingStream(io.StringIO(), state)
    wrapper = _SafeWriter(proxy)
    proxy._passthrough = wrapper   # type: ignore[assignment]  # the cycle the live stack created

    assert wrapper._inner is proxy and proxy._target() is wrapper
    assert getattr(proxy, "line_buffering", False) is False
    assert getattr(wrapper, "line_buffering", False) is False
    assert not hasattr(proxy, "no_such_attribute_at_all")
    assert not hasattr(wrapper, "no_such_attribute_at_all")
    # and the real attributes still work through a cycle-free chain
    plain = _SafeWriter(thread_output._ThreadRoutingStream(io.StringIO(), state))
    assert getattr(plain, "line_buffering") is False


def test_installing_never_stacks_a_proxy_behind_a_wrapper(monkeypatch):
    """Every agent construction wraps stdout again; the proxy under the wrapper must be ADOPTED.

    Recognising a proxy only when it *was* ``sys.stdout`` let the stack grow one layer per agent
    created - the shape that ended in the cycle above. Routing must still be per thread.
    """
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "_routing_states", {}, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    real_out = io.StringIO()
    sys.stdout, sys.stderr = real_out, io.StringIO()
    try:
        with thread_scoped_silence():
            pass
        proxy = thread_output._installed["stdout"]
        for _ in range(5):
            _install_safe_stdio()                    # what AIAgent.__init__ does, every time
            with thread_scoped_silence():
                pass
            sys.stdout.write("kept\n")
        assert thread_output._installed["stdout"] is proxy, "the proxy was replaced, not adopted"
        assert thread_output._installed_proxy_within(sys.stdout) is proxy
        assert "kept" in real_out.getvalue()
        assert getattr(sys.stdout, "line_buffering", False) is False
        with thread_scoped_silence():
            sys.stdout.write("dropped\n")
        assert "dropped" not in real_out.getvalue(), "routing through the wrapper stopped working"
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_silence_survives_redirect_restoring_an_older_proxy(monkeypatch):
    """Silencing is stream-wide, even when a redirect swaps proxy generations."""
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    passthrough = io.StringIO()
    sys.stdout = passthrough
    entered = threading.Event()
    release = threading.Event()

    try:
        with thread_scoped_silence():
            pass

        def worker():
            with thread_scoped_silence():
                entered.set()
                assert release.wait(timeout=10)
                print("must-stay-silenced")

        redirected = io.StringIO()
        with contextlib.redirect_stdout(redirected):
            thread = threading.Thread(target=worker)
            thread.start()
            assert entered.wait(timeout=10)

        release.set()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert "must-stay-silenced" not in passthrough.getvalue()
        assert "must-stay-silenced" not in redirected.getvalue()
    finally:
        release.set()
        sys.stdout, sys.stderr = original_stdout, original_stderr
