"""Low-level I/O and process-control utilities for Jarvis.

Holds the native-output suppression context managers (used to silence PDFBox/
Java warnings and phonemizer/Kokoro chatter), the SIGALRM-based ``time_limit``
guard, and the logging filter that drops PDFBox noise from captured stderr.
Foundation module: depends only on the standard library.
"""
import logging
import math
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional


class JavaWarningFilter(logging.Filter):
    """Filter out Java/PDFBox warnings from stderr captured by libraries"""
    def filter(self, record):
        if record.getMessage():
            msg = record.getMessage()
            if any(x in msg for x in [
                "org.apache.pdfbox",
                "No Unicode mapping",
                "Cannot read JBIG2",
                "toUnicode",
                "Got stderr: Apr",
                "Got stderr: Feb",
                "Got stderr: Jan",
                "Got stderr: Mar",
                "Got stderr: May",
                "Got stderr: Jun",
                "Got stderr: Jul",
                "Got stderr: Aug",
                "Got stderr: Sep",
                "Got stderr: Oct",
                "Got stderr: Nov",
                "Got stderr: Dec",
            ]):
                return False
        return True


java_filter = JavaWarningFilter()
for handler in logging.root.handlers:
    handler.addFilter(java_filter)


@contextmanager
def suppress_external_output():
    """Suppress stdout AND stderr written by C/Java/native libraries (e.g.
    PDFBox warnings during PDF processing, phonemizer/Kokoro chatter during
    TTS synthesis). Python loggers configured to use sys.stderr are also
    silenced for the duration of the block."""
    import sys
    import os
    stderr_backup = sys.stderr
    stdout_backup = sys.stdout
    try:
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
        sys.stdout = devnull
        yield
    finally:
        sys.stderr = stderr_backup
        sys.stdout = stdout_backup
        devnull.close()


@contextmanager
def suppress_external_stderr():
    """Suppress Python-level stderr chatter (phonemizer 'words count mismatch'
    warnings, Kokoro INFO lines) WITHOUT touching sys.stdout.

    Used on the background TTS worker thread. The full suppress_external_output
    replaces the process-global sys.stdout, so when synthesis overlapped with the
    main REPL thread's input("You: ") the prompt was written to /dev/null and
    appeared to vanish until the next turn. Leaving sys.stdout untouched keeps the
    interactive prompt visible while still silencing the (stderr-only) TTS noise.
    """
    import sys
    stderr_backup = sys.stderr
    devnull = open(os.devnull, 'w')
    try:
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = stderr_backup
        devnull.close()


# Backwards-compatible alias for existing callers that imported the
# original PDF-specific name.
suppress_java_warnings = suppress_external_output


@contextmanager
def time_limit(seconds: Optional[float], operation: str = "operation"):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    timeout_seconds = max(1, int(math.ceil(seconds)))

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Timed out after {timeout_seconds}s during {operation}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class Spinner:
    """Single-line, in-place progress indicator for long synchronous calls.

    While active it animates one terminal line (rewritten with ``\\r``) showing a
    spinner frame, the current phase label, and the elapsed time, instead of
    emitting a fresh ``print`` for every phase/attempt. A daemon thread keeps it
    ticking while the calling thread blocks in a synchronous LLM request. Because
    generation is uncapped (no known total), elapsed time + motion is the honest
    signal -- there is no meaningful percentage to show.

    On exit it leaves a **persistent** one-line summary showing the total
    duration and a per-phase breakdown (aggregated by label, in first-seen
    order), so the timing stays on screen after the answer prints::

        ✓ 2m05s total  ·  Synthesizing 78s  ·  Reviewing 32s  ·  Refining 15s

    Degrades gracefully: only animates on an interactive TTY. When stdout is
    redirected (pipes, tests, non-interactive runs) it prints one plain line per
    distinct phase, then the same summary. Never raises into the caller.

    Usage::

        with Spinner("Synthesizing answer") as spin:
            answer = llm.chat(...)
            spin.update("Reviewing answer")
            ok = review(...)
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str, stream=None, interval: float = 0.1):
        self._stream = stream if stream is not None else sys.stdout
        self._label = label
        self._interval = interval
        self._start = time.monotonic()
        self._phase_start = self._start
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_plain: Optional[str] = None
        # Per-phase durations aggregated by label, preserving first-seen order.
        self._phase_order: list[str] = [label]
        self._phase_totals: dict[str, float] = {label: 0.0}
        try:
            self._tty = bool(self._stream.isatty())
        except Exception:
            self._tty = False

    @staticmethod
    def _format_duration(seconds: float, force_minutes: bool = False) -> str:
        seconds = max(0.0, seconds)
        if force_minutes or seconds >= 60:
            minutes, secs = divmod(int(round(seconds)), 60)
            return f"{minutes}m{secs:02d}s"
        if seconds < 10:
            return f"{seconds:.1f}s"
        return f"{seconds:.0f}s"

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            pass

    def _emit_plain(self) -> None:
        if self._label != self._last_plain:
            self._last_plain = self._label
            self._write(f"[jarvis] {self._label}...\n")

    def _close_phase(self) -> None:
        now = time.monotonic()
        self._phase_totals[self._label] = (
            self._phase_totals.get(self._label, 0.0) + (now - self._phase_start)
        )
        self._phase_start = now

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            elapsed = time.monotonic() - self._start
            self._write(
                f"\r\033[K{frame} {self._label}... {self._format_duration(elapsed)}"
            )
            i += 1
            self._stop.wait(self._interval)

    def _summary_line(self) -> str:
        total = time.monotonic() - self._start
        # When the total crosses a minute, render every step in the same m/s
        # format so the breakdown stays visually consistent with the total.
        force_minutes = total >= 60
        parts = [
            f"{label} {self._format_duration(self._phase_totals.get(label, 0.0), force_minutes)}"
            for label in self._phase_order
        ]
        if len(parts) == 1:
            # Single phase: total == that phase, so don't repeat it.
            return f"✓ {self._phase_order[0]} {self._format_duration(total, force_minutes)}"
        breakdown = "  ·  ".join(parts)
        return f"✓ {self._format_duration(total, force_minutes)} total  ·  {breakdown}"

    def update(self, label: str) -> None:
        """Switch the phase label shown on the live line, banking the elapsed
        time against the phase that just ended (or emit a new plain line when not
        attached to a TTY)."""
        if label == self._label:
            return
        self._close_phase()
        self._label = label
        if label not in self._phase_totals:
            self._phase_totals[label] = 0.0
            self._phase_order.append(label)
        if not self._tty:
            self._emit_plain()

    def __enter__(self) -> "Spinner":
        if self._tty:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            self._emit_plain()
        return self

    def __exit__(self, exc_type, _exc, _tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._close_phase()
        # Replace the transient animated line with a persistent result so the
        # duration stays on screen after completion or cancellation.
        if self._tty:
            self._write("\r\033[K")
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            elapsed = time.monotonic() - self._start
            self._write(f"Cancelled after {self._format_duration(elapsed)}\n")
        else:
            self._write(self._summary_line() + "\n")
        return False


def _is_port_open(host: str, port: str, timeout: float = 2.0) -> bool:
    import socket as _socket
    try:
        with _socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _drain_buffered_lines(max_lines: int = 2000) -> list[str]:
    """Return lines already buffered on stdin, without blocking.

    A pasted block reaches the terminal as one burst, so the remainder is
    readable immediately; a typed line leaves nothing behind. ``select`` with a
    zero timeout distinguishes the two without ever waiting on the user.

    Returns [] on any platform or stream that cannot be polled, and for
    non-TTY stdin, so piped and scripted input keeps its exact prior behavior.
    """
    try:
        if not sys.stdin.isatty():
            return []
    except Exception:
        return []

    try:
        import select
    except Exception:
        return []

    extra: list[str] = []
    while len(extra) < max_lines:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            break
        if not ready:
            break
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        extra.append(line.rstrip("\n"))
    return extra


def read_user_query(
    prompt: str = "You: ",
    *,
    open_marker: str = "<<<",
    close_marker: str = ">>>",
) -> str:
    """Read a query from stdin, supporting large multi-line / multi-paragraph input.

    The built-in ``input()`` returns at the first newline, so pasting text that
    contains blank lines between paragraphs only captures the first line; the
    rest spills into later prompts. Terminals also cap a single canonical-mode
    line at ``MAX_CANON`` (~1024 bytes on macOS), truncating very long lines
    before Python sees them. Neither limit is related to the LLM context window.

    Behaviour:
    - Fast path (unchanged): type a line and press Enter to submit immediately.
    - Paste / multi-line mode: enter a line that is exactly ``open_marker``
      (``<<<``) to begin, then paste or type any number of lines (blank lines
      and paragraph breaks are preserved), and finish with a line that is exactly
      ``close_marker`` (``>>>``) or end-of-input (Ctrl-D). The wrapper markers
      themselves are stripped from the returned text. Reading line-by-line also
      sidesteps the per-line terminal cap because each line is read separately.

    EOF (Ctrl-D) at the top-level prompt returns ``"exit"`` so the REPL can quit
    cleanly instead of raising ``EOFError``.

    - Bare paste (no markers): a multi-line paste arrives as ONE burst, so the
      lines after the first are already sitting in the terminal buffer when
      ``input()`` returns. Without draining them each would be consumed as its
      OWN query, turning one paste into N full RAG turns with no way to reach the
      prompt to quit. They are drained and joined into a single query instead.
      Interactive only: when stdin is not a TTY (pipes, tests, scripted input)
      the line-at-a-time contract is preserved exactly.
    """
    try:
        first_line = input(prompt)
    except EOFError:
        return "exit"

    if first_line.strip() != open_marker:
        pending = _drain_buffered_lines()
        if pending:
            print(
                f"[jarvis] Captured a {len(pending) + 1}-line paste as ONE question "
                f"(use {open_marker} ... {close_marker} to do this explicitly)."
            )
            return "\n".join([first_line, *pending])
        return first_line

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == close_marker:
            break
        lines.append(line)
    return "\n".join(lines)
