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
