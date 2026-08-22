"""Chatterbox speech engine: a Jarvis-side client for the out-of-process worker.

Chatterbox pins ``transformers==5.2.0`` / ``torch==2.6.0``, which cannot
co-resolve with Jarvis's dependency set, so synthesis runs under a separate
interpreter (see ``scripts/chatterbox_tts_worker.py``) and this class speaks to
it over stdin/stdout. It presents the same ``speak`` / ``say_goodbye`` /
``cleanup`` surface as :class:`~jarvisrag.speech.TTSManager`, so the answer
pipeline needs no engine-specific branches.

The worker synthesizes in chunks (runs of whole sentences) and delivers a wav
for each, but they are joined here and played through a single player call, so
an answer is heard the way Kokoro's is: printed first, then read back in one
continuous pass with no seam between chunks.

Any failure -- worker missing, spawn error, synthesis error -- falls back to
Kokoro, which is constructed lazily so the fallback costs nothing unless used.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime

from jarvisrag.config import logger
from jarvisrag.speech import (
    _audio_player_command,
    _normalize_spoken_semantics,
    _resolve_audio_player,
)

# Startup is a model load plus a warmup generation; measured ~31s on Apple
# Silicon, so the handshake needs a far more generous budget than a normal IPC.
_READY_TIMEOUT_SECONDS = 420.0
# A single chunk is bounded by the worker's ~40s-of-audio budget, but a cold GPU
# can stretch that several-fold before it settles into its steady rate.
_CHUNK_TIMEOUT_SECONDS = 420.0

# Digit runs not touching a letter, hyphen or slash, so identifiers like
# "MD-102770", versions like "v1.10.0" and dates like "02/13/2034" are left as
# written rather than half-converted into a mix of words and separators.
_STANDALONE_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9\-/])\d+(?![A-Za-z0-9\-/])")
# Beyond this, a digit run is a reference number rather than a quantity.
_MAX_CARDINAL_DIGITS = 9
# The fractional digits the semantics pass leaves after "point".
_DECIMAL_TAIL_RE = re.compile(r"\bpoint (\d+)")

# The worker's stderr is always kept: when it dies mid-answer this file is the
# only record of why.
_WORKER_LOG = os.path.join(tempfile.gettempdir(), "jarvis-chatterbox-worker.log")
# Speech problems that happen mid-session are recorded here instead of on the
# console, which the REPL owns.
_CLIENT_LOG = os.path.join(tempfile.gettempdir(), "jarvis-tts.log")

# Mirrors the prompt Jarvis.py reads with, so a debug notice can put it back.
_REPL_PROMPT = "You: "


def _notice(message):
    """Record a mid-session speech problem; only show it when debugging.

    Speech runs on a background thread, so printing here would land after the
    REPL has already drawn its prompt and strand the cursor on that line. The
    log keeps the detail; ``JARVIS_TTS_DEBUG=1`` surfaces it, and even then the
    message is isolated and the prompt reissued.
    """
    try:
        with open(_CLIENT_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")
    except Exception:
        pass
    if not _debug_enabled():
        return
    try:
        sys.stdout.write(f"\n\n{message}\n\n{_REPL_PROMPT}")
        sys.stdout.flush()
    except Exception:
        pass


def _debug_enabled() -> bool:
    return os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"


def _concatenate_wavs(paths):
    """Join chunk wavs into one file, or return the first path if joining fails.

    The worker already trims decoder padding and inserts the breath between
    chunks, so a plain concatenation reproduces the single-file output of
    ``generate_long`` sample for sample.
    """
    try:
        import numpy as np
        import soundfile as sf

        blocks = []
        rate = None
        for path in paths:
            data, rate = sf.read(path, dtype="float32")
            blocks.append(data)
        merged = os.path.join(os.path.dirname(paths[0]), "utterance.wav")
        # Keep the chunks' own encoding so joining cannot requantize the audio.
        sf.write(merged, np.concatenate(blocks), rate, subtype=sf.info(paths[0]).subtype)
        return merged
    except Exception as error:
        _notice(f"[tts] could not join chunks, playing them in order: {error}")
        return paths[0]


def _expand_numbers_to_words(text: str) -> str:
    """Spell digit runs out as words, e.g. "184500 dollars" -> "one hundred and
    eighty-four thousand, five hundred dollars".

    Kokoro does not need this: its espeak/misaki front-end expands numerals
    itself. Chatterbox has no such stage -- ``punc_norm`` leaves digits untouched
    and the tokenizer feeds them straight to the model -- so an unexpanded amount
    is mispronounced. Runs adjacent to letters or hyphens (identifiers, model
    numbers) are left alone, and very long runs are read digit by digit because
    they are references rather than quantities.
    """
    try:
        from num2words import num2words
    except Exception:
        return text

    def spoken(match):
        digits = match.group(0)
        try:
            if len(digits) > _MAX_CARDINAL_DIGITS:
                return " ".join(num2words(int(d)) for d in digits)
            return num2words(int(digits))
        except Exception:
            return digits

    # Digits after a decimal point are read one by one ("one point four five"),
    # not as a quantity ("one point forty-five").
    text = _DECIMAL_TAIL_RE.sub(lambda m: "point " + " ".join(m.group(1)), text)
    return _STANDALONE_NUMBER_RE.sub(spoken, text)


class ChatterboxTTSClient:
    """Speaks answers in a cloned voice via the external Chatterbox worker."""

    def __init__(self, python, worker_script, ref_wav, *, exaggeration=None,
                 cfg_weight=None, silence_ms=None, fallback_factory=None):
        self.python = python
        self.worker_script = worker_script
        self.ref_wav = ref_wav
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.silence_ms = silence_ms
        self._fallback_factory = fallback_factory
        self._fallback = None

        self.audio_player = _resolve_audio_player()
        self.sample_rate = None
        self.initialized = False

        self._process = None
        self._events: queue.Queue = queue.Queue()
        self._playback = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._closed = False
        self._stderr_log = None

        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts")
        # Load in the background: the model costs ~31s to come up and no answer
        # can be ready before then anyway.
        self._ready = threading.Event()
        self._start_failed = False
        threading.Thread(target=self._start, daemon=True).start()

    # -- worker lifecycle ---------------------------------------------------

    def _command(self):
        argv = [self.python, self.worker_script, "--ref-wav", self.ref_wav]
        if self.exaggeration is not None:
            argv += ["--exaggeration", str(self.exaggeration)]
        if self.cfg_weight is not None:
            argv += ["--cfg-weight", str(self.cfg_weight)]
        if self.silence_ms is not None:
            argv += ["--silence-ms", str(self.silence_ms)]
        return argv

    def _start(self, notify=print):
        try:
            for path in (self.python, self.worker_script, self.ref_wav):
                if not os.path.exists(path):
                    raise FileNotFoundError(path)

            # Keep the worker's stderr on disk rather than discarding it, so a
            # mid-answer death can be diagnosed after the fact.
            self._stderr_log = (
                sys.stderr if _debug_enabled()
                else open(_WORKER_LOG, "w")  # noqa: SIM115 - closed in cleanup()
            )
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_log,
                text=True,
                bufsize=1,
                # Malloc* settings inherited from the shell make libmalloc write
                # diagnostics straight to the terminal, past our redirection.
                env={k: v for k, v in os.environ.items() if not k.startswith("Malloc")},
            )
            threading.Thread(target=self._read_events, daemon=True).start()

            # Stray OS-level diagnostics name a PID but not a process; recording
            # both ends lets an unattributed line be traced to one of them.
            _notice(
                f"[tts] pids: host={os.getpid()} worker={self._process.pid} "
                f"({os.path.basename(self.python)})"
            )

            ready = self._next_event(_READY_TIMEOUT_SECONDS)
            if not ready or ready.get("event") != "ready":
                raise RuntimeError(f"unexpected handshake: {ready}")
            self.sample_rate = ready.get("sample_rate")
            self.initialized = True
            notify(
                f"[tts] Chatterbox voice ready on {ready.get('device', 'cpu')} "
                f"({self.sample_rate} Hz)."
            )
        except Exception as error:
            self._start_failed = True
            notify(f"[tts] Chatterbox unavailable ({error}); falling back to Kokoro.")
            logger.error(f"Chatterbox TTS unavailable, falling back to Kokoro: {error}")
        finally:
            self._ready.set()

    def _read_events(self):
        try:
            for line in self._process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._events.put(json.loads(line))
                except ValueError:
                    _notice(f"[tts] non-protocol line from worker: {line[:200]}")
        finally:
            self._events.put(None)

    def _next_event(self, timeout):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _drain_events(self):
        """Discard anything an aborted request left behind."""
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return

    def _send(self, payload):
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

    def _worker_alive(self):
        return self._process is not None and self._process.poll() is None

    def wait_until_ready(self, timeout=_READY_TIMEOUT_SECONDS):
        """Block until the worker has loaded (or failed), so startup output stays ordered."""
        self._ready.wait(timeout)
        return self.initialized

    # -- speaking -----------------------------------------------------------

    def _worker_failure_detail(self) -> str:
        """Explain why the worker stopped: exit status plus the tail of its log."""
        code = self._process.poll() if self._process is not None else None
        state = "still running but silent" if code is None else f"exited with code {code}"
        tail = ""
        try:
            if not _debug_enabled() and os.path.exists(_WORKER_LOG):
                with open(_WORKER_LOG, "r", errors="replace") as handle:
                    lines = [ln.strip() for ln in handle.readlines() if ln.strip()]
                if lines:
                    tail = "; last worker output: " + " | ".join(lines[-3:])
        except Exception:
            pass
        return f"{state}{tail} (full log: {_WORKER_LOG})"

    def _restart_worker(self) -> bool:
        """Bring the worker back after a death so the next answer is not degraded."""
        try:
            if self._process is not None and self._process.poll() is None:
                return True
            self._events = queue.Queue()
            self._ready = threading.Event()
            self._start_failed = False
            self.initialized = False
            self._start(notify=_notice)
            return self.initialized
        except Exception:
            return False

    def speak(self, text):
        if self._closed or not text:
            return
        try:
            self.executor.submit(self._speak_async, text)
        except RuntimeError:
            pass  # executor already shut down

    def _speak_async(self, text):
        self._ready.wait(_READY_TIMEOUT_SECONDS)
        if self._start_failed or not self._worker_alive():
            self._speak_with_fallback(text)
            return
        try:
            if not self._render_utterance(text):
                self._speak_with_fallback(text)
        except Exception as error:
            self._speak_with_fallback(text)
            _notice(f"[tts] Chatterbox synthesis failed ({error}); used Kokoro for this answer.")
            # Reinstate the worker so only this answer is degraded.
            if not self._worker_alive():
                self._restart_worker()

    def _render_utterance(self, text) -> bool:
        """Synthesize the whole answer, then play it as one continuous file.

        The worker delivers a wav per chunk as each finishes, but they are
        joined and played through a single player invocation so the answer is
        heard exactly as Kokoro's is: printed first, then read back without a
        break. Playing the chunks individually would add a process gap at every
        seam and risks running dry when a cold GPU generates slower than the
        audio plays.
        """
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            self._drain_events()
            # Say numbers, money and abbreviations correctly, but keep the
            # apostrophes and punctuation this engine phrases from.
            spoken = _expand_numbers_to_words(_normalize_spoken_semantics(text))
            self._send({"id": request_id, "text": spoken})

        paths: list[str] = []
        while True:
            event = self._next_event(_CHUNK_TIMEOUT_SECONDS)
            if event is None:
                raise RuntimeError(f"TTS worker stopped responding — {self._worker_failure_detail()}")
            # A late event from an aborted request would otherwise be read as
            # this one's result.
            if event.get("id") not in (None, request_id):
                continue
            kind = event.get("event")
            if kind == "chunk":
                paths.append(event.get("path"))
                continue
            if kind == "error":
                for path in paths:
                    self._discard(path)
                raise RuntimeError(event.get("message", "unknown worker error"))
            if kind == "done":
                break

        if not paths:
            # Zero chunks for non-empty text means synthesis produced nothing.
            return not text.strip()

        combined = _concatenate_wavs(paths) if len(paths) > 1 else paths[0]
        try:
            played = self._play(combined)
            if not played and combined is not paths[0]:
                # Joining failed or the merged file would not play; the
                # individual chunks are still valid audio.
                played = any(self._play(path) for path in paths)
            return played
        finally:
            for path in paths + [combined]:
                self._discard(path)

    def _play(self, path) -> bool:
        if not self.audio_player or not path or not os.path.exists(path):
            return False
        try:
            # Popen rather than run() so cleanup() can cut playback short.
            self._playback = subprocess.Popen(
                _audio_player_command(self.audio_player, path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._playback.wait()
            return True
        except Exception as error:
            _notice(f"[tts] playback failed for {path}: {error}")
            return False
        finally:
            self._playback = None

    @staticmethod
    def _discard(path):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass

    # -- fallback -----------------------------------------------------------

    def _speak_with_fallback(self, text):
        """Delegate to Kokoro, which applies its own spoken-text normalization.

        Chatterbox is fed the engine-agnostic prose directly, while Kokoro needs
        the orthographic scrubbing in ``TTSManager.clean_text`` -- so the raw
        text is handed over rather than anything shaped for Chatterbox.
        """
        engine = self._ensure_fallback()
        if engine is not None:
            engine.speak(text)

    def _ensure_fallback(self):
        if self._fallback is None and self._fallback_factory is not None:
            try:
                self._fallback = self._fallback_factory()
            except Exception as error:
                _notice(f"[tts] Kokoro fallback unavailable: {error}")
                self._fallback_factory = None
        return self._fallback

    # -- shutdown -----------------------------------------------------------

    def say_goodbye(self):
        if os.environ.get("JARVIS_TTS_GOODBYE", "0") != "1":
            return
        try:
            self.speak("Goodbye!")
        except Exception as error:
            logger.error(f"Error during goodbye: {error}")

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        try:
            playback = self._playback
            if playback is not None and playback.poll() is None:
                playback.terminate()

            if self.executor is not None:
                self.executor.shutdown(wait=False)
                self.executor = None

            if self._worker_alive():
                try:
                    self._send({"cmd": "shutdown"})
                    self._process.wait(timeout=10)
                except Exception:
                    self._process.kill()

            if self._fallback is not None:
                self._fallback.cleanup()

            if self._stderr_log not in (None, sys.stderr):
                self._stderr_log.close()
        except Exception as error:
            logger.error(f"Error during Chatterbox cleanup: {error}")
