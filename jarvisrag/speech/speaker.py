"""The interface every speech engine presents, and the silent speaker.

``speak`` queues an answer on the engine's one background thread and returns at
once, so the printed answer and the next prompt are never held up. The text may
be a zero-argument callable; it is then worded on the speech thread, because the
context step that words it needs a model call. Each engine renders the words in
``_render``.
"""

import os

from jarvisrag.config import logger


def _tts_debug_enabled() -> bool:
    """JARVIS_TTS_DEBUG=1 turns on detailed speech diagnostics (read when used)."""
    return os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"


class Speaker:
    """Base for the speech engines: speak, say_goodbye, cleanup."""

    executor = None

    def _accepting(self) -> bool:
        """Whether the engine can take new speech now."""
        return True

    def speak(self, text) -> None:
        if not text or not self._accepting():
            return
        executor = self.executor
        if executor is None:
            return
        try:
            executor.submit(self._speak_async, text)
        except RuntimeError:
            pass  # executor already shut down

    def _speak_async(self, text) -> None:
        # Deferred wording runs here, on the speech thread, not the REPL's.
        if callable(text):
            try:
                text = text()
            except Exception as error:
                self._wording_failed(error)
                return
            if not text:
                return
        self._render(text)

    def _wording_failed(self, error: Exception) -> None:
        if _tts_debug_enabled():
            logger.debug(f"Failed to word speech: {error}")

    def _render(self, text: str) -> None:
        raise NotImplementedError

    def say_goodbye(self) -> None:
        if os.environ.get("JARVIS_TTS_GOODBYE", "0") != "1" or not self._accepting():
            return
        try:
            self.speak("Goodbye!")
        except Exception as error:
            logger.error(f"Error during goodbye: {error}")

    def cleanup(self) -> None:
        pass


class SilentSpeaker(Speaker):
    """No-op speaker used when a runtime has no speech engine attached or when
    speech is disabled via JARVIS_ENABLE_TTS=0 (e.g. smoke tests)."""

    def _accepting(self) -> bool:
        return False
