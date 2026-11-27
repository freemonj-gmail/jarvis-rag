"""The Kokoro speech engine.

``KokoroSpeaker`` runs Kokoro synthesis on one background worker thread and
plays the audio through the platform player.
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from contextlib import nullcontext

import numpy as np

from jarvisrag.config import logger
from jarvisrag.iotools import suppress_external_output, suppress_external_stderr
from jarvisrag.speech.playback import _audio_player_command, _resolve_audio_player
from jarvisrag.speech.speaker import Speaker
from jarvisrag.speech.wording import _normalize_spoken_semantics


class KokoroSpeaker(Speaker):
    # Curated Kokoro voice list (en-US + en-GB). Daniel is the default
    # British male voice requested by the user.
    _KOKORO_VOICES = (
        "bm_daniel", "bm_george", "bm_lewis", "bm_fable",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_puck",
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    )
    _DEFAULT_VOICE = "bm_daniel"

    async def list_voices(self):
        """Return the curated Kokoro voice list. Kept async to preserve the
        prior public signature even though Kokoro voices are static."""
        return [{"ShortName": v} for v in self._KOKORO_VOICES]

    async def get_valid_voice(self, desired_voice):
        """Validate a requested voice against the curated Kokoro voice list.
        Falls back to bm_daniel if the requested voice is not available."""
        if desired_voice in self._KOKORO_VOICES:
            return desired_voice
        return self._DEFAULT_VOICE

    def __init__(self):
        self.enabled = os.environ.get("JARVIS_ENABLE_TTS", "1") == "1"
        if not self.enabled:
            self.initialized = False
            self.tts_available = False
            self.executor = None
            self.audio_player = None
            self.pipeline = None
            return

        try:
            desired_voice = os.environ.get("JARVIS_TTS_VOICE", self._DEFAULT_VOICE)
            try:
                self.voice = asyncio.run(self.get_valid_voice(desired_voice))
            except Exception:
                self.voice = self._DEFAULT_VOICE

            # British English voices use lang_code 'b'; American voices use 'a'.
            self.lang_code = "b" if self.voice.startswith(("bm_", "bf_")) else "a"

            # Cadence: Kokoro takes a `speed` multiplier (1.0 == default).
            # We retain `self.rate`/`self.volume` strings for backwards
            # compatibility with any external callers/inspectors, but Kokoro
            # uses `self.speed` for actual cadence control.
            self.rate = "+0%"
            self.volume = "+0%"
            try:
                self.speed = float(os.environ.get("JARVIS_TTS_SPEED", "1.0"))
            except (TypeError, ValueError):
                self.speed = 1.0

            self.audio_player = _resolve_audio_player()

            # Lazy-import + eagerly construct the Kokoro pipeline so the model
            # weights are downloaded/loaded once at startup instead of on the
            # first user query.
            #
            # Silence Kokoro/torch/phonemizer chatter unless JARVIS_TTS_DEBUG=1:
            #   * torch UserWarning about LSTM dropout / weight_norm at load
            #   * Kokoro 'Defaulting repo_id' warning (resolved by passing
            #     repo_id explicitly below)
            #   * phonemizer 'words count mismatch' warnings on every speak()
            tts_debug = os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"
            if not tts_debug:
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning, module="torch")
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
                for noisy in (
                    "phonemizer", "kokoro", "espeakng_loader",
                    "huggingface_hub", "httpx", "urllib3",
                ):
                    logging.getLogger(noisy).setLevel(logging.ERROR)
                # phonemizer's own logger setup runs lazily on first speak()
                # and resets handlers + setLevel(WARNING), overriding the line
                # above. A Filter survives that reset. Logger-level filters
                # only catch records emitted on that exact logger, so attach
                # the filter to (a) the 'phonemizer' logger (where the record
                # originates) AND (b) every handler on the root logger
                # (catches the record as it propagates up to the root format
                # that Jarvis configured: "asctime - levelname - message").
                class _DropWordsMismatch(logging.Filter):
                    def filter(self, record: logging.LogRecord) -> bool:
                        return "words count mismatch" not in record.getMessage()
                _drop = _DropWordsMismatch()
                logging.getLogger("phonemizer").addFilter(_drop)
                for _h in list(logging.getLogger().handlers):
                    _h.addFilter(_drop)
            from kokoro import KPipeline  # noqa: WPS433 — local import keeps
            # the kokoro/torch import cost off any non-TTS code path.
            kpipeline_kwargs = {"lang_code": self.lang_code, "repo_id": "hexgrad/Kokoro-82M"}
            if not tts_debug:
                # Kokoro prints a few INFO-level lines to stderr during load
                # that aren't covered by the logger silencing above; route
                # them to /dev/null so the REPL prompt stays clean.
                with suppress_external_output():
                    self.pipeline = KPipeline(**kpipeline_kwargs)
            else:
                self.pipeline = KPipeline(**kpipeline_kwargs)
            self.sample_rate = 24000  # Kokoro emits 24 kHz mono float32.

            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts")
            self.initialized = True
            self.tts_available = True
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_available = False
            self.executor = None
            self.pipeline = None

    def clean_text(self, text):
        cleaned = _normalize_spoken_semantics(text)
        cleaned = re.sub(r"[\u2010-\u2015\-]+", " ", cleaned)
        cleaned = cleaned.replace("'", "")
        cleaned = cleaned.replace("\u2019", "")
        # Keep sentence/clause punctuation (, . ; : ? !) so the Neural voice has
        # natural pause and intonation cues. Strip everything else.
        cleaned = re.sub(r"[^A-Za-z0-9\s,\.;:?!]", " ", cleaned)
        # Collapse repeats of the same punctuation (e.g. "..", "!!").
        cleaned = re.sub(r"([,\.;:?!])\1+", r"\1", cleaned)
        # Drop stray spaces before punctuation introduced by replacements above.
        cleaned = re.sub(r"\s+([,\.;:?!])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


    def cleanup(self):
        if not getattr(self, 'initialized', False):
            return
        try:
            executor = getattr(self, 'executor', None)
            if executor is not None:
                executor.shutdown(wait=True)
                self.executor = None
            self.initialized = False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _accepting(self):
        if not getattr(self, 'enabled', True):
            return False
        if not getattr(self, 'tts_available', False):
            if os.environ.get("JARVIS_TTS_DEBUG", "0") == "1":
                logger.debug("TTS is unavailable, cannot speak text.")
            return False
        return True

    def _render(self, text):
        try:
            tts_debug = os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"
            clean_text = self.clean_text(text)
            if not clean_text:
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                generation_succeeded = False
                try:
                    import soundfile as sf  # noqa: WPS433 — local import
                    audio_chunks: list[np.ndarray] = []
                    # Suppress phonemizer 'words count mismatch' warnings and
                    # any other Kokoro stderr chatter on each speak() call.
                    # stderr-only: this runs on a background worker thread, so it
                    # must NOT touch the process-global sys.stdout that the main
                    # REPL thread uses for the interactive "You: " prompt.
                    pipeline_ctx = (
                        suppress_external_stderr() if not tts_debug
                        else nullcontext()
                    )
                    with pipeline_ctx:
                        for _gs, _ps, audio in self.pipeline(
                            clean_text,
                            voice=self.voice,
                            speed=self.speed,
                        ):
                            if audio is None:
                                continue
                            # Kokoro yields torch tensors; convert defensively.
                            try:
                                audio_np = audio.detach().cpu().numpy()
                            except AttributeError:
                                audio_np = np.asarray(audio)
                            audio_chunks.append(audio_np)
                    if audio_chunks:
                        arr = np.concatenate(audio_chunks)
                        sf.write(tmp_file.name, arr, self.sample_rate)
                        generation_succeeded = True
                    elif tts_debug:
                        logger.debug("Kokoro produced no audio chunks for input.")
                except Exception as e:
                    if tts_debug:
                        logger.debug(f"TTS generation failed: {e}")
                playback_succeeded = False
                if generation_succeeded and self.audio_player:
                    try:
                        subprocess.run(
                            _audio_player_command(self.audio_player, tmp_file.name),
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        playback_succeeded = True
                    except Exception as e:
                        if tts_debug:
                            logger.debug(f"Audio playback failed with player {self.audio_player}: {e}")
                elif generation_succeeded and tts_debug:
                    logger.debug(f"Speech audio generated and saved to {tmp_file.name}")

                if tts_debug:
                    try:
                        stat = os.stat(tmp_file.name)
                        logger.debug(f"TTS temp file: {tmp_file.name} size={stat.st_size}")
                    except Exception:
                        pass

                if playback_succeeded or not tts_debug:
                    try:
                        os.unlink(tmp_file.name)
                    except Exception as e:
                        if tts_debug:
                            logger.debug(f"Failed to clean up temporary TTS file: {e}")
                elif tts_debug:
                    logger.debug(f"Left TTS temp file for inspection: {tmp_file.name}")
        except Exception as e:
            if os.environ.get("JARVIS_TTS_DEBUG", "0") == "1":
                logger.debug(f"Failed to generate or play speech: {e}")
