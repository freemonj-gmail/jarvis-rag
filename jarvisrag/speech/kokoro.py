"""The Kokoro speech engine and the engine-independent spoken-semantics pass.

``TTSManager`` runs Kokoro synthesis on one background worker thread and plays
the audio through the platform player.
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
from jarvisrag.speech.wording import (
    _cardinal_words,
    _DOLLAR_SCALE_SUFFIX,
    _normalize_for_speech,
    _SCALE_WORDS,
)
from jarvisrag.speech.playback import _audio_player_command, _resolve_audio_player


class TTSManager:
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

    # Words that, when present in the same sentence as a comma-grouped number,
    # strongly imply the number is a dollar amount and should be spoken as such.
    _TTS_MONEY_KEYWORDS = (
        "income", "wages", "salary", "salaries", "tax", "taxes", "withheld",
        "withholding", "deduction", "deductions", "credit", "credits", "refund",
        "refunds", "paid", "payment", "payments", "balance", "due", "owed",
        "owing", "amount", "amounts", "total", "totals", "subtotal", "fee",
        "fees", "cost", "costs", "price", "expense", "expenses", "revenue",
        "revenues", "earnings", "compensation", "interest", "dividend",
        "dividends", "contribution", "contributions", "agi", "gross",
        "adjusted", "exemption", "exemptions", "loan", "principal", "mortgage",
        "premium", "rebate", "value", "values", "dollars", "dollar",
    )

    @classmethod
    def _add_dollar_sign_in_financial_context(cls, text: str) -> str:
        """Prepend `$` to comma-grouped numbers (e.g. `453,172`, `8,577.50`)
        when their sentence mentions a money keyword. This lets the existing
        `$N` -> 'N dollars' rule fire so the spoken output says 'dollars'
        instead of just reading a bare number.

        We only touch comma-grouped numbers (>= 1,000 with thousands separator)
        to avoid mangling years, line numbers, phone digits, dates, etc. Numbers
        already written with the word "dollar(s)" are skipped so we don't produce
        "dollars dollars"."""
        money_pattern = re.compile(
            r"(?<![\w$,])(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)(?!\w)(?![,\d])(?!\s*[Dd]ollars?\b)"
        )
        keywords = cls._TTS_MONEY_KEYWORDS
        # Split on sentence boundaries while keeping the boundary characters so
        # we can rejoin without altering pause cues.
        parts = re.split(r"(?<=[.!?])\s+", text)
        rewritten: list[str] = []
        for sentence in parts:
            lowered = sentence.lower()
            has_money_word = any(kw in lowered for kw in keywords)
            if has_money_word:
                sentence = money_pattern.sub(r"$\1", sentence)
            rewritten.append(sentence)
        return " ".join(rewritten)

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

    @staticmethod
    def _spoken_dollar_amount(match):
        whole = match.group(1).replace(",", "")
        frac = match.group(2) or ""
        scale_word = _SCALE_WORDS.get(
            (match.group("abbr") or match.group("word") or "").lower(), ""
        )
        if scale_word:
            # A scaled amount: the decimal is a fractional multiplier, not cents
            # ("$4.5 million" -> "4 point 5 million dollars").
            number = f"{whole} point {frac}" if frac else whole
            return f"{number} {scale_word} dollars"
        cents_raw = frac.ljust(2, "0")[:2]
        cents = int(cents_raw) if cents_raw else 0
        # Spell the amount out so the engine never reads a 4-digit figure as a year.
        dollars = _cardinal_words(whole)
        if cents:
            return f"{dollars} dollars and {_cardinal_words(str(cents))} cents"
        return f"{dollars} dollars"

    def say_goodbye(self):
        if not getattr(self, 'initialized', False):
            return
        if os.environ.get("JARVIS_TTS_GOODBYE", "0") != "1":
            return
        try:
            self.speak("Goodbye!")
        except Exception as e:
            logger.error(f"Error during goodbye: {e}")

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

    def speak(self, text):
        if not getattr(self, 'enabled', True):
            return
        if not getattr(self, 'tts_available', False):
            if os.environ.get("JARVIS_TTS_DEBUG", "0") == "1":
                logger.debug("TTS is unavailable, cannot speak text.")
            return
        executor = getattr(self, 'executor', None)
        if executor is None:
            return
        try:
            executor.submit(self._speak_async, text)
        except RuntimeError:
            # Executor already shut down; ignore.
            pass

    def _speak_async(self, text):
        try:
            tts_debug = os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"
            # Deferred wording runs here, on the speech thread, not the REPL's.
            if callable(text):
                text = text()
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


def _normalize_spoken_semantics(text: str) -> str:
    """Rewrite text into the words a speech engine should actually say.

    Covers what a listener would otherwise hear wrong regardless of engine:
    context-dependent abbreviations (Dr., Jr., III), URLs, money, percentages
    and decimals. Purely orthographic scrubbing -- stripping apostrophes and
    punctuation -- is deliberately excluded, because that is a Kokoro/espeak
    front-end requirement and degrades engines that use punctuation for phrasing.
    """
    cleaned = str(text or "")
    # Expand abbreviations before any punctuation/hyphen stripping downstream, so
    # the rules can still see periods and hyphens ("Dr.", "J-R", "I-I-I").
    cleaned = _normalize_for_speech(cleaned)
    cleaned = re.sub(r"https?://\S+", " link omitted ", cleaned)
    # Recognize money-shaped numbers in financial sentences (no `$` needed).
    cleaned = TTSManager._add_dollar_sign_in_financial_context(cleaned)
    cleaned = re.sub(
        r"\$\s*(\d[\d,]*)(?:\.(\d{1,2}))?" + _DOLLAR_SCALE_SUFFIX,
        TTSManager._spoken_dollar_amount,
        cleaned,
    )
    cleaned = re.sub(
        r"\b(\d[\d,]*)\.(\d+)\s*%",
        lambda m: f"{m.group(1).replace(',', '')} point {m.group(2)} percent",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(\d[\d,]*)\s*%",
        lambda m: f"{m.group(1).replace(',', '')} percent",
        cleaned,
    )
    # Drop only thousands separators: in "Jan 29,2003" the comma separates a
    # day from a year, and removing it would read one large number.
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", cleaned)
    cleaned = re.sub(r"(?<=\d),(?=\d)", ", ", cleaned)
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", " point ", cleaned)
    return cleaned
