"""Speech synthesis and spoken-text normalization for Jarvis.

Owns everything on the spoken-output path, split by job:
  * ``kokoro`` -- ``TTSManager``, the Kokoro pipeline wrapper that runs synthesis
    on a single background worker thread, and the engine-independent
    spoken-semantics pass (money, percentages, abbreviations, URLs).
  * ``wording`` -- fixed rules that rewrite written forms into spoken words.
  * ``context`` -- context-aware readings for tokens whose spoken form depends
    on the sentence (one model call, number-preserving acceptance).
  * ``citations`` -- citation regexes and their removal before speaking; the
    regexes are also consumed by the printed-display formatter.
  * ``spoken`` -- the text to speak for an answer (summary or full prose).
  * ``playback`` -- the platform audio player and ``_SilentTTS``, the no-op
    fallback used when a runtime has no TTS attached.
  * ``chatterbox`` -- the Chatterbox engine client.

This package re-exports the names callers use, so ``from jarvisrag.speech
import ...`` keeps working.
"""

from jarvisrag.speech.citations import (  # noqa: F401
    _CITATION_FILE_EXT,
    _INLINE_CITATION_RE,
    _source_label_words,
    _SOURCE_REFERENCE_RE,
    _SPEECH_DOC_RUN_RE,
    _strip_inline_citations,
    _strip_label_citations,
    _strip_source_filenames_for_speech,
)
from jarvisrag.speech.wording import (  # noqa: F401
    _cardinal_words,
    _DIGIT_WORDS,
    _DOLLAR_SCALE_SUFFIX,
    _expand_slash_dates,
    _expand_time_units,
    _MONTH_YEAR_RANGE,
    _NON_NAME_BEFORE_ROMAN,
    _normalize_for_speech,
    _PER_SLASH_RE,
    _PER_UNITS,
    _ROMAN_NAME_ORDINALS,
    _SCALE_WORDS,
    _SLASH_DATE_RE,
    _spell_digits,
    _STATE_IN_ADDRESS_RE,
    _STATE_NAME_SET,
    _STATE_NAMES,
    _STATE_ZIP_RE,
    _STREET_ADDRESS_RE,
    _STREET_WORDS,
    _TIME_UNIT_RE,
    _TIME_UNIT_WORDS,
    _US_STATE_ABBR,
)
from jarvisrag.speech.context import (  # noqa: F401
    _DIGIT_NAMES,
    _LEADING_WRAPPERS,
    _number_readings,
    _reading_keeps_numbers,
    _resolve_spoken_tokens,
    _speech_llm,
    _SPEECH_TOKEN_RE,
    _speech_tokens,
    _SPOKEN_FORM_PROMPT,
    _TRAILING_WRAPPERS,
)
from jarvisrag.speech.spoken import (  # noqa: F401
    _speech_text_for_response,
    _spoken_response_text,
    _summarize_for_speech,
)
from jarvisrag.speech.playback import (  # noqa: F401
    _audio_player_command,
    _resolve_audio_player,
    _SilentTTS,
)
from jarvisrag.speech.kokoro import (  # noqa: F401
    _normalize_spoken_semantics,
    TTSManager,
)
