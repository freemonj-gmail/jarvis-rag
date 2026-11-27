"""Audio output: the platform audio player and the silent fallback speaker.
"""

import shutil
import sys


def _resolve_audio_player():
    """Return the platform's command-line audio player, or None if absent."""
    if sys.platform == "darwin":
        return shutil.which("afplay")
    if sys.platform.startswith("linux"):
        return shutil.which("ffplay") or shutil.which("mpg123") or shutil.which("aplay")
    if sys.platform == "win32":
        return "powershell"
    return None


def _audio_player_command(player, path):
    """Build the argv that plays `path` to completion with `player`."""
    if sys.platform == "win32":
        return [player, "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync();"]
    return [player, path]


class _SilentTTS:
    """Fallback no-op TTS used when a runtime has no speech manager attached or
    when speech is disabled via JARVIS_ENABLE_TTS=0 (e.g. smoke tests)."""

    def speak(self, _text: str) -> None:  # pragma: no cover - trivial
        pass

    def say_goodbye(self) -> None:  # pragma: no cover - trivial
        pass

    def cleanup(self) -> None:  # pragma: no cover - trivial
        pass
