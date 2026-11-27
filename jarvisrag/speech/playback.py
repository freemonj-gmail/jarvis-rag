"""Audio output: the platform's command-line audio player.
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
