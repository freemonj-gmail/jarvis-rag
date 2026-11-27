"""Check that the voice says the words Jarvis sends it (audio, local Whisper).

Run when something under the voice changes -- Python, PyTorch, macOS, the
Chatterbox environment, or Kokoro -- not on every change to Jarvis: the words
themselves are graded by evaluate_speech.py. Each labelled sentence is worded
as Jarvis would word it, rendered by the real voice, transcribed by
openai/whisper-small.en (local, offline once cached), and scored by word error
rate after Whisper's English normalizer, averaged over two renders of each
sentence. Audio files are deleted after scoring.

Regression gate: pass ``--baseline <earlier report>``; exits 1 when the voice
drops or garbles clearly more words than it did then.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jarvisrag import speech
from jarvisrag.evaluation.speech import load_speech_cases
from jarvisrag.evaluation.voice import compare_voice_reports, voice_report, word_error_rate
from jarvisrag.speech.chatterbox import _expand_numbers_to_words

RENDERS = 2
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _worded(texts: list[str]) -> list[str]:
    """Word each sentence for speech exactly as Jarvis does, then free the model."""
    from jarvisrag.answer.bootstrap import _build_llm, release_llm

    llm = _build_llm()
    try:
        return [speech._resolve_spoken_tokens(text, llm) for text in texts]
    finally:
        release_llm(llm)


def _render_chatterbox(texts: list[str]) -> list[tuple[list, int]]:
    import soundfile as sf

    directory = os.environ["JARVIS_CHATTERBOX_DIR"]
    worker = subprocess.Popen(
        [os.path.join(directory, ".venv", "bin", "python"), os.path.join(_ROOT, "scripts", "chatterbox_tts_worker.py"),
         "--ref-wav", os.environ["JARVIS_CHATTERBOX_REF_WAV"]],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    event = lambda: json.loads(worker.stdout.readline())  # noqa: E731
    ready = event()
    if ready.get("event") != "ready":
        raise RuntimeError(f"Chatterbox worker did not start: {ready}")
    audio = []
    for number, text in enumerate(texts, 1):
        # Chatterbox is fed the same words as in Jarvis (see _render_utterance).
        worker.stdin.write(json.dumps({"id": number, "text": _expand_numbers_to_words(speech._normalize_spoken_semantics(text))}) + "\n")
        clips, rate = [], 24000
        while True:
            item = event()
            if item.get("event") == "chunk":
                data, rate = sf.read(item["path"], dtype="float32")
                clips.append(data if data.ndim == 1 else data.mean(1))
                os.remove(item["path"])
            elif item.get("event") in ("done", "error"):
                break
        audio.append((clips, rate))
    worker.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    worker.wait(timeout=120)
    return audio


def _render_kokoro(texts: list[str]) -> list[tuple[list, int]]:
    import numpy as np

    manager = speech.KokoroSpeaker()
    audio = []
    for text in texts:
        clips = []
        for _gs, _ps, chunk in manager.pipeline(manager.clean_text(text), voice=manager.voice, speed=manager.speed):
            if chunk is not None:
                clips.append(np.asarray(chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else chunk))
        audio.append((clips, manager.sample_rate))
    manager.cleanup()
    return audio


def _spoken_for(engine: str, text: str) -> str:
    """The words the engine is actually given for ``text``."""
    if engine == "chatterbox":
        return _expand_numbers_to_words(speech._normalize_spoken_semantics(text))
    return speech.KokoroSpeaker.clean_text(speech.KokoroSpeaker.__new__(speech.KokoroSpeaker), text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", help="labelled speech cases (JSON)")
    parser.add_argument("--output", required=True, help="where to write the report")
    parser.add_argument("--baseline", help="earlier report to compare with")
    parser.add_argument("--details", action="store_true", help="also record what was sent and heard per sentence")
    parser.add_argument("--engine", default=os.environ.get("JARVIS_TTS_ENGINE", "kokoro"), choices=("chatterbox", "kokoro"))
    args = parser.parse_args()

    import numpy as np
    from scipy.signal import resample_poly
    from transformers import pipeline

    cases = load_speech_cases(args.cases)
    worded = _worded([case.text for case in cases])
    # Each sentence is rendered twice and its error averaged: a voice samples
    # randomly, so one render can slip where the next does not.
    renders = (_render_chatterbox if args.engine == "chatterbox" else _render_kokoro)(worded * RENDERS)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device="mps")
    normalize = asr.tokenizer.normalize
    def transcript(clips, rate):
        if not clips:
            return ""
        samples = resample_poly(np.concatenate(clips), 16000, rate).astype("float32")
        # Whisper tends to skip the first words of audio that starts abruptly;
        # a short silence on each side, as in real recordings, avoids that.
        pad = np.zeros(8000, dtype="float32")
        samples = np.concatenate([pad, samples, pad])
        return normalize(asr({"raw": samples, "sampling_rate": 16000})["text"])

    scores, details = [], []
    for index, (case, text) in enumerate(zip(cases, worded)):
        sent = normalize(_spoken_for(args.engine, text))
        heard = [transcript(*renders[index + n * len(worded)]) for n in range(RENDERS)]
        scores.append((case.subject, sum(word_error_rate(sent, h) for h in heard) / RENDERS))
        details.append({"sent": sent, "heard": heard})
    report = voice_report(args.engine, scores)
    if args.details:
        for entry, detail in zip(report["per_sentence"], details):
            entry.update(detail)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    m = report["metrics"]
    print(f"Voice report written ({args.engine}): mean WER {m['mean_wer']:.3f}, misspoken {m['misspoken']}/{m['sentences']}.")
    if not args.baseline:
        return 0
    with open(args.baseline, encoding="utf-8") as handle:
        problems = compare_voice_reports(json.load(handle), report)
    for problem in problems:
        print(f"REGRESSION: {problem}")
    print("Voice check FAILED." if problems else "Voice check passed: the voice says what Jarvis sends, as before.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
