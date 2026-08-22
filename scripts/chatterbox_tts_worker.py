"""Long-lived Chatterbox synthesis worker, driven over stdin/stdout by Jarvis.

Chatterbox pins ``transformers==5.2.0`` / ``torch==2.6.0``, which cannot
co-resolve with Jarvis's dependency set, so it runs as a separate process under
the Chatterbox virtualenv's interpreter while Jarvis stays on its own stack.

Launched by ``jarvisrag.speech`` roughly as::

    <chatterbox venv>/bin/python scripts/chatterbox_tts_worker.py \
        --ref-wav "reference-voice.wav"

The model and the reference-voice conditionals are loaded ONCE at startup and
held warm; measured cost is ~12s + ~2.3s, which is why this is a resident
process rather than one invocation per utterance.

PROTOCOL -- newline-delimited JSON, one object per line.

  worker -> host   {"event": "ready", "sample_rate": 24000}
                   {"event": "chunk", "id": N, "index": i, "path": "/tmp/x.wav"}
                   {"event": "done",  "id": N, "chunks": i}
                   {"event": "error", "id": N, "message": "..."}
  host -> worker   {"id": N, "text": "..."}
                   {"cmd": "cancel"}
                   {"cmd": "shutdown"}

Chunks are emitted AS THEY FINISH rather than withheld until the end, leaving the
presentation choice to the host; today it joins them and plays the answer as one
continuous file. A chunk is a run of whole sentences (up to ~510 chars, so tens
of seconds of audio), never a word or phrase, and the seam handling below matches
``generate_long`` -- so the joined result is identical to the single-file path.

Chunking is delegated to ``chatterbox.long_text`` untouched: its ~510-char
budget is calibrated against ``MIN_SPEECH_TOKENS_PER_CHAR``, which holds off EOS
to prevent long-text truncation. Re-splitting the text here would silently
invalidate that calibration, so the host sends whole text and this worker never
imposes a budget of its own.
"""

from __future__ import annotations

# chatterbox is installed in the worker's own virtualenv, never in Jarvis's, so
# it is unresolvable from this workspace by design.
# pyright: reportMissingImports=false

import argparse
import json
import os
import queue
import sys
import tempfile
import threading


def redirect_library_output():
    """Reserve stdout for the protocol and send library chatter to stderr.

    torch, HuggingFace and the sampler progress bar all write to the standard
    streams; a single stray line on fd 1 would corrupt the JSON stream. The real
    stdout is duplicated to a private descriptor and fd 1 is pointed at stderr,
    so anything a library prints is merely logged rather than parsed.
    """
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")


class Worker:
    def __init__(self, args):
        self.args = args
        self.protocol = redirect_library_output()
        self.cancelled = threading.Event()
        self.requests: queue.Queue = queue.Queue()
        self.shutdown = threading.Event()
        self.tempdir = tempfile.mkdtemp(prefix="jarvis-chatterbox-")
        self.model = None
        self.chunk_text = None
        self.trim_trailing_silence = None
        self.silence = None

    def emit(self, **payload):
        self.protocol.write(json.dumps(payload) + "\n")
        self.protocol.flush()

    def load(self):
        import torch

        device = self.args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # Checkpoints are CPU-saved; without this each torch.load needs an
        # explicit map_location on Apple Silicon.
        original_load = torch.load

        def patched_load(*a, **kw):
            kw.setdefault("map_location", torch.device(device))
            return original_load(*a, **kw)

        torch.load = patched_load

        from chatterbox.tts import ChatterboxTTS
        # _trim_trailing_silence is private but reused deliberately: copying it
        # would let the seam behaviour drift from generate_long's.
        from chatterbox.long_text import chunk_text, _trim_trailing_silence

        self.chunk_text = chunk_text
        self.trim_trailing_silence = _trim_trailing_silence
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.model.prepare_conditionals(
            self.args.ref_wav, exaggeration=self.args.exaggeration
        )
        self.silence = torch.zeros(1, int(self.model.sr * self.args.silence_ms / 1000))
        if self.args.warmup:
            self._warmup()
        self.emit(event="ready", sample_rate=self.model.sr, device=device)

    def _warmup(self):
        """Generate throwaway audio so the first real utterance is not the slow one.

        Measured on this machine: sampling runs at 4-12 it/s for roughly the
        first few minutes of a fresh process, then settles at 36-42 it/s and
        stays there. Discarding some synthesis up front moves that ramp off the
        user's first answer; failures are ignored because warmup is optional.
        """
        try:
            self.model.generate(
                self.args.warmup_text,
                exaggeration=self.args.exaggeration,
                cfg_weight=self.args.cfg_weight,
            )
        except Exception:
            pass

    def read_stdin(self):
        """Feed requests to the main loop; act on control messages immediately."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            command = message.get("cmd")
            if command == "shutdown":
                self.shutdown.set()
                self.requests.put(None)
                return
            if command == "cancel":
                # Set outside the queue so it is observed while a request is
                # already being synthesized.
                self.cancelled.set()
                while not self.requests.empty():
                    try:
                        self.requests.get_nowait()
                    except queue.Empty:
                        break
                continue
            self.requests.put(message)
        self.shutdown.set()
        self.requests.put(None)

    def write_wav(self, wav, request_id, index):
        import soundfile as sf

        path = os.path.join(self.tempdir, f"{request_id}-{index}.wav")
        sf.write(path, wav.squeeze(0).cpu().numpy(), self.model.sr)
        return path

    def synthesize(self, request):
        request_id = request.get("id")
        text = (request.get("text") or "").strip()
        if not text:
            self.emit(event="done", id=request_id, chunks=0)
            return

        generate_kwargs = {
            "exaggeration": self.args.exaggeration,
            "cfg_weight": self.args.cfg_weight,
        }

        import torch

        chunks = self.chunk_text(text)
        last = len(chunks) - 1
        emitted = 0

        for index, chunk in enumerate(chunks):
            if self.cancelled.is_set():
                break
            wav = self.model.generate(chunk, **generate_kwargs)
            if index != last:
                # Mirrors generate_long: drop the decoder's trailing padding so
                # the inserted breath is the only gap between chunks.
                wav = self.trim_trailing_silence(wav, self.model.sr)
                if self.args.silence_ms > 0:
                    wav = torch.cat([wav, self.silence], dim=1)
            if self.cancelled.is_set():
                break
            self.emit(
                event="chunk",
                id=request_id,
                index=index,
                path=self.write_wav(wav, request_id, index),
            )
            emitted += 1

        self.emit(event="done", id=request_id, chunks=emitted)

    def run(self):
        self.load()
        reader = threading.Thread(target=self.read_stdin, daemon=True)
        reader.start()

        while not self.shutdown.is_set():
            request = self.requests.get()
            if request is None:
                break
            self.cancelled.clear()
            try:
                self.synthesize(request)
            except Exception as error:  # never let one utterance kill the worker
                self.emit(event="error", id=request.get("id"), message=str(error))
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-wav", required=True, help="Reference voice clip.")
    parser.add_argument("--device", default=None, help="Defaults to mps, else cpu.")
    # Defaults match the settings the reference voice was validated with.
    parser.add_argument("--exaggeration", type=float, default=0.3)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--silence-ms", type=int, default=125)
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip the startup warmup generation.",
    )
    parser.add_argument(
        "--warmup-text",
        default=(
            "Good evening. Systems are online and standing by, ready whenever "
            "you are. All indexed sources have been loaded successfully."
        ),
        help="Throwaway text used to move the cold-start ramp off the first answer.",
    )
    args = parser.parse_args(argv)

    worker = Worker(args)
    try:
        return worker.run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        worker.emit(event="error", id=None, message=f"worker failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
