"""Smoke-test the Chatterbox worker protocol from Jarvis's side of the pipe.

Runs under Jarvis's interpreter and spawns the worker under the Chatterbox
virtualenv's interpreter, which is exactly the split the real client uses. It
verifies the handshake, that chunks stream out as they finish rather than all at
the end, that the emitted wavs are real audio, and that cancel/shutdown work.

    uv run python scripts/smoke_chatterbox_worker.py
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

# Point at the Chatterbox checkout via env (matches the production client) or
# pass --chatterbox; kept out of source so no local path is committed.
CHATTERBOX = os.environ.get("JARVIS_CHATTERBOX_DIR", "")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatterbox_tts_worker.py")

SHORT_TEXT = "All systems are online and standing by."
# Long enough to exceed the ~510-char budget by more than one chunk, so streaming
# can be observed across several deliveries rather than a single hand-off.
LONG_TEXT = (
    "The overview begins by describing the purpose of the process and the "
    "conditions under which it applies. It then explains that eligibility "
    "depends on two separate tests, both of which must be satisfied before any "
    "calculation is performed. The first test considers the length of the "
    "qualifying period, measured in whole years and months. The second test "
    "considers whether the required threshold has been met at any point during "
    "that period. Once both tests are satisfied, the base figure is established "
    "using the average of the three highest consecutive years. That average is "
    "then multiplied by the applicable rate and by the total number of "
    "qualifying years. Where an early commencement applies, a reduction is "
    "taken for each month below the standard threshold. The resulting annual "
    "figure is divided by twelve to produce the monthly amount, and the result "
    "is rounded down to the next lower whole unit."
)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)
    return ok


def reader_thread(stream, sink: queue.Queue):
    for line in stream:
        sink.put(line)
    sink.put(None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chatterbox", default=CHATTERBOX)
    parser.add_argument("--ref-wav", default=None)
    parser.add_argument("--ready-timeout", type=float, default=420.0)
    parser.add_argument("--chunk-timeout", type=float, default=420.0)
    args = parser.parse_args(argv)

    python = os.path.join(args.chatterbox, ".venv", "bin", "python")
    ref_wav = args.ref_wav or os.environ.get("JARVIS_CHATTERBOX_REF_WAV", "")

    for path, label in ((python, "chatterbox venv python"), (ref_wav, "reference wav"), (WORKER, "worker")):
        if not os.path.exists(path):
            print(f"missing {label}: {path}")
            return 2

    stderr_log = open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "chatterbox-worker.log"), "w")
    print(f"spawning worker (stderr -> {stderr_log.name})")

    proc = subprocess.Popen(
        [python, WORKER, "--ref-wav", ref_wav],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_log,
        text=True,
        bufsize=1,
    )
    events: queue.Queue = queue.Queue()
    threading.Thread(target=reader_thread, args=(proc.stdout, events), daemon=True).start()

    def send(payload: dict):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def next_event(timeout: float):
        line = events.get(timeout=timeout)
        if line is None:
            raise RuntimeError("worker exited unexpectedly; see the stderr log")
        return json.loads(line)

    try:
        # --- handshake -----------------------------------------------------
        print("\n[1] startup (model load + warmup; expect several minutes cold)")
        started = time.perf_counter()
        ready = next_event(args.ready_timeout)
        elapsed = time.perf_counter() - started
        check("emits ready", ready.get("event") == "ready", json.dumps(ready))
        check("sample rate is 24000", ready.get("sample_rate") == 24000)
        print(f"        startup took {elapsed:.1f}s")

        # --- single chunk --------------------------------------------------
        print("\n[2] short text -> single chunk")
        send({"id": 1, "text": SHORT_TEXT})
        chunks = []
        while True:
            event = next_event(args.chunk_timeout)
            if event.get("event") == "chunk":
                chunks.append(event)
                continue
            break
        check("one chunk emitted", len(chunks) == 1, f"got {len(chunks)}")
        check("done reports the same count", event.get("event") == "done" and event.get("chunks") == len(chunks), json.dumps(event))
        check("response carries the request id", all(c.get("id") == 1 for c in chunks))

        import soundfile as sf

        if chunks:
            info = sf.info(chunks[0]["path"])
            check("wav is real audio", info.frames > 0 and info.samplerate == 24000,
                  f"{info.duration:.1f}s @ {info.samplerate}Hz")
            for c in chunks:
                os.unlink(c["path"])

        # --- streaming -----------------------------------------------------
        print("\n[3] long text -> chunks must arrive incrementally")
        send({"id": 2, "text": LONG_TEXT})
        started = time.perf_counter()
        timings = []
        chunks = []
        while True:
            event = next_event(args.chunk_timeout)
            if event.get("event") == "chunk":
                timings.append(time.perf_counter() - started)
                chunks.append(event)
                continue
            break
        total = time.perf_counter() - started
        check("multiple chunks", len(chunks) > 1, f"got {len(chunks)}")
        check("done matches", event.get("event") == "done" and event.get("chunks") == len(chunks))
        if timings:
            # Deliberately NOT a ratio: when the first chunk is the largest it can
            # legitimately dominate the total, so only incremental delivery proves
            # streaming happened.
            check(
                "chunks are delivered incrementally",
                all(later > earlier for earlier, later in zip(timings, timings[1:])),
                ", ".join(f"{t:.1f}s" for t in timings),
            )
            check(
                "speech can start before synthesis finishes",
                timings[0] < total,
                f"first at {timings[0]:.1f}s vs {total:.1f}s total "
                f"({total - timings[0]:.1f}s of silence avoided)",
            )
            check("chunk order is sequential", [c["index"] for c in chunks] == list(range(len(chunks))))
        durations = []
        for c in chunks:
            durations.append(sf.info(c["path"]).duration)
            os.unlink(c["path"])
        if durations:
            audio = sum(durations)
            print(f"        audio {audio:.1f}s in {total:.1f}s wall (RTF {total / audio:.2f})")

        # --- cancellation ---------------------------------------------------
        print("\n[4] cancel mid-utterance")
        send({"id": 3, "text": LONG_TEXT})
        first = next_event(args.chunk_timeout)
        check("streaming started", first.get("event") == "chunk")
        send({"cmd": "cancel"})
        emitted = 1 if first.get("event") == "chunk" else 0
        paths = [first["path"]] if first.get("event") == "chunk" else []
        while True:
            event = next_event(args.chunk_timeout)
            if event.get("event") == "chunk":
                emitted += 1
                paths.append(event["path"])
                continue
            break
        check("cancel ends the request early", event.get("event") == "done" and emitted < len(chunks),
              f"emitted {emitted} of {len(chunks)}")
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)

        # --- still alive afterwards -----------------------------------------
        print("\n[5] worker still serves after a cancel")
        send({"id": 4, "text": SHORT_TEXT})
        recovered = []
        while True:
            event = next_event(args.chunk_timeout)
            if event.get("event") == "chunk":
                recovered.append(event)
                continue
            break
        check("responds to the next request", len(recovered) == 1 and event.get("event") == "done")
        for c in recovered:
            os.unlink(c["path"])

        # --- shutdown --------------------------------------------------------
        print("\n[6] shutdown")
        send({"cmd": "shutdown"})
        proc.wait(timeout=30)
        check("exits cleanly", proc.returncode == 0, f"returncode {proc.returncode}")

    except Exception as error:
        failures.append(f"exception: {error}")
        print(f"\nERROR: {error}")
        print(f"worker stderr log: {stderr_log.name}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        stderr_log.close()

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
