"""Measure Chatterbox synthesis speed to decide the Jarvis TTS integration shape.

Runs OUTSIDE the Jarvis environment: Chatterbox pins ``transformers==5.2.0`` /
``torch==2.6.0``, which cannot co-resolve with Jarvis's stack, so this script is
executed by the Chatterbox venv's interpreter (see the usage block below).

What it answers:
  * real-time factor (RTF = wall time / audio seconds). RTF < 1 is faster than
    realtime; RTF > 1 means a long answer takes longer to speak than to hear.
  * time-to-first-audio, per chunk, which is what a streaming design would
    actually deliver versus waiting for the whole answer.
  * one-time costs (model load, reference-clip embedding) that decide whether to
    keep a worker warm and whether to cache the voice conditionals to disk.
  * peak RSS, to size the memory cost of a resident worker.

Usage (from the chatterbox checkout, on the branch carrying the truncation fix):

    cd <chatterbox checkout>
    .venv/bin/python <jarvis-rag>/scripts/bench_chatterbox_tts.py \
        --ref-wav "reference-voice.wav"

Add ``--model turbo`` or ``--model nano`` to compare variants. Text samples are
neutral filler on purpose so no private corpus content is ever synthesized.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import resource
import statistics
import sys
import time

import torch


# Neutral prose sized to the shapes Jarvis actually speaks. The long cases matter
# most: they exceed the ~510-char single-call budget and therefore get chunked.
SAMPLES: list[tuple[str, str]] = [
    (
        "short",
        "The requested figure appears in the summary section of the report.",
    ),
    (
        "medium",
        "The quarterly summary lists three separate line items. The first covers "
        "routine operating costs, the second covers equipment maintenance, and "
        "the third covers a one-time adjustment recorded at the end of the "
        "period. Each entry is supported by an itemized schedule.",
    ),
    (
        "long",
        "The document sets out the general method used to determine the final "
        "amount. It begins by establishing the base value, then applies the "
        "adjustment rate that corresponds to the relevant category. Where more "
        "than one rate could apply, the schedule directs the reader to use the "
        "one that produces the smaller result. After the adjustment is applied, "
        "the total is divided into equal monthly portions and rounded down to "
        "the nearest whole unit. Several worked examples follow the main text, "
        "each showing the calculation carried through from start to finish so "
        "the method can be checked against a known result.",
    ),
    (
        "very_long",
        "The overview begins by describing the purpose of the process and the "
        "conditions under which it applies. It then explains that eligibility "
        "depends on two separate tests, both of which must be satisfied before "
        "any calculation is performed. The first test considers the length of "
        "the qualifying period, measured in whole years and months. The second "
        "test considers whether the required threshold has been met at any point "
        "during that period. Once both tests are satisfied, the base figure is "
        "established using the average of the three highest consecutive years. "
        "That average is then multiplied by the applicable rate and by the total "
        "number of qualifying years. Where an early commencement applies, a "
        "reduction is taken for each month below the standard threshold. The "
        "resulting annual figure is divided by twelve to produce the monthly "
        "amount, and the result is rounded down to the next lower whole unit. "
        "The remaining sections describe the exceptions to the general method, "
        "including the treatment of partial periods, the effect of an election "
        "made at the time of commencement, and the adjustments that apply when a "
        "survivor benefit has been chosen. Worked examples appear at the end.",
    ),
]


def peak_rss_gb() -> float:
    """Peak resident set size. macOS reports bytes; Linux reports kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak *= 1024
    return peak / (1024 ** 3)


def load_model(variant: str, device: str):
    """Import and construct the requested Chatterbox variant."""
    # Checkpoints are saved on CPU; without this every torch.load needs an
    # explicit map_location on Apple Silicon.
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device(device))
        return original_load(*args, **kwargs)

    torch.load = patched_load

    if variant == "standard":
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=device)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_pretrained(device=device, nano=(variant == "nano"))


def supported_kwargs(model, requested: dict) -> dict:
    """Drop kwargs this variant's generate() does not accept.

    Turbo/Nano ignore cfg_weight, exaggeration and min_p (they log a warning),
    while the standard model accepts them -- so the caller cannot use one fixed
    kwarg set across variants.
    """
    accepted = set(inspect.signature(model.generate).parameters)
    return {k: v for k, v in requested.items() if k in accepted and v is not None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-wav", required=True, help="Reference voice clip.")
    parser.add_argument(
        "--model",
        default="standard",
        choices=("standard", "turbo", "nano"),
        help="Chatterbox variant to benchmark.",
    )
    parser.add_argument("--device", default=None, help="Defaults to mps, else cpu.")
    parser.add_argument("--exaggeration", type=float, default=0.3)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--json-out", default=None, help="Write raw results here.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Runs per case; reports the median so one slow chunk cannot skew it.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated subset of: " + ",".join(name for name, _ in SAMPLES),
    )
    args = parser.parse_args(argv)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    from chatterbox.long_text import chunk_text

    print(f"device={device}  variant={args.model}  ref={args.ref_wav}")
    print("loading model...", flush=True)

    started = time.perf_counter()
    model = load_model(args.model, device)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model.prepare_conditionals(args.ref_wav, exaggeration=args.exaggeration)
    conditionals_seconds = time.perf_counter() - started

    print(f"model load          {load_seconds:7.1f}s")
    print(f"prepare_conditionals{conditionals_seconds:7.1f}s")

    gen_kwargs = supported_kwargs(
        model,
        {
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
        },
    )

    # The first call on MPS pays kernel/graph warmup; measuring it as if it were
    # steady state would overstate every later number.
    print("warmup call...", flush=True)
    started = time.perf_counter()
    model.generate("Warming up the synthesis path.", **gen_kwargs)
    warmup_seconds = time.perf_counter() - started
    print(f"first call (warmup) {warmup_seconds:7.1f}s  <- excluded from results")

    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = [(n, t) for n, t in SAMPLES if not selected or n in selected]

    results = []
    print()
    header = (
        f"{'case':<10} {'chars':>6} {'chunks':>7} {'audio_s':>8} "
        f"{'RTF_med':>8} {'RTF_min':>8} {'RTF_max':>8} {'first_s':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, text in cases:
        chunks = chunk_text(text)
        runs = []

        for _ in range(args.repeat):
            chunk_times: list[float] = []
            audio_samples = 0
            case_started = time.perf_counter()
            for chunk in chunks:
                chunk_started = time.perf_counter()
                wav = model.generate(chunk, **gen_kwargs)
                chunk_times.append(time.perf_counter() - chunk_started)
                audio_samples += wav.shape[-1]
            wall = time.perf_counter() - case_started
            audio_seconds = audio_samples / model.sr
            runs.append(
                {
                    "wall_seconds": wall,
                    "audio_seconds": audio_seconds,
                    "rtf": wall / audio_seconds if audio_seconds else float("nan"),
                    "first_chunk_seconds": chunk_times[0] if chunk_times else float("nan"),
                    "chunk_seconds": chunk_times,
                }
            )

        rtfs = sorted(r["rtf"] for r in runs)
        rtf_med = statistics.median(rtfs)
        audio_med = statistics.median(r["audio_seconds"] for r in runs)
        first_med = statistics.median(r["first_chunk_seconds"] for r in runs)
        wall_med = statistics.median(r["wall_seconds"] for r in runs)

        print(
            f"{name:<10} {len(text):>6} {len(chunks):>7} {audio_med:>8.1f} "
            f"{rtf_med:>8.2f} {rtfs[0]:>8.2f} {rtfs[-1]:>8.2f} {first_med:>8.1f}"
        )
        results.append(
            {
                "case": name,
                "chars": len(text),
                "chunks": len(chunks),
                "audio_seconds": round(audio_med, 2),
                "wall_seconds": round(wall_med, 2),
                "rtf_median": round(rtf_med, 3),
                "rtf_min": round(rtfs[0], 3),
                "rtf_max": round(rtfs[-1], 3),
                "first_chunk_seconds": round(first_med, 2),
                "runs": [
                    {
                        "rtf": round(r["rtf"], 3),
                        "wall_seconds": round(r["wall_seconds"], 2),
                        "chunk_seconds": [round(t, 2) for t in r["chunk_seconds"]],
                    }
                    for r in runs
                ],
            }
        )

    print()
    print(f"peak RSS {peak_rss_gb():.2f} GB")

    # A single slow chunk (background load, thermal throttling) can be 10x the
    # normal rate and would otherwise drive the verdict, so judge on the median
    # and report instability separately instead of designing around an outlier.
    spread = max(
        (r["rtf_max"] / r["rtf_min"] for r in results if r["rtf_min"] > 0),
        default=1.0,
    )
    within_case_spread = max(
        (
            max(run["chunk_seconds"]) / min(run["chunk_seconds"])
            for r in results
            for run in r["runs"]
            if len(run["chunk_seconds"]) > 1 and min(run["chunk_seconds"]) > 0
        ),
        default=1.0,
    )

    multi = [r for r in results if r["chunks"] > 1]
    if multi:
        typical = statistics.median([r["rtf_median"] for r in results])
        saved = max(r["wall_seconds"] - r["first_chunk_seconds"] for r in multi)
        print(f"median RTF {typical:.2f}  (<1 means synthesis outruns playback)")
        print(f"streaming would start speaking up to {saved:.1f}s sooner")

        if args.repeat > 1 and spread > 2.0:
            print(
                f"WARNING: RTF varied {spread:.1f}x across repeats -- unstable machine "
                "state (background load/thermal). Treat the verdict as provisional."
            )
        if within_case_spread > 5.0:
            print(
                f"WARNING: chunk times varied {within_case_spread:.1f}x within a single "
                "run of one case; that is contamination, not model behavior."
            )

        if typical >= 1.0:
            print("VERDICT: slower than realtime -> streaming playback is required.")
        elif saved > 3.0:
            print("VERDICT: faster than realtime, but streaming still cuts the initial wait.")
        else:
            print("VERDICT: fast enough that non-streaming is fine.")

    if args.json_out:
        payload = {
            "device": device,
            "variant": args.model,
            "model_load_seconds": round(load_seconds, 2),
            "prepare_conditionals_seconds": round(conditionals_seconds, 2),
            "warmup_seconds": round(warmup_seconds, 2),
            "repeat": args.repeat,
            "peak_rss_gb": round(peak_rss_gb(), 2),
            "sample_rate": model.sr,
            "results": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
