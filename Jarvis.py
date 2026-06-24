"""Jarvis — interactive RAG assistant entry point.

This script is intentionally thin: all heavy startup work (model loading,
document extraction, index building, retriever/TTS construction) lives in
``jarvis_bootstrap.build_runtime``. Importing this module has no side effects
beyond configuring log verbosity — the runtime is built only when ``build`` or
``main`` is called.
"""

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
import sys

if os.environ.get("JARVIS_DEBUG_HTTP", "0") != "1":
    logging.getLogger("httpx").setLevel(logging.WARNING)

from jarvis_bootstrap import build_runtime
from jarvis_answer import QueryRuntime, handle_query
from jarvis_io import read_user_query


def _resolve_input_dir(argv: list[str]) -> str:
    if len(argv) > 1:
        return argv[1]
    return r"./documents"


def build(argv: list[str] | None = None) -> QueryRuntime:
    """Build and return the fully-initialized `QueryRuntime`.

    `argv[1]`, when present, overrides the input document directory."""
    argv = list(sys.argv if argv is None else argv)
    print(f"Python version = {sys.version_info}")
    return build_runtime(_resolve_input_dir(argv))


def main(argv: list[str] | None = None) -> int:
    """Run the interactive question/answer REPL."""
    runtime = build(argv)
    print("\n\nAsk questions about your data! Type 'exit' to quit.")
    print("For long, multi-paragraph input, type '<<<', paste your text, "
          "then type '>>>' (or press Ctrl-D) to submit.")

    last_turn: dict[str, object] = {}
    while True:
        user_query = read_user_query("You: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            runtime.tts.say_goodbye()
            runtime.tts.cleanup()
            break
        last_turn = handle_query(user_query, runtime, last_turn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
