"""Focused tests for interactive Ctrl-C request cancellation."""

import io
import os
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import Jarvis
from jarvisrag.iotools import Spinner


class RecordingTTS:
    def __init__(self) -> None:
        self.goodbye_calls = 0
        self.cleanup_calls = 0

    def say_goodbye(self) -> None:
        self.goodbye_calls += 1

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class RuntimeStub:
    def __init__(self) -> None:
        self.tts = RecordingTTS()


def run_tests() -> None:
    runtime = RuntimeStub()
    queries = iter(("cancel this", "continue", "exit"))
    prior_turns: list[dict[str, object]] = []
    prompts: list[str] = []

    def next_query(prompt: str) -> str:
        prompts.append(prompt)
        return next(queries)

    def fake_handle_query(
        query: str,
        _runtime: object,
        prior_turn: dict[str, object],
    ) -> dict[str, object]:
        del _runtime
        prior_turns.append(dict(prior_turn))
        if query == "cancel this":
            raise KeyboardInterrupt
        return {"user_query": query, "response_text": "completed"}

    output = io.StringIO()
    with (
        patch.object(Jarvis, "build", return_value=runtime),
        patch.object(Jarvis, "read_user_query", side_effect=next_query),
        patch.object(Jarvis, "handle_query", side_effect=fake_handle_query),
        redirect_stdout(output),
    ):
        assert Jarvis.main([]) == 0

    rendered = output.getvalue()
    assert "Current request cancelled" in rendered
    assert prior_turns == [{}, {}], "a cancelled turn must not replace conversation state"
    assert prompts == ["\n\nYou: "] * 3
    assert runtime.tts.goodbye_calls == 1
    assert runtime.tts.cleanup_calls == 1

    input_runtime = RuntimeStub()
    input_events = iter((KeyboardInterrupt(), "exit"))
    input_prompts: list[str] = []

    def fake_read(prompt: str) -> str:
        input_prompts.append(prompt)
        event = next(input_events)
        if isinstance(event, BaseException):
            raise event
        return event

    input_output = io.StringIO()
    with (
        patch.object(Jarvis, "build", return_value=input_runtime),
        patch.object(Jarvis, "read_user_query", side_effect=fake_read),
        redirect_stdout(input_output),
    ):
        assert Jarvis.main([]) == 0

    assert "Input cancelled" in input_output.getvalue()
    assert input_prompts == ["\n\nYou: "] * 2
    assert input_runtime.tts.cleanup_calls == 1

    spinner_output = io.StringIO()
    try:
        with Spinner("Synthesizing", stream=spinner_output):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Spinner swallowed KeyboardInterrupt")
    spinner_text = spinner_output.getvalue()
    assert "Cancelled after" in spinner_text
    assert "✓" not in spinner_text

    print("Ctrl-C cancellation tests passed")


if __name__ == "__main__":
    run_tests()