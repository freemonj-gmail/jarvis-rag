"""Speech-wording checks: context-aware readings of tokens whose spoken form
depends on the sentence.

Uses a mocked model, never a real LLM or audio. Run with ``make speech-test``
or directly through uv.
"""

import os
import sys
import unittest
from unittest.mock import Mock

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from tests.helpers import _patch_answer


class SpeechContextTests(unittest.TestCase):
    """Tokens whose reading depends on meaning are worded by the model, but only
    those tokens change, and never so that a number says a different value."""

    class _FakeLLM:
        def __init__(self, reply=None, error=None):
            self.additional_kwargs = {"num_predict": 256}
            self.reply, self.error, self.prompts, self.copies = reply, error, [], []

        def model_copy(self, update):
            copy = SpeechContextTests._FakeLLM(self.reply, self.error)
            copy.additional_kwargs = update["additional_kwargs"]
            copy.prompts = self.prompts
            self.copies.append(copy)
            return copy

        def chat(self, messages):
            self.prompts.append(messages[0].content)
            if self.error:
                raise self.error
            resp = Mock()
            resp.message = Mock()
            resp.message.content = self.reply
            return resp

    def test_flagged_tokens_take_the_model_reading(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        llm = self._FakeLLM('{"1|2": "two", "2|ct.": "carat", "3|1903": "nineteen oh three"}')
        out = _resolve_spoken_tokens("A 2 ct. stone bought in 1903 for $1,903 at 5% off.", llm)
        self.assertEqual(out, "A two carat stone bought in nineteen oh three for $1,903 at 5% off.")
        # Money and percentages have one reading and are left to the fixed rules.
        self.assertNotIn("|$1,903]]", llm.prompts[0])
        self.assertNotIn("|5%]]", llm.prompts[0])

    def test_reading_under_another_marker_is_not_applied(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        # The reply miscounts: "carat" is keyed to the number's marker.
        llm = self._FakeLLM('{"1|ct.": "carat", "2|1903": "nineteen oh three"}')
        self.assertEqual(_resolve_spoken_tokens("A 2 ct. stone.", llm), "A 2 ct. stone.")

    def test_message_ids_cited_from_source_labels_are_not_spoken(self):
        from jarvisrag.speech import _spoken_response_text

        labels = ["7c41d0e9b2fa4e6d8a13c5f0.emlx", "e5a2b9c07d3f61a84b2e9d17.emlx"]
        spoken = _spoken_response_text(
            "The insurer sent a renewal notice (7c41d0e9b2fa4e6d8a13c5f0, e5a2b9c07d3f61a84b2e9d17) in (2026).", labels
        )
        self.assertEqual(spoken, "The insurer sent a renewal notice in (2026).")

    def test_readings_survive_a_malformed_reply(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        # A missing comma breaks the JSON; the well-formed pairs are still used.
        llm = self._FakeLLM('{"1|1903": "nineteen oh three" "2|and": "and", "3|1911": "nineteen eleven"}')
        self.assertEqual(
            _resolve_spoken_tokens("Won in 1903 and 1911.", llm), "Won in nineteen oh three and nineteen eleven."
        )

    def test_date_written_without_a_space_keeps_its_reading(self):
        from jarvisrag.speech import _reading_keeps_numbers

        self.assertTrue(_reading_keeps_numbers("29,2003,as", "twenty ninth, two thousand three, as"))
        self.assertTrue(_reading_keeps_numbers("1,500", "fifteen hundred"))
        self.assertFalse(_reading_keeps_numbers("29,2003", "twenty nine thousand three"))

    def test_reading_in_another_script_is_rejected(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        llm = self._FakeLLM('{"1|Ph.D.": "pi \u0627\u0686 dee"}')
        self.assertEqual(_resolve_spoken_tokens("She has a Ph.D. now.", llm), "She has a Ph.D. now.")

    def test_roman_numerals_are_left_to_the_model(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        llm = self._FakeLLM('{"1|II": "two", "2|VIII": "the eighth"}')
        self.assertEqual(_resolve_spoken_tokens("World War II; Henry VIII.", llm), "World War two; Henry the eighth.")
        # Without the model, the fixed rule still reads a name's numeral.
        self.assertEqual(_resolve_spoken_tokens("Henry VIII ruled.", None), "Henry the Eighth ruled.")

    def test_reading_that_changes_a_number_is_rejected(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        llm = self._FakeLLM('{"1|370": "thirty seven"}')
        self.assertEqual(_resolve_spoken_tokens("Flight 370 departs.", llm), "Flight 370 departs.")

    def test_fixed_rules_run_first_and_stand_when_the_model_fails(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        for llm in (self._FakeLLM("not json"), self._FakeLLM(error=RuntimeError("down")), None):
            self.assertEqual(_resolve_spoken_tokens("Dr. Smith filed in 1903.", llm), "Doctor Smith filed in 1903.")

    def test_shared_model_settings_are_never_changed(self):
        from jarvisrag.speech import _resolve_spoken_tokens

        llm = self._FakeLLM('{"1|1903": "nineteen oh three"}')
        _resolve_spoken_tokens("Built in 1903.", llm)
        self.assertEqual(llm.additional_kwargs, {"num_predict": 256})
        self.assertGreater(llm.copies[0].additional_kwargs["num_predict"], 0)

    def test_answer_is_worded_on_the_speech_thread_only_when_spoken(self):
        import jarvisrag.answer.pipeline as ans

        tts = Mock()
        with _patch_answer("_resolve_spoken_tokens", return_value="worded") as resolve:
            ans._speak_answer(tts, object(), "Built in 1903.")
            resolve.assert_not_called()
            deferred = tts.speak.call_args.args[0]
            self.assertEqual(deferred(), "worded")
            ans._speak_answer(ans.SilentSpeaker(), object(), "Built in 1903.")
        resolve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
