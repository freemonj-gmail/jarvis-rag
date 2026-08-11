"""Focused tests for the deterministic Jarvis calculator lane."""

import os
import sys
from decimal import Decimal
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvisrag.answer.math import MathQueryRouter, suggest_math_correction
from jarvisrag.answer.answer import QueryRuntime, handle_query
from jarvisrag.speech import _spoken_response_text


class RecordingTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


def run_tests() -> None:
    router = MathQueryRouter()

    cases = {
        "What is 0.1 + 0.2?": Decimal("0.3"),
        "Please subtract 5 from 12": Decimal("7"),
        "multiply 6 by 7": Decimal("42"),
        "divide 20 by 4": Decimal("5"),
        "What is (2 + 3) * 4?": Decimal("20"),
        "What is 20% of 50?": Decimal("10"),
        "What is 30 percent of 100 equal to?": Decimal("30"),
        "Calculate 12.5 per cent of 80": Decimal("10.000"),
        "30 is what percent of 80?": Decimal("37.500"),
        "What percentage of 80 is 30?": Decimal("37.500"),
        "What percent is 30 out of 80?": Decimal("37.500"),
        "What is the percentage increase of 80 to 100?": Decimal("25"),
        "What is the percentage increase from 80 to 100?": Decimal("25"),
        "What is the percent decrease from 100 to 80?": Decimal("20"),
        "What percent did revenue drop from 500 to 400?": Decimal("20"),
        "If my total salary is $120,000 and I am getting a 5% increase, what is my new salary amount?": Decimal("126000.00"),
        "Increase 200 by 15%": Decimal("230"),
        "What is 20% off of 50?": Decimal("40"),
        "Reduce 80 by 25 percent": Decimal("60"),
        "What is the mean of 1, 2, 3, 4?": Decimal("2.5"),
        "Find the median of 1, 9, 3, 5": Decimal("4"),
        "What is the mode of 1, 2, 2, 3?": Decimal("2"),
        "Calculate the population variance of 1, 2, 3": Decimal(2) / Decimal(3),
        "What is the population standard deviation of 1, 1, 1?": Decimal("0"),
        "What is the sample variance of 2, 4, 4, 4, 5, 5, 7, 9?": Decimal(32) / Decimal(7),
        "What is the sample standard deviation of 2, 4, 6?": Decimal("2"),
        "What is the probability of 3 favorable outcomes out of 10 total outcomes?": Decimal("0.3"),
        "What is the chance of rolling a 4 on a fair 6-sided die?": Decimal(1) / Decimal(6),
        "What is the chance of rolling a 9 on a fair 6-sided die?": Decimal("0"),
        "For a six sided single dice, what is the probability of getting any number on that dice?": Decimal(1) / Decimal(6),
        "What is the probability of a number on a six sided dice roll?": Decimal(1) / Decimal(6),
        "What is the probability of a side on a single dice?": Decimal(1) / Decimal(6),
        "What is the probability of a roll for any given side on a single dice?": Decimal(1) / Decimal(6),
        "What is the probability of a 6 sided die?": Decimal(1) / Decimal(6),
        "What is the probability of a coin flip?": Decimal("0.5"),
        "On a 6-sided die, find the chance for one face.": Decimal(1) / Decimal(6),
        "What chance does any face have when rolling a twelve-sided die?": Decimal(1) / Decimal(12),
        "For a twenty-sided die, what is the probability of rolling seven?": Decimal(1) / Decimal(20),
        "What is the probability of heads on a fair coin?": Decimal("0.5"),
        "On a fair coin toss, what chance does tails have?": Decimal("0.5"),
        "What is the probability of an even number on a 6-sided die?": Decimal("0.5"),
        "What is the probability of rolling an odd number on a 20-sided die?": Decimal("0.5"),
        "What is the probability of rolling a number greater than 4 on a 6-sided die?": Decimal(1) / Decimal(3),
        "What is the probability of rolling at least 3 on a 6-sided die?": Decimal(2) / Decimal(3),
        "What is the probability of exactly 2 successes in 4 trials with success probability of 50%?": Decimal("0.3750"),
    }
    for query, expected in cases.items():
        result = router.solve(query)
        assert result is not None, f"calculation was not recognized: {query!r}"
        assert result.value == expected, f"wrong result for {query!r}: {result.value}"
        assert result.response_text.endswith("."), result.response_text

    division_by_zero = router.solve("divide 10 by 0")
    assert division_by_zero is not None
    assert division_by_zero.value is None
    assert "zero" in division_by_zero.response_text.lower()

    no_mode = router.solve("What is the mode of 1, 2, 3?")
    assert no_mode is not None
    assert no_mode.value is None
    assert "no single mode" in no_mode.response_text.lower()

    zero_percentage_base = router.solve("30 is what percent of 0?")
    assert zero_percentage_base is not None
    assert zero_percentage_base.value is None
    assert "zero" in zero_percentage_base.response_text.lower()

    repeated_coin = router.solve("What is the probability of 2 heads on a fair coin?")
    assert repeated_coin is not None
    assert repeated_coin.value is None
    assert "number of trials" in repeated_coin.response_text.lower()

    unsupported_math = router.solve("What is 2 ** 8?")
    assert unsupported_math is not None
    assert unsupported_math.value is None
    assert "only addition" in unsupported_math.response_text.lower()

    not_calculations = (
        "What is the square root of 16?",
        "How old is someone born on 2000-01-01?",
        "What does the report mean?",
        "Summarize the 2025 financial results.",
        "What is probability?",
        # Incidental numbers in prose are not a dataset; these belong to RAG.
        "What is the average rainfall in 2020?",
        "What was the average salary reported in 2024?",
        "What is the variance in the 2024 budget?",
        "What is the median home price mentioned in the report?",
        "What was the average revenue for the Seattle office?",
        # Questions about indexed material must never be answered by the calculator.
        "What is the probability of default described in the filing?",
        "What is the average of the values listed in the report?",
        "According to the document, what percent of 80 is 30?",
        # Operator words used as ordinary English are not operators.
        "What did I take away from the meeting notes?",
        "Summarize the difference between the two reports",
        "How many are less than 5",
    )
    for query in not_calculations:
        assert router.solve(query) is None, f"non-calculation was incorrectly routed: {query!r}"

    for multi_object in (
        "What is the probability of rolling a 7 with two dice?",
        "What is the probability of two heads on a pair of coins?",
    ):
        result = router.solve(multi_object)
        assert result is not None and result.value is None, multi_object
        assert "single die or coin" in result.response_text.lower()

    qualified_die_event = router.solve(
        "For a six sided die, what is the probability of rolling a prime number?"
    )
    assert qualified_die_event is not None
    assert qualified_die_event.value is None
    assert "favorable outcomes" in qualified_die_event.response_text.lower()

    tts = RecordingTTS()
    runtime = QueryRuntime(all_docs=[], llm=None, doc_retriever=None, tts=tts)
    turn: dict[str, Any] = handle_query(
        "30 is what percent of 80?",
        runtime,
    )
    assert turn["response_text"] == (
        "30 as a percentage of 80 equals 37.5 percent."
    )
    assert turn["evidence_lines"] == []
    assert turn["calculation"] == {
        "category": "arithmetic",
        "operation": "percentage rate",
        "expression": "30 as a percentage of 80",
        "value": "37.5",
        "currency": None,
    }
    assert tts.spoken == [
        "30 as a percentage of 80 equals 37.5 percent."
    ]

    # A follow-up may reuse the value the previous turn already computed.
    chain_runtime = QueryRuntime(all_docs=[], llm=None, doc_retriever=None, tts=RecordingTTS())
    first = handle_query(
        "If my total salary is $120,000 and I am getting a 5% increase, what is my new salary amount?",
        chain_runtime,
    )
    assert first["response_text"] == "$120,000.00 increased by 5 percent equals $126,000.00."
    second = handle_query(
        "If I am in a 24% tax bracket, what is my take home salary after taxes?",
        chain_runtime,
        first,
    )
    assert second["response_text"] == (
        "$126,000.00 reduced by 24 percent equals $95,760.00. "
        "This applies only the rate you gave; other withholdings and deductions are not included."
    )

    # Currency stated once is carried forward and spoken as dollars and cents.
    assert _spoken_response_text(second["response_text"]).startswith("$126,000.00")
    for query, expected in {
        "What is 20% of $50?": "20 percent of $50.00 equals $10.00.",
        "Add $1,200.50 and $300": "The sum of $1,200.50, $300.00 equals $1,500.50.",
        "Subtract $30 from $80": "$80.00 minus $30.00 equals $50.00.",
        "What is the average of $10, $20 and $60?": "The mean of $10.00, $20.00, $60.00 equals $30.00.",
    }.items():
        result = router.solve(query)
        assert result is not None and result.response_text == expected, query

    # The caveat is scoped to net/take-home framing, not to plain arithmetic.
    plain_reduction = router.solve("Reduce 80 by 25 percent")
    assert plain_reduction is not None
    assert plain_reduction.response_text == "80 reduced by 25 percent equals 60."

    # With no prior calculation there is no base to reuse, so nothing is invented.
    assert router.solve("If I am in a 24% tax bracket, what is my take home salary after taxes?") is None

    # A follow-up may name the previous answer, or omit the base entirely.
    prior = {"calculation": {"value": "50648.75", "currency": "$"}}
    for query, expected in {
        "Divide that by 4": "$50,648.75 divided by 4 equals $12,662.19.",
        "Take away 27%": "$50,648.75 reduced by 27 percent equals $36,973.59.",
        "Take away 27 percent": "$50,648.75 reduced by 27 percent equals $36,973.59.",
        # 500 was given as a bare number, so it stays scalar.
        "the difference between that and 500": "$50,648.75 minus 500 equals $50,148.75.",
        "the difference between that and $500": "$50,648.75 minus $500.00 equals $50,148.75.",
        "the difference between that and 500 dollars": "$50,648.75 minus $500.00 equals $50,148.75.",
    }.items():
        result = router.solve(query, prior)
        assert result is not None and result.response_text == expected, query

    # Currency follows the operand that carries the marker, in either position,
    # whether the marker is a symbol or the word.
    for query, expected in {
        "Multiply $50,648.75 by 4": "The product of $50,648.75, 4 equals $202,595.00.",
        "Multiply 4 by $50,648.75": "The product of 4, $50,648.75 equals $202,595.00.",
        "Multiply 4 by 3": "The product of 4, 3 equals 12.",
        "Divide 100 by $8": "100 divided by $8.00 equals $12.50.",
        "Add $1,200.50 and 300": "The sum of $1,200.50, 300 equals $1,500.50.",
        "Multiply 500 dollars by 4": "The product of $500.00, 4 equals $2,000.00.",
        "Multiply 4 by 500 dollars": "The product of 4, $500.00 equals $2,000.00.",
        "Multiply 500 USD by 4": "The product of $500.00, 4 equals $2,000.00.",
        "Add 1200 dollars and 300": "The sum of $1,200.00, 300 equals $1,500.00.",
        "Subtract 30 dollars from 80 dollars": "$80.00 minus $30.00 equals $50.00.",
    }.items():
        result = router.solve(query)
        assert result is not None and result.response_text == expected, query

    # Steps joined by a connector are evaluated left to right.
    for query in (
        "Divide that by 4 and take away 27%",
        "Divide the previous answer by 4 and take away 27 percent",
        "divide by 4 then take away 27%",
    ):
        chained = router.solve(query, prior)
        assert chained is not None and chained.value == Decimal("9243.396875"), query

    # Every operator answers to its symbol and to its spelled-out names.
    for expected, phrasings in {
        Decimal("48"): (
            "What is 12 * 4",
            "What is 12 times 4",
            "What is 12 multiplied by 4",
            "What is the product of 12 and 4",
        ),
        Decimal("12.5"): (
            "What is 100 / 8",
            "What is 100 divided by 8",
            "What is the quotient of 100 and 8",
        ),
        Decimal("75"): (
            "What is 90 - 15",
            "What is 90 minus 15",
            "What is 90 take away 15",
            "What is the difference between 90 and 15",
            "What is the difference of 90 and 15",
        ),
        Decimal("12"): (
            "What is 5 + 7",
            "What is 5 plus 7",
            "What is the sum of 5 and 7",
        ),
        Decimal("25"): (
            "What is 10% of 250",
            "What is 10 percent of 250",
        ),
    }.items():
        for query in phrasings:
            result = router.solve(query)
            assert result is not None and result.value == expected, query

    # A misspelled operator is offered as a correction, not silently assumed.
    misspelled_runtime = QueryRuntime(all_docs=[], llm=None, doc_retriever=None, tts=RecordingTTS())
    base = handle_query("What is 25% of $200,000?", misspelled_runtime)
    assert base["response_text"] == "25 percent of $200,000.00 equals $50,000.00."
    corrected = handle_query("Multiple the previous answer by 4.", misspelled_runtime, base)
    assert corrected["response_text"] == (
        'Did you mean "multiply" instead of "Multiple"? '
        "Read that way: The product of $50,000.00, 4 equals $200,000.00. "
        "If not, ask again and I will search your documents instead."
    ), corrected["response_text"]

    # The closest spelling is not always the calculable one, so every candidate is tried.
    for query, expected in (
        ("Multiple 12 by 3", "multiply"),
        ("Divde 12 by 3", "divide"),
        ("Subtrac 3 from 12", "subtract"),
    ):
        found = suggest_math_correction(query)
        assert found is not None and found[1] == expected, query

    # No suggestion unless the respelling is what makes the query calculable.
    for no_suggestion in (
        "multiply 12 by 3",
        "What is in the quarterly report for 2024",
        "Show me the 3 attachments",
        "What certification is shown in CISSP_Certificate.pdf?",
        # Two near-miss words at once are too ambiguous to guess from.
        "Multiple the divde by 3",
    ):
        assert suggest_math_correction(no_suggestion) is None, no_suggestion

    print(f"math router tests passed ({len(cases)} calculations)")


if __name__ == "__main__":
    run_tests()
