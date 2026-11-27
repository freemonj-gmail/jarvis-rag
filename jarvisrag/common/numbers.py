"""Numbers written in text: one shared reading of digits and separators.

A comma groups thousands only when exactly three digits follow it and no further
digit does: "1,234,567" is one number, while in "Jan 29,2003" the comma separates
a day from a year and in "5,10,15" it separates list items. Every place that pulls
numbers out of text builds its own pattern from these and adds its own context
rules (a currency sign, a percent sign, what may come before or after).
"""

import re
from decimal import Decimal

# The whole part of a number: thousands groups of exactly three digits, or plain digits.
INTEGER_PATTERN = r"(?:\d{1,3}(?:,\d{3})+(?!\d|,\d)|\d+)"
# A whole part with an optional decimal fraction.
NUMBER_PATTERN = INTEGER_PATTERN + r"(?:\.\d+)?"

_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def drop_thousands_commas(text: str) -> str:
    """Remove the commas that group thousands, keeping any other comma between digits."""
    return _THOUSANDS_COMMA_RE.sub("", text)


def number_value(raw: str) -> Decimal:
    """The exact value of a number matched by these patterns (a sign is allowed)."""
    return Decimal(raw.replace(",", ""))


def canonical_number(raw: str) -> str:
    """A number written so that equal numbers compare equal however they are
    written: "7,500.00" == "7500", "2.540" == "2.54", "05" == "5". Empty when
    ``raw`` holds no digits."""
    value = raw.replace(",", "").rstrip(".")
    if not value:
        return ""
    whole, _, fraction = value.partition(".")
    fraction = fraction.rstrip("0")
    return (whole.lstrip("0") or "0") + (f".{fraction}" if fraction else "")
