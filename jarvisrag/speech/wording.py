"""Fixed wording rules that rewrite written forms into spoken words.

Numerals, units, dates, addresses, state names, and Roman numerals.
"""

import re

from jarvisrag.common.numbers import drop_thousands_commas, INTEGER_PATTERN


# --- Spoken-text abbreviation / numeral normalization -----------------------
# Kokoro (and the espeak/misaki G2P front-end) has no semantic understanding of
# context-dependent abbreviations: it cannot tell "Dr. Smith" (Doctor) from
# "Elm Dr." (Drive), or "Henry III" (the Third) from "Section III" (three).
# That disambiguation must happen upstream, so we expand the common cases here
# with lightweight, context-aware rules BEFORE the text reaches the TTS engine.
# This runs on the spoken path only; printed/displayed output is unaffected.

# Roman-numeral name suffixes -> ordinal words (e.g. "Henry III" -> "the Third").
_ROMAN_NAME_ORDINALS = {
    "II": "the Second",
    "III": "the Third",
    "IV": "the Fourth",
    "V": "the Fifth",
    "VI": "the Sixth",
    "VII": "the Seventh",
    "VIII": "the Eighth",
}


# Words commonly followed by a Roman numeral that are NOT personal names, so we
# must NOT rewrite "Section III" -> "Section the Third" or "World War II" ->
# "World War the Second".
_NON_NAME_BEFORE_ROMAN = frozenset({
    "section", "chapter", "part", "volume", "article", "phase", "type",
    "class", "appendix", "act", "book", "title", "figure", "table", "war",
    "level", "grade", "stage", "group", "round", "episode", "step", "tier",
})


# Spoken word for each digit, used to read ZIP codes one digit at a time.
_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


# US state / territory postal abbreviations. A 5-digit number immediately
# following one of these is treated as a ZIP code (read digit by digit).
_US_STATE_ABBR = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
})


def _spell_digits(digits: str) -> str:
    """Render a run of digits as space-separated number words, e.g.
    "98052" -> "nine eight zero five two" (read one digit at a time)."""
    return " ".join(_DIGIT_WORDS[d] for d in digits if d in _DIGIT_WORDS)


# Rate units that turn a "/" into the spoken word "per" (e.g. "$7,500/month" ->
# "7500 dollars per month"). Scoped to known denominators so ordinary slashes
# (dates like "12/25", "and/or", "TCP/IP") are left untouched.
_PER_UNITS = (
    "month", "months", "mo", "year", "years", "yr", "yrs", "annum",
    "week", "weeks", "wk", "wks", "day", "days", "hour", "hours", "hr", "hrs",
    "minute", "minutes", "min", "mins", "second", "seconds", "sec",
    "quarter", "quarters", "capita", "person", "unit", "units", "share",
    "shares", "mile", "miles", "gallon", "gallons", "acre", "acres",
)


_PER_SLASH_RE = re.compile(
    r"\s*/\s*(?=(?:" + "|".join(_PER_UNITS) + r")\b)",
    re.IGNORECASE,
)


# Magnitude scales that may trail a dollar amount, as either a spelled-out word
# ("$4 million") or a single-letter suffix ("$96k", "$4m"). Maps every accepted
# token to the spoken word so "$4 million" reads "4 million dollars" and "$96k"
# reads "96 thousand dollars" instead of dropping the scale.
_SCALE_WORDS = {
    "k": "thousand", "m": "million", "b": "billion", "t": "trillion",
    "thousand": "thousand", "million": "million",
    "billion": "billion", "trillion": "trillion",
}


# Optional scale after a "$N" amount: a letter suffix touching the digits
# (guarded so "million" is not read as a bare "m") or a spelled-out word after
# whitespace. Kept out of the number capture so the existing cents path is
# unaffected when no scale is present.
_DOLLAR_SCALE_SUFFIX = (
    r"(?:\s*(?P<abbr>[kKmMbBtT])(?![A-Za-z])"
    r"|\s+(?P<word>[Tt]housand|[Mm]illion|[Bb]illion|[Tt]rillion))?"
)


def _cardinal_words(digits: str) -> str:
    """Spell an integer string as cardinal words so a speech engine never applies
    a year-style reading (e.g. 4534 -> 'four thousand ...' not 'forty-five
    thirty-four'). Falls back to the digits if num2words is unavailable."""
    try:
        from num2words import num2words

        return num2words(int(digits)).replace(",", "")
    except Exception:
        return digits


# Sub-second time-unit abbreviations -> spoken words. Only expanded when the
# unit is attached to a number (e.g. "200 ms" / "5us" / "10 ns"), so the plain
# English word "us" ("tell us") and unrelated tokens are left untouched. The
# micro sign (µ) and the letter "u" both stand in for "micro". Case-sensitive
# match ("ms" not "MS") so "MS" (Microsoft/Missouri/degree) is unaffected.
_TIME_UNIT_WORDS = {
    "ms": "millisecond",
    "us": "microsecond",
    "\u00b5s": "microsecond",  # µs (micro sign)
    "ns": "nanosecond",
    "ps": "picosecond",
}


_TIME_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*(ms|us|\u00b5s|ns|ps)\b"
)


def _expand_time_units(text: str) -> str:
    """Read attached sub-second durations in full: "200 ms" -> "200 milliseconds",
    "1 ms" -> "1 millisecond". Pluralizes based on the numeric value."""
    def _repl(match: "re.Match[str]") -> str:
        number, unit = match.group(1), match.group(2)
        word = _TIME_UNIT_WORDS[unit]
        try:
            plural = float(number) != 1.0
        except ValueError:
            plural = True
        return f"{number} {word}{'s' if plural else ''}"

    return _TIME_UNIT_RE.sub(_repl, text)


# US-style slash dates with a four-digit year: "1/15/2026" (month/day/year) or
# "1/2026" (month/year). Year-less forms like "12/25" stay as written because
# they cannot be told apart from fractions, and a slash inside a larger token
# (paths, URLs, "1/2026/3") is left alone.
_SLASH_DATE_RE = re.compile(r"(?<![\w/.\-])(\d{1,2})/(?:(\d{1,2})/)?(\d{4})(?![\w/\-])")


# A month/year pair has no day to validate it, so "1/1000" (a ratio) must not be
# read as a month and year; only calendar years of modern documents qualify.
_MONTH_YEAR_RANGE = range(1900, 2100)


def _expand_slash_dates(text: str) -> str:
    """Speak slash dates as a person says them: "1/15/2026" -> "January
    fifteenth, twenty twenty-six" and "1/2026" -> "January twenty twenty-six".
    Impossible dates (month 13, February 30) are left as written."""
    try:
        from num2words import num2words
    except Exception:
        return text
    import calendar
    import datetime

    def _repl(match: "re.Match[str]") -> str:
        month, day, year = int(match.group(1)), match.group(2), int(match.group(3))
        if not 1 <= month <= 12:
            return match.group(0)
        if day is None:
            if year not in _MONTH_YEAR_RANGE:
                return match.group(0)
            return f"{calendar.month_name[month]} {num2words(year, to='year')}"
        try:
            datetime.date(year, month, int(day))
        except ValueError:
            return match.group(0)
        return (
            f"{calendar.month_name[month]} {num2words(int(day), to='ordinal')}, "
            f"{num2words(year, to='year')}"
        )

    return _SLASH_DATE_RE.sub(_repl, text)


# Street abbreviations that collide with a title ("Dr." Drive/Doctor, "St."
# Street/Saint). A house number in front is what distinguishes the address.
_STREET_WORDS = {"Dr": "Drive", "St": "Street"}


_STREET_ADDRESS_RE = re.compile(r"(\d+\s+(?:[A-Z][\w'\u2019-]*\s+)*)(Dr|St)\.")


# Spoken names for the two-letter state codes, expanded only inside an address
# so ordinary words that share a code ("OR", "IN", "ME") are left alone.
_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico",
}


# "City, ST" followed by a ZIP, punctuation or the end of the text. Requiring
# that trailing context keeps prose like "sales, OR marketing" untouched.
_STATE_IN_ADDRESS_RE = re.compile(
    r",\s*([A-Z]{2})\b(?=\s*(?:\d{5}(?:-\d{4})?|[.,;:!?)]|$))"
)


# A ZIP is read digit-by-digit when it follows a state, whether written as the
# 2-letter code ("WA 98052") or the full name that _STATE_IN_ADDRESS_RE expands
# the code into before step 5 runs ("Maryland 20878"). One pattern, reused by
# step 5's _state_zip_repl alongside _US_STATE_ABBR / _spell_digits.
_STATE_NAME_SET = frozenset(_STATE_NAMES.values())


_STATE_ZIP_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(n) for n in _STATE_NAME_SET), key=len, reverse=True))
    + r"|[A-Z]{2})\s+(\d{5})(?:-(\d{4}))?\b"
)


def _normalize_for_speech(text: str, roman_numerals: bool = True) -> str:
    """Expand context-dependent abbreviations and numerals for natural speech.

    Handles the cases TTS gets wrong on its own:
      * Titles before a name:  "Dr./Mr./Mrs./Ms./Prof./St. <Name>" -> spoken word
      * Address abbreviations: remaining "Dr." -> "Drive", "St." -> "Street"
      * Generational suffixes: "Jr."/"J-R" -> "Junior", "Sr."/"S-R" -> "Senior"
      * Name numerals:         "<Name> III"/"I-I-I" -> "<Name> the Third"
      * Slash dates:           "1/15/2026" -> "January fifteenth, twenty twenty-six"

    Rules are case-sensitive where it matters (e.g. "Ms" -> "Miss" but "MS",
    which may mean Microsoft or Missouri, is left untouched). Only the spoken
    string is affected; printed output keeps the original text.
    """
    if not text:
        return text
    s = text

    # 0) A street number ahead of the abbreviation settles it as an address, so
    #    resolve those before the title rules below can claim them: in
    #    "11955 Freedom Dr. Reston" the following word is a city, not a name.
    s = _STREET_ADDRESS_RE.sub(
        lambda m: m.group(1) + _STREET_WORDS[m.group(2)], s
    )

    # 1) Honorific / title abbreviations that are FOLLOWED by a capitalized name
    #    read as titles. The name itself is preserved via a zero-width lookahead.
    title_map = (
        (r"Dr\.", "Doctor"),
        (r"Mrs\.", "Missus"),
        (r"Mr\.", "Mister"),
        (r"Ms\.?", "Miss"),
        (r"Prof\.", "Professor"),
        (r"St\.", "Saint"),     # e.g. "St. Louis" -> "Saint Louis"
    )
    for abbr, full in title_map:
        s = re.sub(rf"\b{abbr}\s+(?=[A-Z][a-z])", full + " ", s)

    # 2) Remaining "Dr." / "St." are address abbreviations (not before a name).
    s = re.sub(r"\bDr\.", "Drive", s)
    s = re.sub(r"\bSt\.", "Street", s)

    # 2b) State codes in a "City, ST" address read as the full state name.
    s = _STATE_IN_ADDRESS_RE.sub(
        lambda m: ", " + _STATE_NAMES[m.group(1)] if m.group(1) in _STATE_NAMES else m.group(0),
        s,
    )

    # 3) Generational suffixes (period-terminated form first so the standalone
    #    boundary form does not re-match the same token).
    s = re.sub(r"\bJr\.", "Junior", s)
    s = re.sub(r"\bJr\b", "Junior", s)
    s = re.sub(r"\bJ-R\b", "Junior", s)
    s = re.sub(r"\bSr\.", "Senior", s)
    s = re.sub(r"\bSr\b", "Senior", s)
    s = re.sub(r"\bS-R\b", "Senior", s)

    # 4) Roman-numeral name suffixes -> ordinal words, skipping structural words
    #    (Section, World War, Type, ...) that are not personal names.
    def _roman_repl(match: "re.Match[str]") -> str:
        prefix_word = match.group(1)
        roman = match.group(2).upper().replace("-", "").replace(".", "")
        if prefix_word.lower() in _NON_NAME_BEFORE_ROMAN:
            return match.group(0)
        ordinal = _ROMAN_NAME_ORDINALS.get(roman)
        if not ordinal:
            return match.group(0)
        return f"{prefix_word} {ordinal}"

    #    When the context step runs, it reads Roman numerals by meaning instead
    #    ("World War Two", "Rocky Four", "Henry the Eighth"); this rule is the
    #    fallback for when the model is unavailable.
    if roman_numerals:
        s = re.sub(
            r"\b([A-Za-z]+)\s+"
            r"((?:I-I-I|I-I|I\.I\.I|VIII|VII|VI|IV|III|II|V))\b",
            _roman_repl,
            s,
        )

    # 5) ZIP codes in addresses -> read digit by digit. Only fire on a strong
    #    address signal so ordinary 5-digit numbers (years, counts, IDs) are
    #    left alone: (a) ZIP+4 form "12345-6789", or (b) a 5-digit number
    #    immediately following a US state, as the 2-letter code ("WA 98052") or
    #    the full name step 2b expanded it into ("Maryland 20878").
    def _zip4_repl(match: "re.Match[str]") -> str:
        return f"{_spell_digits(match.group(1))} {_spell_digits(match.group(2))}"

    s = re.sub(r"\b(\d{5})-(\d{4})\b", _zip4_repl, s)

    def _state_zip_repl(match: "re.Match[str]") -> str:
        state, zip5, plus4 = match.group(1), match.group(2), match.group(3)
        if state not in _US_STATE_ABBR and state not in _STATE_NAME_SET:
            return match.group(0)
        spoken = _spell_digits(zip5)
        if plus4:
            spoken = f"{spoken} {_spell_digits(plus4)}"
        return f"{state} {spoken}"

    s = _STATE_ZIP_RE.sub(_state_zip_repl, s)

    # 6) Rate slash -> "per" before a known unit, e.g. "$7,500/month" ->
    #    "$7,500 per month". Dates ("12/25"), "and/or", "TCP/IP" etc. are
    #    untouched because their right-hand side is not a rate unit.
    s = _PER_SLASH_RE.sub(" per ", s)

    # 7) Sub-second time units attached to a number: "200 ms" -> "200
    #    milliseconds", "5us" -> "5 microseconds", "10 ns" -> "10 nanoseconds".
    s = _expand_time_units(s)

    # 8) Slash dates -> spoken dates: "1/1/2026" -> "January first, twenty
    #    twenty-six", "1/2026" -> "January twenty twenty-six". Runs after the
    #    rate-slash rule, whose units never follow a date's slash.
    s = _expand_slash_dates(s)
    return s


# Words that, when present in the same sentence as a comma-grouped number,
# strongly imply the number is a dollar amount and should be spoken as such.
_TTS_MONEY_KEYWORDS = (
    "income", "wages", "salary", "salaries", "tax", "taxes", "withheld",
    "withholding", "deduction", "deductions", "credit", "credits", "refund",
    "refunds", "paid", "payment", "payments", "balance", "due", "owed",
    "owing", "amount", "amounts", "total", "totals", "subtotal", "fee",
    "fees", "cost", "costs", "price", "expense", "expenses", "revenue",
    "revenues", "earnings", "compensation", "interest", "dividend",
    "dividends", "contribution", "contributions", "agi", "gross",
    "adjusted", "exemption", "exemptions", "loan", "principal", "mortgage",
    "premium", "rebate", "value", "values", "dollars", "dollar",
)


def _add_dollar_sign_in_financial_context(text: str) -> str:
    """Prepend `$` to comma-grouped numbers (e.g. `453,172`, `8,577.50`)
    when their sentence mentions a money keyword. This lets the existing
    `$N` -> 'N dollars' rule fire so the spoken output says 'dollars'
    instead of just reading a bare number.

    We only touch comma-grouped numbers (>= 1,000 with thousands separator)
    to avoid mangling years, line numbers, phone digits, dates, etc. Numbers
    already written with the word "dollar(s)" are skipped so we don't produce
    "dollars dollars"."""
    money_pattern = re.compile(
        r"(?<![\w$,])(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)(?!\w)(?![,\d])(?!\s*[Dd]ollars?\b)"
    )
    keywords = _TTS_MONEY_KEYWORDS
    # Split on sentence boundaries while keeping the boundary characters so
    # we can rejoin without altering pause cues.
    parts = re.split(r"(?<=[.!?])\s+", text)
    rewritten: list[str] = []
    for sentence in parts:
        lowered = sentence.lower()
        has_money_word = any(kw in lowered for kw in keywords)
        if has_money_word:
            sentence = money_pattern.sub(r"$\1", sentence)
        rewritten.append(sentence)
    return " ".join(rewritten)


def _spoken_dollar_amount(match):
    whole = match.group(1).replace(",", "")
    frac = match.group(2) or ""
    scale_word = _SCALE_WORDS.get(
        (match.group("abbr") or match.group("word") or "").lower(), ""
    )
    if scale_word:
        # A scaled amount: the decimal is a fractional multiplier, not cents
        # ("$4.5 million" -> "4 point 5 million dollars").
        number = f"{whole} point {frac}" if frac else whole
        return f"{number} {scale_word} dollars"
    cents_raw = frac.ljust(2, "0")[:2]
    cents = int(cents_raw) if cents_raw else 0
    # Spell the amount out so the engine never reads a 4-digit figure as a year.
    dollars = _cardinal_words(whole)
    if cents:
        return f"{dollars} dollars and {_cardinal_words(str(cents))} cents"
    return f"{dollars} dollars"


def _normalize_spoken_semantics(text: str) -> str:
    """Rewrite text into the words a speech engine should actually say.

    Covers what a listener would otherwise hear wrong regardless of engine:
    context-dependent abbreviations (Dr., Jr., III), URLs, money, percentages
    and decimals. Purely orthographic scrubbing -- stripping apostrophes and
    punctuation -- is deliberately excluded, because that is a Kokoro/espeak
    front-end requirement and degrades engines that use punctuation for phrasing.
    """
    cleaned = str(text or "")
    # Expand abbreviations before any punctuation/hyphen stripping downstream, so
    # the rules can still see periods and hyphens ("Dr.", "J-R", "I-I-I").
    cleaned = _normalize_for_speech(cleaned)
    cleaned = re.sub(r"https?://\S+", " link omitted ", cleaned)
    # Recognize money-shaped numbers in financial sentences (no `$` needed).
    cleaned = _add_dollar_sign_in_financial_context(cleaned)
    cleaned = re.sub(
        r"\$\s*(" + INTEGER_PATTERN + r")(?:\.(\d{1,2}))?" + _DOLLAR_SCALE_SUFFIX,
        _spoken_dollar_amount,
        cleaned,
    )
    cleaned = re.sub(
        r"\b(" + INTEGER_PATTERN + r")\.(\d+)\s*%",
        lambda m: f"{m.group(1).replace(',', '')} point {m.group(2)} percent",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(" + INTEGER_PATTERN + r")\s*%",
        lambda m: f"{m.group(1).replace(',', '')} percent",
        cleaned,
    )
    # Drop only thousands separators: in "Jan 29,2003" the comma separates a
    # day from a year, and removing it would read one large number.
    cleaned = drop_thousands_commas(cleaned)
    cleaned = re.sub(r"(?<=\d),(?=\d)", ", ", cleaned)
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", " point ", cleaned)
    return cleaned
