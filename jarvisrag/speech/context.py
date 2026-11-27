"""Context-aware readings for tokens whose spoken form depends on the sentence.

Flags tokens by shape, asks the model once for their readings, and accepts a
reading only when it keeps every number's value.
"""

import json
import re

from jarvisrag.common.numbers import drop_thousands_commas
from jarvisrag.config import debug_print
from jarvisrag.speech.wording import _normalize_for_speech


# -- Context step: how tokens whose reading depends on meaning are spoken -------
#
# Fixed rules cannot tell "Ct." in an address (Court) from "ct." on a diamond
# (carat), a year from a count, or a postal code from a quantity. The model that
# wrote the answer can, so it is asked how each such token is said in its
# sentence. Tokens are found by shape alone (digits, symbols, short abbreviations
# with a period, runs of capitals, a short unit after a number), and the model may
# change only those tokens. A reading is used only if it is plain words that keep
# every number's exact value; anything else stays for the fixed rules.

_SPEECH_TOKEN_RE = re.compile(r"\S+")


_LEADING_WRAPPERS = "\"'\u201c\u2018[{("


_TRAILING_WRAPPERS = "\"'\u201d\u2019]},;:!?"


_DIGIT_NAMES = ("zero one two three four five six seven eight nine".split(),
                "oh one two three four five six seven eight nine".split())


_SPOKEN_FORM_PROMPT = (
    "You prepare text to be read aloud by a speech engine that can only pronounce plain words.\n"
    "Tokens to resolve are marked [[n|token]]. For each one, write exactly how a careful native English "
    "speaker would say it aloud IN THIS CONTEXT, as plain words only (no digits, symbols or abbreviations), "
    "keeping the exact value of every number. Decide from the meaning of the surrounding text:\n"
    "- an abbreviation, symbol or unit: the word it stands for here;\n"
    "- capital letters: if people pronounce them as a word, write that word in lowercase; if people say the "
    "letters, write each letter as a capital letter, separated by spaces (\"F B I\"), never spelled out as "
    "letter names;\n"
    "- an abbreviation written with periods between its letters: its capital letters, separated by spaces, "
    "unless it stands for words people say in full;\n"
    "- a year: the way years are spoken;\n"
    "- a number that counts or measures something: an ordinary quantity, even when it looks like a year;\n"
    "- an identifier such as a postal code, phone, account or serial number: digit by digit;\n"
    "- a form, section, flight, room or model number: the way people say such codes, usually in pairs of digits;\n"
    "- the day in a date, a rank or a position: an ordinal;\n"
    "- a Roman numeral: after the name of an individual person (a monarch, pope or other named person), \"the\" "
    "and an ordinal (\"Henry V\" -> \"the fifth\"); after anything else -- an event, a vehicle, a product, a "
    "title, a series or a category -- the plain cardinal number (\"Episode IV\" -> \"four\");\n"
    "- a score, range, ratio or time: the way people say it.\n"
    "Reply with JSON only. Use each marker's contents exactly as written as the key and its spoken words as "
    "the value, e.g. {{\"1|token\": \"...\", \"2|token\": \"...\"}}.\n\n"
    "Text:\n{text}\n\nJSON:"
)


def _speech_tokens(text: str) -> list[tuple[int, int, str]]:
    """Tokens whose spoken form depends on context, as (start, end, token)."""
    found: list[tuple[int, int, str]] = []
    after_number = False
    previous = ""
    for match in _SPEECH_TOKEN_RE.finditer(text):
        token, start, end = match.group(0), match.start(), match.end()
        # A unit follows its number within a sentence ("8849 m", "6 hrs."),
        # never across a sentence end or a line break.
        if after_number and (previous.endswith((".", "!", "?")) or "\n" in text[match.start() - 2:match.start()]):
            after_number = False
        raw = token
        if token.startswith("(") and token.endswith(")"):
            token, start, end = token[1:-1], start + 1, end - 1
        while token and token[0] in _LEADING_WRAPPERS and not (token[0] == "(" and ")" in token):
            token, start = token[1:], start + 1
        while token and (
            token[-1] in _TRAILING_WRAPPERS or (token[-1] == ")" and "(" not in token)
        ):
            token, end = token[:-1], end - 1
        # A full stop after a number or a run of capitals ends the sentence; it is
        # not part of the token ("in 1911." / "Henry VIII."), unlike "Ph.D." or "Ct.".
        if token.endswith(".") and "." not in token[:-1] and (
            any(ch.isdigit() for ch in token) or token[:-1].isupper()
        ):
            token, end = token[:-1], end - 1
        if not token:
            after_number = False
            continue
        letters = re.sub(r"[^A-Za-z]", "", token)
        # Money and percentages already have one reading; the fixed rules say them.
        settled = token.startswith("$") or token.rstrip(".").endswith("%")
        has_digit = any(ch.isdigit() for ch in token)
        if (
            (has_digit and not settled)
            or any(ch in token for ch in "\u00a7\u00b0#&")
            or (token.endswith(".") and 0 < len(letters) <= 5 and re.fullmatch(r"[A-Za-z.]+", token))
            or sum(ch.isupper() for ch in token) >= 2
            or (after_number and re.fullmatch(r"[A-Za-z]{1,4}\.?", token))
            # A one-letter Roman numeral after a capitalized word ("World War I", "Henry V").
            or (token in ("I", "V", "X") and previous[:1].isupper() and previous.isalpha())
        ):
            found.append((start, end, token))
        after_number = has_digit and not settled
        previous = raw
    return found


def _number_readings(digits: str) -> set[str]:
    """Every standard spoken form of a run of digits, as plain lowercase words."""
    from num2words import num2words

    value = int(digits)
    forms = {num2words(value), num2words(value, to="ordinal")}
    if len(digits) == 4 or value >= 1000:
        forms.add(num2words(value, to="year"))
    for names in _DIGIT_NAMES:
        forms.add(" ".join(names[int(d)] for d in digits))
    if len(digits) == 4 and value % 100 and value // 100 % 10:  # "nineteen hundred three", "fifteen hundred"
        forms.add(f"{num2words(value // 100)} hundred {num2words(value % 100)}")
    if len(digits) == 4 and not value % 100:
        forms.add(f"{num2words(value // 100)} hundred")
    if len(digits) == 3:  # codes said "four oh one", "three seventy"
        head, tail = num2words(int(digits[0])), digits[1:]
        if tail == "00":
            forms.add(f"{head} hundred")
        elif tail[0] == "0":
            forms.add(f"{head} oh {num2words(int(tail[1]))}")
        else:
            forms.add(f"{head} {num2words(int(tail))}")
    if value == 2:
        forms |= {"half", "halves"}
    if value == 4:
        forms |= {"quarter", "quarters"}
    for form in list(forms):
        if form.startswith(("one hundred", "one thousand")):
            forms |= {form[4:], "a " + form[4:]}
    readings = set()
    for form in forms:
        words = " ".join(w for w in form.lower().replace("-", " ").replace(",", "").split() if w != "and")
        readings |= {words, words + "s", words[:-1] + "ies" if words.endswith("y") else words}
    return readings


def _reading_keeps_numbers(token: str, words: str) -> bool:
    """True when every number in ``token`` is said in ``words`` with its own value."""
    said = " " + " ".join(
        w for w in re.sub(r"[^a-z ]", " ", words.lower().replace("-", " ")).split() if w != "and"
    ) + " "
    # Only a thousands separator joins digits: "29,2003" is a day and a year.
    plain = drop_thousands_commas(token)
    return all(
        any(f" {reading} " in said for reading in _number_readings(run))
        for run in re.findall(r"\d+", plain)
    )


def _speech_llm(llm_client, num_predict: int):
    """A copy of the answer model with its own output budget. The context step
    runs on the speech thread, so it must not change the shared client's
    settings while an answer is being generated; the copy keeps the same model
    and context size, so Ollama does not load anything new."""
    try:
        options = dict(getattr(llm_client, "additional_kwargs", None) or {})
        options["num_predict"] = num_predict
        return llm_client.model_copy(update={"additional_kwargs": options})
    except Exception:
        return None


def _resolve_spoken_tokens(text: str, llm_client) -> str:
    """Word ``text`` for speech, saying each context-dependent token as its
    sentence means it. Falls back to the fixed rules alone when the model is
    unavailable or replies off-format."""
    if not text:
        return text
    if llm_client is None:
        return _normalize_for_speech(text)
    worded = _normalize_for_speech(text, roman_numerals=False)
    tokens = _speech_tokens(worded)
    if not tokens:
        return worded
    marked, last = [], 0
    for number, (start, end, token) in enumerate(tokens, 1):
        marked += [worded[last:start], f"[[{number}|{token}]]"]
        last = end
    marked.append(worded[last:])
    llm = _speech_llm(llm_client, 64 + 24 * len(tokens))
    if llm is None:
        return _normalize_for_speech(text)
    try:
        from llama_index.core.llms import ChatMessage, MessageRole

        reply = llm.chat(
            [ChatMessage(role=MessageRole.USER, content=_SPOKEN_FORM_PROMPT.format(text="".join(marked)))]
        )
        content = reply.message.content or ""
        found = re.search(r"\{.*\}", content, re.S)
        try:
            readings = json.loads(found.group(0)) if found else {}
        except ValueError:
            # A reply with a stray syntax error still names its readings; keep
            # every well-formed pair instead of losing the whole paragraph.
            readings = dict(re.findall(r'"(\d+\|[^"\n]*)"\s*:\s*"([^"\n]*)"', content))
    except Exception as error:
        debug_print(f"[debug] Speech context step skipped ({type(error).__name__}: {error}); using the fixed rules.")
        return _normalize_for_speech(text)
    if not isinstance(readings, dict) or not readings:
        debug_print("[debug] Speech context step gave no readings; using the fixed rules.")
        return _normalize_for_speech(text)
    readings = {str(key).strip().rstrip("."): value for key, value in readings.items()}
    pieces, last = [], 0
    for number, (start, end, token) in enumerate(tokens, 1):
        pieces.append(worded[last:start])
        # Keyed by number AND token, so a miscounted reply cannot land on another token.
        said = readings.get(f"{number}|{token}".rstrip("."))
        usable = (
            isinstance(said, str)
            and said.strip()
            and re.fullmatch(r"[A-Za-z][A-Za-z' .,-]*", said.strip())
            and _reading_keeps_numbers(token, said)
        )
        if usable:
            said = " ".join(said.split())
            # Keep the full stop when the abbreviation ended the line; mid-line, a
            # capital after it is usually a name ("v. Jones"), not a new sentence.
            rest = worded[end:]
            if token.endswith(".") and not said.endswith(".") and (not rest.strip() or re.match(r"[ \t]*\n", rest)):
                said += "."
            pieces.append(said)
        else:
            if isinstance(said, str) and said.strip():
                debug_print(f"[debug] Speech reading rejected for {token!r}: {said!r}; using the fixed rules.")
            pieces.append(token)
        last = end
    pieces.append(worded[last:])
    return "".join(pieces)
