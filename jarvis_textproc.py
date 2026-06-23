"""Text-quality and OCR-garbage heuristics for Jarvis.

Pure-function utilities that normalize raw extracted text, score layout
artifacts, decide whether a chunk is low-information, and reject OCR garbage
using a vocabulary-grounded gate. Foundation module: depends only on the
standard library (``os`` for env/vocab files, ``re`` for the heuristics).
"""
import os
import re


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"(?:\(cid:\d+\)){2,}", " ", text)
    text = re.sub(r"(?:\s*\.\s*){4,}", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_artifact_score(text: str) -> int:
    if not text:
        return 0

    score = 0
    score += len(re.findall(r"\(cid:\d+\)", text)) * 5
    score += len(re.findall(r"(?:\b[A-Za-z0-9]\b\s+){8,}", text)) * 25
    score += len(re.findall(r"(?:\b[A-Za-z]{1,2}\b\s+){10,}", text)) * 20
    return score


def is_low_information_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True

    compact = " ".join(text.lower().split())
    words = re.findall(r"[a-zA-Z]{3,}", compact)
    unique_words = set(words)
    digit_count = sum(char.isdigit() for char in compact)
    alnum_count = sum(char.isalnum() for char in compact)
    punctuation_count = sum(not char.isalnum() and not char.isspace() for char in compact)
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", compact)
    value_numbers = []
    for number in numbers:
        digits_only = re.sub(r"\D", "", number)
        if not digits_only:
            continue
        numeric_value = int(digits_only)
        if 1900 <= numeric_value <= 2099:
            continue
        value_numbers.append(number)

    if "intentionally left blank" in compact:
        return True

    if alnum_count < 20:
        return True

    if len(unique_words) <= 3 and digit_count < 4:
        return True

    if len(unique_words) <= 6 and not value_numbers:
        return True

    if compact.count(".") >= 12 and digit_count <= 4 and len(unique_words) <= 8:
        return True

    if punctuation_count > alnum_count * 2 and digit_count <= 4 and len(unique_words) <= 12:
        return True

    return False


_OCR_GARBAGE_WHITELIST_PATTERNS = (
    re.compile(r"\$\s*\d"),                                  # currency
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),              # dates
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                    # ISO dates
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                    # SSN
    re.compile(r"\b\d{2}-\d{7}\b"),                          # EIN
    re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}"),                  # phone
    re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),                    # phone alt
    re.compile(r"\bForm\s+\d{3,5}\b", re.IGNORECASE),        # IRS forms
    re.compile(r"\bW-?2\b|\b1099-?[A-Z]{1,4}\b|\b1098\b|\b1040\b"),
    re.compile(r"\bbox\s+\d+\b", re.IGNORECASE),             # W-2/1099 box refs
    re.compile(r"\bline\s+\d+\b", re.IGNORECASE),            # IRS line refs
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b"),       # comma-grouped amounts
)


_ENGLISH_VOCAB_FALLBACK: frozenset[str] = frozenset(
    """
    the be to of and a in that have it for not on with he as you do at this but his by from
    they we say her she or an will my one all would there their what so up out if about who get
    which go me when make can like time no just him know take people into year your good some
    could them see other than then now look only come its over think also back after use two
    how our work first well way even new want because any these give day most us is was are were
    been being am does did has had said made went got come came known seen taken given found
    am pm est cst pst gmt utc inc llc llp ltd corp company corporation address street avenue
    road drive lane court place city state zip phone email name date amount total subtotal
    balance payment paid received credit debit account number invoice receipt statement summary
    customer client member account user password username email phone address city state country
    employer employee wages salary income tax withheld federal state local social security medicare
    deduction credit refund return filing status married single dependent spouse child children
    mortgage interest principal property real estate property tax insurance premium policy
    bank checking savings investment portfolio fund stock bond dividend capital gain loss
    contribution distribution rollover ira roth traditional pension retirement plan
    education tuition student loan grant scholarship fellowship qualified expense
    health medical dental vision prescription pharmacy hospital clinic doctor patient
    charity charitable donation contribution gift nonprofit foundation organization
    business expense receipt mileage travel lodging meal entertainment
    page section part chapter article paragraph line item table figure column row
    note please see attached enclosed include including such following before after during
    yes no true false none null empty blank reserved required optional default
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    one two three four five six seven eight nine ten eleven twelve hundred thousand million billion
    """.split()
)


def _load_system_english_vocab() -> frozenset[str]:
    """Load /usr/share/dict/words (or path from JARVIS_VOCAB_FILE) once and
    cache it. Returns frozenset of lowercase words >=3 chars. Falls back to a
    compact embedded set if neither is available."""
    cached = getattr(_load_system_english_vocab, "_cache", None)
    if cached is not None:
        return cached
    candidates = [
        os.environ.get("JARVIS_VOCAB_FILE"),
        "/usr/share/dict/words",
        "/usr/share/dict/american-english",
        "/usr/share/dict/british-english",
    ]
    words: set[str] = set()
    for path in candidates:
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    w = line.strip().lower()
                    if len(w) >= 3 and w.isalpha():
                        words.add(w)
            if words:
                break
        except OSError:
            continue
    if not words:
        words = set(_ENGLISH_VOCAB_FALLBACK)
    # Always merge fallback so domain-specific words (tax/finance) are present
    # even when system dict is loaded.
    words.update(_ENGLISH_VOCAB_FALLBACK)
    cached = frozenset(words)
    _load_system_english_vocab._cache = cached  # type: ignore[attr-defined]
    return cached


def _is_ocr_garbage_text(text: str) -> tuple[bool, str]:
    """Return (is_garbage, reason). Vocabulary-grounded gate: rejects text only
    when the fraction of recognized English words is well below normal AND no
    structured signals (currency, dates, IDs, IRS forms, proper names) appear.
    Pure-numeric receipts and form-line content are preserved via whitelist."""
    if os.environ.get("JARVIS_OCR_GARBAGE_FILTER", "1") != "1":
        return False, ""

    raw = (text or "").strip()
    if len(raw) < 40:
        return False, ""

    for pattern in _OCR_GARBAGE_WHITELIST_PATTERNS:
        if pattern.search(raw):
            return False, ""

    # Multi-word proper-cased names (likely real content even if low vocab match).
    if len(re.findall(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", raw)) >= 2:
        return False, ""

    compact = " ".join(raw.split())
    tokens = compact.split()
    word_tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    pure_number_tokens = [t for t in tokens if re.fullmatch(r"\d[\d,\.]*", t)]

    # Need a meaningful sample of word-shaped tokens to judge vocabulary match.
    if len(word_tokens) < 6:
        return False, ""

    vocab = _load_system_english_vocab()
    letters_only_tokens: list[str] = []
    for tok in word_tokens:
        letters = re.sub(r"[^A-Za-z]", "", tok).lower()
        if len(letters) >= 2:
            letters_only_tokens.append(letters)
    if not letters_only_tokens:
        return False, ""

    in_vocab = sum(1 for w in letters_only_tokens if w in vocab)
    vocab_ratio = in_vocab / len(letters_only_tokens)

    # If a healthy fraction of word tokens are real English, keep it.
    threshold = float(os.environ.get("JARVIS_OCR_VOCAB_THRESHOLD", "0.25"))
    if vocab_ratio >= threshold:
        return False, ""

    reason = (
        f"vocab_ratio={vocab_ratio:.2f} (threshold={threshold:.2f}) "
        f"in_vocab={in_vocab}/{len(letters_only_tokens)} "
        f"len={len(compact)} numeric_tokens={len(pure_number_tokens)}"
    )
    return True, reason
