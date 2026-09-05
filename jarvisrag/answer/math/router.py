"""Conservative query routing for Jarvis's deterministic calculator lane."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable

from .lexicon import (
    CHAIN_SPLIT_RE,
    LEADING_OPERATOR_RE,
    PERCENT_MARKER_RE as _PERCENT_MARKER_RE,
    REFERENCE_RE,
    suggest_operator_corrections,
    to_operator_symbols,
)
from .operations import (
    ArithmeticOperations,
    FinancialOperations,
    MathOperationError,
    ProbabilityOperations,
    RatioOperations,
    StatisticsOperations,
    to_decimal,
)


_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_DATE_LIKE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_TOKEN_RE = re.compile(r"-?\d+|[a-z]+")
_CARDINALS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
_EVENT_QUALIFIERS = {
    "above", "below", "between", "even", "except", "greater", "less",
    "multiple", "odd", "over", "prime", "under",
}
_OUTCOME_VERBS = {"get", "gets", "getting", "land", "landing", "lands", "roll", "rolling", "rolls"}
# A die is six-faced unless stated otherwise, mirroring the two-sided coin default.
_STANDARD_DIE_SIDES = 6
_MULTIPLE_OBJECT_WORDS = {"pair", "both"}
# Questions that point at indexed material belong to retrieval, never to the calculator.
_DOCUMENT_CONTEXT_RE = re.compile(
    r"\b(?:document|documents|report|reports|file|files|pdf|spreadsheet|manual|"
    r"filing|invoice|contract|transcript|attachment|attached|described|mentioned|"
    r"stated|listed|according\s+to|in\s+the\s+text)\b",
    re.I,
)
# Words that may introduce a supplied dataset. Anything else between the
# statistic and its numbers means the numbers belong to prose, not to a dataset.
_STAT_LEAD_RE = re.compile(
    r"^(?:\s*(?:of|for|from|the|these|those|this|that|a|an|is|are|was|were|"
    r"numbers?|values?|data|dataset|set|list|series|observations|readings|"
    r"scores|sample|population|following)\b|\s*[:=,-])+",
    re.I,
)
_LIST_SEPARATOR_RE = re.compile(r"\band\b|[,;&]", re.I)
_PERCENT_CHANGE_RE = re.compile(
    r"\b(increases?d?|growth|grew|rises?|rose|gains?|"
    r"decreases?d?|declines?d?|drops?(?:ped)?|falls?|fell|losses?|loss)\b",
    re.I,
)
_DECREASE_WORDS = ("decreas", "declin", "drop", "fall", "fell", "loss")
# Applying a rate to a base ("a 5% increase") rather than comparing two amounts.
_APPLY_INCREASE_RE = re.compile(
    r"\b(?:increase[sd]?|raise[sd]?|raising|rise[sn]?|rose|up|plus|added|adding|"
    r"markup|marked\s+up|gain(?:s|ed)?|growth|grew)\b",
    re.I,
)
_APPLY_DECREASE_RE = re.compile(
    r"\b(?:decrease[sd]?|reduc(?:e|es|ed|ing|tion)|discount(?:ed)?|off|minus|"
    r"deduct(?:s|ed|ing|ion|ions)?|after|net|drop(?:s|ped)?|decline[sd]?|falls?|"
    r"fell|cut|withheld|withholding|subtract(?:s|ed|ing)?|"
    r"tak(?:e|es|en|ing)\s+(?:away|off))\b",
    re.I,
)
# Net/after-deduction framing, where a single rate is not the whole story.
_NET_RESULT_RE = re.compile(
    r"\btake[-\s]?home\b|\bbring\s+home\b|"
    r"\bnet\s+(?:pay|salary|income|amount|earnings|proceeds|wage[s]?)\b|"
    r"\bafter\s+(?:tax|taxes|deduction[s]?|withholding[s]?|fees|expenses)\b",
    re.I,
)
_NET_DEDUCTION_NOTE = (
    "This applies only the rate you gave; other withholdings and deductions are not included."
)
# Currency words are limited to unambiguous ones so "pounds" of weight is not money.
_CURRENCY_RE = re.compile(r"[$\u20ac\u00a3\u00a5]|\b(?:dollars?|usd)\b", re.I)
_CURRENCY_WORD_SYMBOLS = {"dollar": "$", "dollars": "$", "usd": "$"}
_THRESHOLD_RE = re.compile(
    r"\b(greater\s+than|more\s+than|larger\s+than|above|over|"
    r"less\s+than|fewer\s+than|smaller\s+than|below|under|"
    r"at\s+least|at\s+most)\s+([\w-]+)",
    re.I,
)
_REQUEST_PREFIX_RE = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:can|could|would|will)\s+you(?:\s+please)?\s+|"
    r"i(?:'d|\s+would)\s+like\s+you\s+to\s+)*",
    re.IGNORECASE,
)
_ACTION_PREFIX_RE = re.compile(
    r"^(?:what\s+is|what's|calculate|compute|solve|find)\s+",
    re.IGNORECASE,
)
_ARTICLE_PREFIX_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

# Financial lanes only run when their intent word is present, so an ordinary
# percentage question is never mistaken for a loan or an investment.
_COMPOUND_RE = re.compile(r"\bcompound(?:ed|ing|s)?\b|\bfuture\s+value\b", re.I)
_AMORTIZE_RE = re.compile(
    r"\bamorti[sz]\w*\b|\bmortgage\b|\bloan\b|\binstal?lments?\b|"
    r"\b(?:monthly|weekly|biweekly|bi-weekly|quarterly|annual)\s+payments?\b|"
    r"\bpayments?\s+(?:on|for)\b",
    re.I,
)
_COMPOUND_TIMES_RE = re.compile(
    r"\bcompounded\s+(\d+)\s+times\s+(?:a|per)\s+year\b", re.I
)
# Checked in order: a specific compounding cadence wins over a generic "annual".
_COMPOUND_FREQUENCIES = (
    (re.compile(r"\bsemi[-\s]?annual(?:ly)?\b|\bbi[-\s]?annual(?:ly)?\b|\btwice\s+(?:a|per)\s+year\b", re.I), 2, "semiannually"),
    (re.compile(r"\bquarterly\b", re.I), 4, "quarterly"),
    (re.compile(r"\bmonthly\b", re.I), 12, "monthly"),
    (re.compile(r"\bbi[-\s]?weekly\b", re.I), 26, "biweekly"),
    (re.compile(r"\bweekly\b", re.I), 52, "weekly"),
    (re.compile(r"\bdaily\b", re.I), 365, "daily"),
    (re.compile(r"\bannual(?:ly)?\b|\byearly\b|\bper\s+year\b|\ba\s+year\b|\bonce\s+(?:a|per)\s+year\b", re.I), 1, "annually"),
)
_FREQUENCY_LABELS = {1: "annually", 2: "semiannually", 4: "quarterly", 12: "monthly", 24: "semimonthly", 26: "biweekly", 52: "weekly", 365: "daily"}
_PAYMENT_LABELS = {1: "annual", 2: "semiannual", 4: "quarterly", 12: "monthly", 24: "semimonthly", 26: "biweekly", 52: "weekly"}
_RATE_UNIT = r"%|percent|per\s*cent"

# Ratios: fire only on explicit ratio intent so ordinary numbers, ranges, and
# clock times are never mistaken for a ratio.
_RATIO_WORD_RE = re.compile(r"\bratio\b|\bproportion(?:al|ally|s)?\b", re.I)
_RATIO_SHARE_RE = re.compile(
    r"\b(?:divide[sd]?|split|shared?|allocate[sd]?|distribute[sd]?|apportion(?:ed)?)\b",
    re.I,
)
_RATIO_SIMPLIFY_RE = re.compile(
    r"\bsimplif(?:y|ied|ies)\b|\breduc(?:e|ed|es)\b|\blowest\s+terms\b|\bsimplest\s+form\b",
    re.I,
)
_RATIO_UNKNOWN = r"(?:x|n|\?|what|unknown|blank|missing)"
_RATIO_TERM = rf"(?:{_NUMBER}|{_RATIO_UNKNOWN})"
# Each ratio term may carry a leading currency symbol ("$1,200"); it is matched
# outside the capture groups so the groups hold only the number or the unknown.
_CURRENCY_PREFIX = r"[$\u20ac\u00a3\u00a5]?\s*"
# A full proportion equation with one unknown: "3:4 = 9:x", "3 is to 4 as 9 is to what".
_PROPORTION_RE = re.compile(
    rf"{_CURRENCY_PREFIX}({_RATIO_TERM})\s*(?::|\bis\s+to\b)\s*{_CURRENCY_PREFIX}({_RATIO_TERM})"
    rf"\s*(?:=|::|\bas\b)\s*"
    rf"{_CURRENCY_PREFIX}({_RATIO_TERM})\s*(?::|\bis\s+to\b)\s*{_CURRENCY_PREFIX}({_RATIO_TERM})",
    re.I,
)
_RATIO_COLON_LIST_RE = re.compile(
    rf"{_CURRENCY_PREFIX}{_NUMBER}(?:\s*:\s*{_CURRENCY_PREFIX}{_NUMBER})+"
)
_RATIO_TO_LIST_RE = re.compile(
    rf"{_CURRENCY_PREFIX}{_NUMBER}(?:\s+to\s+{_CURRENCY_PREFIX}{_NUMBER})+", re.I
)


@dataclass(frozen=True)
class CalculationResult:
    """A calculation answer and its machine-readable provenance."""

    category: str
    operation: str
    expression: str
    value: Decimal | None
    response_text: str
    currency: str | None = None

    def metadata(self) -> dict[str, str | None]:
        return {
            "category": self.category,
            "operation": self.operation,
            "expression": self.expression,
            "value": _format_decimal(self.value)[0] if self.value is not None else None,
            "currency": self.currency,
        }


@dataclass(frozen=True)
class _UniformSampleSpace:
    """A finite experiment whose elementary outcomes are equally likely."""

    kind: str
    size: int


@dataclass(frozen=True)
class _NumericTerm:
    value: Decimal
    start: int
    end: int


def _numbers(text: str) -> list[Decimal]:
    return [to_decimal(match.group(0).replace(",", "")) for match in _NUMBER_RE.finditer(text)]


_RATIO_CLEAN_RE = re.compile(r"[$\u20ac\u00a3\u00a5,\s]")


def _ratio_decimal(token: str) -> Decimal:
    """Read a ratio term, dropping any currency symbol and thousands separators."""
    return to_decimal(_RATIO_CLEAN_RE.sub("", token))


def _numeric_terms(text: str) -> list[_NumericTerm]:
    return [
        _NumericTerm(
            value=to_decimal(match.group(0).replace(",", "")),
            start=match.start(),
            end=match.end(),
        )
        for match in _NUMBER_RE.finditer(text)
    ]


def _format_decimal(value: Decimal) -> tuple[str, bool]:
    """Format a Decimal for display/TTS, rounding only beyond 12 places."""
    if not value.is_finite():
        raise MathOperationError("The result is not a finite number.")
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f"), False

    text = format(value, "f")
    fractional = text.partition(".")[2]
    if len(fractional) <= 12:
        return text.rstrip("0").rstrip("."), False

    rounded = value.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    return format(rounded, "f").rstrip("0").rstrip("."), rounded != value


def _value_result(
    category: str,
    operation: str,
    expression: str,
    value: Decimal,
    note: str = "",
    currency: str | None = None,
) -> CalculationResult:
    if currency:
        rendered, approximate = _format_amount(value, currency), False
    else:
        rendered, approximate = _format_decimal(value)
    qualifier = "approximately " if approximate else ""
    response_text = f"{expression} equals {qualifier}{rendered}."
    return CalculationResult(
        category=category,
        operation=operation,
        expression=expression,
        value=value,
        response_text=f"{response_text} {note}" if note else response_text,
        currency=currency,
    )


def _currency_symbol(query: str) -> str | None:
    match = _CURRENCY_RE.search(query)
    if match is None:
        return None
    token = match.group(0)
    return _CURRENCY_WORD_SYMBOLS.get(token.lower(), token)


def _format_amount(value: Decimal, symbol: str) -> str:
    return f"{symbol}{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,.2f}"


def _money(value: Decimal, currency: str | None) -> str:
    return _format_amount(value, currency) if currency else _format_decimal(value)[0]


_CURRENCY_BEFORE_RE = re.compile(r"[$\u20ac\u00a3\u00a5]\s*$")
_CURRENCY_AFTER_RE = re.compile(r"^\s*(?:dollars?|usd)\b", re.I)


def _monetary_values(query: str) -> set[Decimal]:
    """Numbers carrying a currency marker of their own, on either side."""
    values: set[Decimal] = set()
    for term in _numeric_terms(query):
        if _CURRENCY_BEFORE_RE.search(query[: term.start]) or _CURRENCY_AFTER_RE.match(
            query[term.end :]
        ):
            values.add(term.value)
    return values


def _operand(value: Decimal, currency: str | None, monetary: set[Decimal]) -> str:
    """Money only if this number was given as money; a bare number stays scalar."""
    return _money(value, currency if value in monetary else None)


def _money_amount(value: Decimal, currency: str | None) -> str:
    """Render a financial figure to cents, with the currency symbol when known."""
    return _format_amount(value, currency or "")


@dataclass(frozen=True)
class _FinancialInputs:
    principal: Decimal
    rate: Decimal
    years: Decimal
    duration_label: str
    periods: int
    frequency_label: str


def _plain_number(value: Decimal) -> str:
    return _format_decimal(value)[0]


def _frequency_label(periods: int) -> str:
    return _FREQUENCY_LABELS.get(periods, f"{periods} times per year")


def _payment_label(periods: int) -> str:
    return _PAYMENT_LABELS.get(periods, f"{periods}-times-per-year")


def _financial_inputs(query: str, default_periods: int, default_label: str) -> _FinancialInputs | None:
    """Extract principal, rate, term, and compounding cadence, or ``None``.

    Every number is matched to a role by the unit beside it, so a stray count
    like "compounded 12 times per year" never masquerades as the principal. A
    missing rate or duration means the query is not a self-contained calculation
    and is left to retrieval.
    """
    terms = _numeric_terms(query)
    consumed: set[int] = set()

    def consume(match: re.Match[str] | None) -> _NumericTerm | None:
        if match is None:
            return None
        start, end = match.span(1)
        for index, term in enumerate(terms):
            if index not in consumed and term.start >= start and term.end <= end:
                consumed.add(index)
                return term
        return None

    rate_term = consume(re.search(rf"({_NUMBER})\s*(?:{_RATE_UNIT})", query, re.I))
    if rate_term is None:
        return None

    years_term = consume(re.search(rf"({_NUMBER})\s*(?:years?|yrs?)\b", query, re.I))
    months_term = consume(re.search(rf"({_NUMBER})\s*(?:months?|mos?)\b", query, re.I))
    if years_term is not None:
        years = years_term.value
        duration_label = f"{_plain_number(years)} {'year' if years == 1 else 'years'}"
    elif months_term is not None:
        months = months_term.value
        years = ArithmeticOperations.divide(months, 12)
        duration_label = f"{_plain_number(months)} {'month' if months == 1 else 'months'}"
    else:
        return None

    times = _COMPOUND_TIMES_RE.search(query)
    if times is not None and int(times.group(1)) > 0:
        consume(times)
        periods = int(times.group(1))
        frequency_label = _frequency_label(periods)
    else:
        periods, frequency_label = default_periods, default_label
        for pattern, candidate_periods, label in _COMPOUND_FREQUENCIES:
            if pattern.search(query):
                periods, frequency_label = candidate_periods, label
                break

    monetary = _monetary_values(query)
    remaining = [term for index, term in enumerate(terms) if index not in consumed]
    money_terms = [term.value for term in remaining if term.value in monetary]
    if money_terms:
        principal = max(money_terms)
    elif len(remaining) == 1:
        principal = remaining[0].value
    else:
        return None

    return _FinancialInputs(principal, rate_term.value, years, duration_label, periods, frequency_label)


def _probability_result(operation: str, expression: str, value: Decimal) -> CalculationResult:
    rendered, approximate = _format_decimal(value)
    percent, percent_approximate = _format_decimal(value * Decimal(100))
    qualifier = "approximately " if approximate else ""
    percent_qualifier = "approximately " if percent_approximate else ""
    return CalculationResult(
        category="probability",
        operation=operation,
        expression=expression,
        value=value,
        response_text=(
            f"{expression} is {qualifier}{rendered}, or "
            f"{percent_qualifier}{percent} percent."
        ),
    )


def _percentage_rate_result(
    expression: str,
    value: Decimal,
    operation: str = "percentage rate",
) -> CalculationResult:
    rendered, approximate = _format_decimal(value)
    qualifier = "approximately " if approximate else ""
    return CalculationResult(
        category="arithmetic",
        operation=operation,
        expression=expression,
        value=value,
        response_text=f"{expression} equals {qualifier}{rendered} percent.",
    )


def _finite_probability_result(
    operation: str,
    expression: str,
    favorable: int,
    total: int,
) -> CalculationResult:
    value = ProbabilityOperations.favorable_over_total(favorable, total)
    decimal_value = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    percent_value = (value * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return CalculationResult(
        category="probability",
        operation=operation,
        expression=expression,
        value=value,
        response_text=(
            f"{expression} is {favorable} out of {total}, or approximately "
            f"{decimal_value} ({percent_value} percent)."
        ),
    )


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _cardinal_value(token: str) -> int | None:
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return _CARDINALS.get(token)


def _cardinal_before(tokens: list[str], index: int) -> int | None:
    if index < 1:
        return None
    value = _cardinal_value(tokens[index - 1])
    if value is None:
        return None
    if index >= 2 and tokens[index - 2] in _CARDINALS:
        leading = _CARDINALS[tokens[index - 2]]
        if leading >= 20 and leading % 10 == 0 and value < 10:
            return leading + value
    return value


def _object_count_before(tokens: list[str], index: int) -> int | None:
    for offset in range(1, 4):
        position = index - offset
        if position < 0:
            return None
        token = tokens[position]
        if token in {"of", "a", "an", "the"}:
            continue
        if token in _MULTIPLE_OBJECT_WORDS:
            return 2
        return _cardinal_value(token)
    return None


def _uniform_sample_space(tokens: list[str]) -> _UniformSampleSpace | None:
    """Extract a supported uniform experiment without depending on sentence order."""
    for object_index, token in enumerate(tokens):
        if token not in {"die", "dice"}:
            continue
        for sided_index in range(object_index - 1, max(-1, object_index - 4), -1):
            if tokens[sided_index] != "sided":
                continue
            sides = _cardinal_before(tokens, sided_index)
            if sides is not None:
                _reject_multiple_objects(_object_count_before(tokens, sided_index - 1))
                return _UniformSampleSpace("die", sides)
        _reject_multiple_objects(_object_count_before(tokens, object_index))
        return _UniformSampleSpace("die", _STANDARD_DIE_SIDES)
    for object_index, token in enumerate(tokens):
        if token in {"coin", "coins"}:
            _reject_multiple_objects(_object_count_before(tokens, object_index))
            return _UniformSampleSpace("coin", 2)
    return None


def _reject_multiple_objects(count: int | None) -> None:
    if count is not None and count > 1:
        raise MathOperationError(
            "Only a single die or coin is supported; state the favorable and total outcomes instead."
        )


def _has_conditioned_event(tokens: list[str]) -> bool:
    if _EVENT_QUALIFIERS.intersection(tokens):
        return True
    joined = " ".join(tokens)
    return bool(
        re.search(r"\b(?:at\s+(?:least|most)|more\s+than|less\s+than)\b", joined)
        or re.search(r"\bfrom\s+\w+\s+to\s+\w+\b", joined)
    )


def _explicit_die_outcome(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if token not in _OUTCOME_VERBS:
            continue
        for candidate_index in range(index + 1, min(len(tokens), index + 4)):
            candidate = tokens[candidate_index]
            if candidate in {"a", "an", "the"}:
                continue
            value = _cardinal_value(candidate)
            if value is None:
                break
            if candidate_index + 1 < len(tokens) and tokens[candidate_index + 1] == "sided":
                break
            return value
    return None


def _event_predicates(query: str, tokens: list[str]) -> list[tuple[str, Callable[[int], bool]]]:
    """Build face predicates for the supported conditioned events."""
    predicates: list[tuple[str, Callable[[int], bool]]] = []
    if "even" in tokens:
        predicates.append(("an even number", lambda face: face % 2 == 0))
    if "odd" in tokens:
        predicates.append(("an odd number", lambda face: face % 2 == 1))
    for match in _THRESHOLD_RE.finditer(query):
        comparison = " ".join(match.group(1).lower().split())
        bound = _cardinal_value(match.group(2).lower())
        if bound is None:
            continue
        if comparison in {"greater than", "more than", "larger than", "above", "over"}:
            predicates.append((f"a number greater than {bound}", lambda face, b=bound: face > b))
        elif comparison in {"less than", "fewer than", "smaller than", "below", "under"}:
            predicates.append((f"a number less than {bound}", lambda face, b=bound: face < b))
        elif comparison == "at least":
            predicates.append((f"a number of at least {bound}", lambda face, b=bound: face >= b))
        elif comparison == "at most":
            predicates.append((f"a number of at most {bound}", lambda face, b=bound: face <= b))
    return predicates


def _dataset_after(text: str) -> list[Decimal] | None:
    """Read an explicitly supplied list of numbers, or ``None`` when the text is prose."""
    cleaned = _STAT_LEAD_RE.sub("", text).strip().rstrip("?.!").strip()
    if not cleaned:
        return None
    residue = _CURRENCY_RE.sub(" ", cleaned)
    residue = _LIST_SEPARATOR_RE.sub(" ", _NUMBER_RE.sub(" ", residue))
    if residue.strip():
        return None
    values = _numbers(cleaned)
    return values if len(values) >= 2 else None


def _ratio_parts_with_span(
    text: str, allow_to: bool
) -> tuple[list[Decimal] | None, tuple[int, int] | None]:
    """Parse a ratio written as "a:b:c" (or "a to b to c") into its terms and span."""
    colon = _RATIO_COLON_LIST_RE.search(text)
    if colon is not None:
        parts = [_ratio_decimal(part) for part in colon.group(0).split(":")]
        return parts, colon.span()
    if allow_to:
        listed = _RATIO_TO_LIST_RE.search(text)
        if listed is not None:
            parts = [
                _ratio_decimal(part)
                for part in re.split(r"\s+to\s+", listed.group(0), flags=re.I)
            ]
            return parts, listed.span()
    return None, None


def _error_result(category: str, operation: str, expression: str, message: str) -> CalculationResult:
    return CalculationResult(category, operation, expression, None, f"I could not calculate that: {message}")


class _ExpressionEvaluator:
    """Evaluate only decimal literals and the four supported binary operators."""

    _OPERATORS: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
        ast.Add: lambda left, right: ArithmeticOperations.add([left, right]),
        ast.Sub: ArithmeticOperations.subtract,
        ast.Mult: lambda left, right: ArithmeticOperations.multiply([left, right]),
        ast.Div: ArithmeticOperations.divide,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Decimal:
        if len(expression) > 200:
            raise MathOperationError("The expression is too long.")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise MathOperationError("The arithmetic expression is not valid.") from exc
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise MathOperationError("The expression is too complex.")
        return cls._evaluate_node(tree.body, expression)

    @classmethod
    def _evaluate_node(cls, node: ast.AST, source: str) -> Decimal:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            literal = ast.get_source_segment(source, node)
            if literal is None:
                raise MathOperationError("A number in the expression could not be read.")
            return to_decimal(literal)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = cls._evaluate_node(node.operand, source)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in cls._OPERATORS:
            left = cls._evaluate_node(node.left, source)
            right = cls._evaluate_node(node.right, source)
            return cls._OPERATORS[type(node.op)](left, right)
        raise MathOperationError("Only addition, subtraction, multiplication, and division are supported.")


class MathQueryRouter:
    """Recognize and execute self-contained queries in the supported math scope."""

    _STATISTICS = (
        (re.compile(r"\b(?:population\s+|sample\s+)?standard\s+deviation\b|\bstd\.?\s+dev(?:iation)?\b", re.I),
         "standard deviation"),
        (re.compile(r"\b(?:population\s+|sample\s+)?variance\b", re.I), "variance"),
        (re.compile(r"\b(?:mean|average)\b", re.I), "mean"),
        (re.compile(r"\bmedian\b", re.I), "median"),
        (re.compile(r"\bmode\b", re.I), "mode"),
    )
    _ESTIMATORS = {
        ("variance", False): ("population variance", StatisticsOperations.population_variance),
        ("variance", True): ("sample variance", StatisticsOperations.sample_variance),
        ("standard deviation", False): (
            "population standard deviation",
            StatisticsOperations.population_standard_deviation,
        ),
        ("standard deviation", True): (
            "sample standard deviation",
            StatisticsOperations.sample_standard_deviation,
        ),
        ("mean", False): ("mean", StatisticsOperations.mean),
        ("median", False): ("median", StatisticsOperations.median),
        ("mode", False): ("mode", StatisticsOperations.mode),
    }

    def solve(
        self,
        user_query: str,
        prior_turn: dict[str, object] | None = None,
    ) -> CalculationResult | None:
        query = " ".join((user_query or "").strip().split())
        if not query or _DOCUMENT_CONTEXT_RE.search(query):
            return None

        previous_value, previous_currency = _prior_context(prior_turn)
        currency = _currency_symbol(query) or previous_currency

        chained = self._try_chain(query, previous_value, currency)
        if chained is not None:
            return chained
        return self._solve_step(query, previous_value, currency)

    def _try_chain(
        self,
        query: str,
        previous_value: Decimal | None,
        currency: str | None,
    ) -> CalculationResult | None:
        """Evaluate "divide that by 4 and take away 27%" one step at a time."""
        segments = [segment.strip() for segment in CHAIN_SPLIT_RE.split(query) if segment.strip()]
        if len(segments) < 2:
            return None

        running = previous_value
        expressions: list[str] = []
        for index, segment in enumerate(segments):
            # A later step that stands on its own is a separate question, not a continuation.
            if index and self._solve_step(segment, None, currency) is not None:
                return None
            result = self._solve_step(segment, running, currency)
            if result is None or result.value is None:
                return None
            expressions.append(result.expression)
            running = result.value
        if running is None:
            return None
        return _value_result(
            "arithmetic",
            "chained calculation",
            ", then ".join(expressions),
            running,
            currency=currency,
        )

    def _solve_step(
        self,
        query: str,
        previous_value: Decimal | None,
        currency: str | None,
    ) -> CalculationResult | None:
        candidates = [query]
        if previous_value is not None:
            resolved = _resolve_reference(query, previous_value, currency)
            if resolved != query:
                candidates.insert(0, resolved)
        for candidate in candidates:
            result = self._run_parsers(candidate, previous_value, currency)
            if result is not None:
                return result
        return None

    def _run_parsers(
        self,
        query: str,
        previous_value: Decimal | None,
        currency: str | None,
    ) -> CalculationResult | None:
        for parser in (
            self._try_probability,
            lambda text: self._try_statistics(text, currency),
            lambda text: self._try_amortization(text, currency),
            lambda text: self._try_compound_interest(text, currency),
            lambda text: self._try_percentage(text, previous_value, currency),
            lambda text: self._try_ratio(text, currency),
            lambda text: self._try_word_arithmetic(text, currency),
            lambda text: self._try_symbolic_arithmetic(text, currency),
        ):
            try:
                result = parser(query)
            except MathOperationError as exc:
                return _error_result("math", "invalid", query, str(exc))
            if result is not None:
                return result
        return None

    def _try_probability(self, query: str) -> CalculationResult | None:
        if not re.search(r"\b(?:probability|chance)\b", query, re.I):
            return None

        binomial = re.search(
            rf"\bexactly\s+(?P<successes>\d+)\s+success(?:es)?\s+(?:in|out\s+of)\s+"
            rf"(?P<trials>\d+)\s+trials?.*?success\s+probability(?:\s+of|\s+is)?\s+"
            rf"(?P<probability>{_NUMBER})\s*(?P<percent>%)?",
            query,
            re.I,
        )
        if binomial:
            probability = to_decimal(binomial.group("probability").replace(",", ""))
            if binomial.group("percent"):
                probability = ArithmeticOperations.divide(probability, 100)
            successes = int(binomial.group("successes"))
            trials = int(binomial.group("trials"))
            value = ProbabilityOperations.exact_binomial(successes, trials, probability)
            expression = f"The probability of exactly {successes} successes in {trials} trials"
            return _probability_result("exact binomial", expression, value)

        favorable = re.search(
            rf"(?P<favorable>{_NUMBER})\s+(?:favorable\s+outcomes?\s+)?out\s+of\s+"
            rf"(?P<total>{_NUMBER})(?:\s+total)?(?:\s+outcomes?)?",
            query,
            re.I,
        )
        if favorable:
            favorable_count = favorable.group("favorable").replace(",", "")
            total_count = favorable.group("total").replace(",", "")
            value = ProbabilityOperations.favorable_over_total(favorable_count, total_count)
            expression = f"The probability of {favorable_count} favorable outcomes out of {total_count} total outcomes"
            return _probability_result("favorable over total", expression, value)

        tokens = _tokenize(query)
        sample_space = _uniform_sample_space(tokens)
        if sample_space is not None:
            if sample_space.size < 1:
                raise MathOperationError("A uniform sample space must contain at least one outcome.")

            if sample_space.kind == "die":
                outcome = _explicit_die_outcome(tokens)
                if outcome is not None:
                    expression = (
                        f"The probability of rolling {outcome} on a fair {sample_space.size}-sided die"
                    )
                    if outcome < 1 or outcome > sample_space.size:
                        return _probability_result("uniform finite outcome", expression, Decimal(0))
                    return _finite_probability_result(
                        "uniform finite outcome", expression, 1, sample_space.size
                    )

            if _has_conditioned_event(tokens):
                if sample_space.kind == "die":
                    predicates = _event_predicates(query, tokens)
                    if predicates:
                        faces = range(1, sample_space.size + 1)
                        favorable = sum(
                            1 for face in faces if all(test(face) for _, test in predicates)
                        )
                        description = " and ".join(label for label, _ in predicates)
                        return _finite_probability_result(
                            "uniform finite event",
                            f"The probability of rolling {description} on a {sample_space.size}-sided die",
                            favorable,
                            sample_space.size,
                        )
                raise MathOperationError(
                    "State the number of favorable outcomes for a conditioned event."
                )

            if sample_space.kind == "coin" and any(
                _cardinal_before(tokens, index) not in (None, 1)
                for index, token in enumerate(tokens)
                if token in {"head", "heads", "tail", "tails"}
            ):
                raise MathOperationError(
                    "For repeated coin tosses, specify an exact binomial event and the number of trials."
                )

            # No stated outcome or condition: the question is about one elementary outcome.
            expression = (
                f"The probability of rolling any specific number on a {sample_space.size}-sided die"
                if sample_space.kind == "die"
                else "The probability of one specified side of a fair coin"
            )
            return _finite_probability_result(
                "uniform finite outcome", expression, 1, sample_space.size
            )

        if _NUMBER_RE.search(query):
            raise MathOperationError(
                "Use favorable outcomes out of total outcomes, or specify an exact binomial event."
            )
        return None

    def _try_statistics(self, query: str, currency: str | None = None) -> CalculationResult | None:
        for pattern, statistic in self._STATISTICS:
            match = pattern.search(query)
            if not match:
                continue
            values = _dataset_after(query[match.end():])
            if values is None:
                return None
            sample = bool(re.search(r"\bsample\b", query, re.I))
            operation, function = self._ESTIMATORS.get(
                (statistic, sample), self._ESTIMATORS[(statistic, False)]
            )
            value = function(values)
            listed = ", ".join(
                _operand(number, currency, _monetary_values(query)) for number in values
            )
            return _value_result(
                "statistics", operation, f"The {operation} of {listed}", value, currency=currency
            )
        return None

    def _try_compound_interest(self, query: str, currency: str | None = None) -> CalculationResult | None:
        if not _COMPOUND_RE.search(query):
            return None
        inputs = _financial_inputs(query, default_periods=1, default_label="annually")
        if inputs is None:
            return None

        amount = FinancialOperations.compound_amount(
            inputs.principal, inputs.rate, inputs.years, inputs.periods
        )
        interest = amount - inputs.principal
        principal_text = _money_amount(inputs.principal, currency)
        amount_text = _money_amount(amount, currency)
        interest_text = _money_amount(interest, currency)
        rate_text = _plain_number(inputs.rate)
        terms = (
            f"{principal_text} at {rate_text} percent for {inputs.duration_label} "
            f"compounded {inputs.frequency_label}"
        )
        # Answer the interest earned only when the question asks for it and not the balance.
        wants_interest = bool(re.search(r"\binterest\b", query, re.I)) and not re.search(
            r"\b(?:future\s+value|balance|worth|total\s+amount|grow[a-z]*)\b", query, re.I
        )
        if wants_interest:
            expression = f"The compound interest on {terms}"
            response_text = f"{expression} is {interest_text}, for a future value of {amount_text}."
            return CalculationResult("finance", "compound interest", expression, interest, response_text, currency)
        expression = f"The future value of {terms}"
        response_text = f"{expression} is {amount_text}, including {interest_text} of compound interest."
        return CalculationResult("finance", "future value", expression, amount, response_text, currency)

    def _try_amortization(self, query: str, currency: str | None = None) -> CalculationResult | None:
        if not _AMORTIZE_RE.search(query):
            return None
        inputs = _financial_inputs(query, default_periods=12, default_label="monthly")
        if inputs is None:
            return None

        payment, _total_paid, total_interest, count = FinancialOperations.amortization_summary(
            inputs.principal, inputs.rate, inputs.years, inputs.periods
        )
        expression = (
            f"The {_payment_label(inputs.periods)} payment on a "
            f"{_money_amount(inputs.principal, currency)} loan at {_plain_number(inputs.rate)} "
            f"percent for {inputs.duration_label}"
        )
        response_text = (
            f"{expression} is {_money_amount(payment, currency)}, with "
            f"{_money_amount(total_interest, currency)} in total interest over "
            f"{_plain_number(count)} payments."
        )
        return CalculationResult("finance", "loan payment", expression, payment, response_text, currency)

    def _try_ratio(self, query: str, currency: str | None = None) -> CalculationResult | None:
        proportion = self._ratio_proportion(query, currency)
        if proportion is not None:
            return proportion
        share = self._ratio_share(query, currency)
        if share is not None:
            return share
        return self._ratio_simplify(query)

    def _ratio_proportion(self, query: str, currency: str | None = None) -> CalculationResult | None:
        match = _PROPORTION_RE.search(query)
        if match is None:
            return None
        terms: list[Decimal | None] = []
        for token in match.groups():
            if re.fullmatch(_RATIO_UNKNOWN, token, re.I):
                terms.append(None)
            else:
                terms.append(_ratio_decimal(token))
        if sum(1 for term in terms if term is None) != 1:
            return None
        value = RatioOperations.solve_proportion(terms)
        labels = [_money(term, currency) if term is not None else "x" for term in terms]
        expression = (
            f"The unknown in the proportion {labels[0]} to {labels[1]} "
            f"as {labels[2]} to {labels[3]}"
        )
        if currency:
            rendered, qualifier = _format_amount(value, currency), ""
        else:
            rendered, approximate = _format_decimal(value)
            qualifier = "approximately " if approximate else ""
        return CalculationResult(
            "ratio",
            "solve proportion",
            expression,
            value,
            f"{expression} is {qualifier}{rendered}.",
            currency,
        )

    def _ratio_share(self, query: str, currency: str | None) -> CalculationResult | None:
        if not _RATIO_SHARE_RE.search(query):
            return None
        has_word = bool(_RATIO_WORD_RE.search(query))
        if not (has_word or _RATIO_COLON_LIST_RE.search(query)):
            return None
        parts, span = _ratio_parts_with_span(query, allow_to=has_word)
        if parts is None or span is None or len(parts) < 2:
            return None
        outside = [
            term for term in _numeric_terms(query) if term.end <= span[0] or term.start >= span[1]
        ]
        if not outside:
            return None
        total = outside[0].value
        shares = RatioOperations.divide_in_ratio(total, parts)
        ratio_text = " to ".join(_plain_number(part) for part in parts)
        shares_text = " and ".join(_money(share, currency) for share in shares)
        expression = f"Dividing {_money(total, currency)} in the ratio {ratio_text}"
        return CalculationResult(
            "ratio", "divide in ratio", expression, None, f"{expression} gives {shares_text}.", currency
        )

    def _ratio_simplify(self, query: str) -> CalculationResult | None:
        if not (_RATIO_SIMPLIFY_RE.search(query) or _RATIO_WORD_RE.search(query)):
            return None
        parts, _span = _ratio_parts_with_span(query, allow_to=True)
        if parts is None or len(parts) < 2:
            return None
        in_text = " to ".join(_plain_number(part) for part in parts)
        simplified = RatioOperations.simplify(parts)
        out_text = " to ".join(_plain_number(part) for part in simplified)
        expression = f"The ratio {in_text} in simplest form"
        return CalculationResult(
            "ratio", "simplify ratio", expression, None, f"{expression} is {out_text}."
        )

    def _try_percent_change(
        self,
        query: str,
        terms: list[_NumericTerm],
        currency: str | None = None,
    ) -> CalculationResult | None:
        direction = _PERCENT_CHANGE_RE.search(query)
        if direction is None:
            return None
        separator = re.search(r"\bto\b", query, re.I)
        if separator is None:
            return None
        original, updated = terms
        if original.end > separator.start() or updated.start < separator.end():
            return None

        signed = ArithmeticOperations.percentage_change(original.value, updated.value)
        word = direction.group(1).lower()
        decreasing = word.startswith(_DECREASE_WORDS)
        label = "decrease" if decreasing else "increase"
        value = -signed if decreasing else signed
        return _percentage_rate_result(
            f"The percentage {label} from {_money(original.value, currency)} "
            f"to {_money(updated.value, currency)}",
            value,
            operation="percentage change",
        )

    def _try_percent_application(
        self,
        query: str,
        marker: re.Match[str],
        terms: list[_NumericTerm],
        previous_value: Decimal | None,
        currency: str | None = None,
    ) -> CalculationResult | None:
        increasing = bool(_APPLY_INCREASE_RE.search(query))
        decreasing = bool(_APPLY_DECREASE_RE.search(query))
        if increasing == decreasing:
            return None

        rate = next(
            (
                term
                for term in terms
                if term.end <= marker.start() and not query[term.end:marker.start()].strip()
            ),
            None,
        )
        if rate is None:
            return None
        base = next((term.value for term in terms if term is not rate), previous_value)
        if base is None:
            return None

        value = ArithmeticOperations.apply_percentage(base, rate.value, increase=increasing)
        verb = "increased" if increasing else "reduced"
        return _value_result(
            "arithmetic",
            "percentage application",
            f"{_money(base, currency)} {verb} by {rate.value} percent",
            value,
            note="" if increasing or not _NET_RESULT_RE.search(query) else _NET_DEDUCTION_NOTE,
            currency=currency,
        )

    def _try_percentage(
        self,
        query: str,
        previous_value: Decimal | None = None,
        currency: str | None = None,
    ) -> CalculationResult | None:
        marker = _PERCENT_MARKER_RE.search(query)
        if marker is None:
            return None

        terms = _numeric_terms(query)
        if len(terms) == 2:
            change = self._try_percent_change(query, terms, currency)
            if change is not None:
                return change

        applied = self._try_percent_application(query, marker, terms, previous_value, currency)
        if applied is not None:
            return applied

        if len(terms) != 2:
            return None

        adjacent_rate = next(
            (
                term
                for term in terms
                if term.end <= marker.start()
                and not query[term.end:marker.start()].strip()
            ),
            None,
        )
        asks_for_rate = adjacent_rate is None or bool(
            re.search(r"\b(?:what|which)\s*$", query[:marker.start()], re.I)
        )

        of_matches = list(re.finditer(r"\bof\b", query, re.I))
        whole = None
        for of_match in reversed(of_matches):
            whole = next((term for term in terms if term.start >= of_match.end()), None)
            if whole is not None:
                break
        if whole is None:
            return None

        if asks_for_rate:
            part = next((term for term in terms if term is not whole), None)
            if part is None:
                return None
            result = ArithmeticOperations.percentage_rate(part.value, whole.value)
            return _percentage_rate_result(
                f"{_money(part.value, currency)} as a percentage of {_money(whole.value, currency)}",
                result,
            )

        value = next((term for term in terms if term is not adjacent_rate), None)
        if value is None:
            return None
        result = ArithmeticOperations.percentage_of(adjacent_rate.value, value.value)
        return _value_result(
            "arithmetic",
            "percentage of total",
            f"{adjacent_rate.value} percent of {_money(value.value, currency)}",
            result,
            currency=currency,
        )

    def _try_word_arithmetic(self, query: str, currency: str | None = None) -> CalculationResult | None:
        core = self._strip_prefixes(query)
        monetary = _monetary_values(query)
        patterns: tuple[tuple[re.Pattern[str], str], ...] = (
            (re.compile(rf"^subtract\s+(?P<a>{_NUMBER})\s+from\s+(?P<b>{_NUMBER})$", re.I), "subtract_from"),
            (re.compile(rf"^(?:difference\s+(?:between|of)|subtract)\s+(?P<a>{_NUMBER})\s+(?:and|minus)\s+(?P<b>{_NUMBER})$", re.I), "subtract"),
            (re.compile(rf"^(?:divide)\s+(?P<a>{_NUMBER})\s+by\s+(?P<b>{_NUMBER})$", re.I), "divide"),
            (re.compile(rf"^(?:quotient\s+of)\s+(?P<a>{_NUMBER})\s+(?:and|by)\s+(?P<b>{_NUMBER})$", re.I), "divide"),
        )
        for pattern, operation in patterns:
            match = pattern.fullmatch(core)
            if not match:
                continue
            left = to_decimal(match.group("a").replace(",", ""))
            right = to_decimal(match.group("b").replace(",", ""))
            if operation == "subtract_from":
                value = ArithmeticOperations.subtract(right, left)
                return _value_result(
                    "arithmetic",
                    "subtraction",
                    f"{_operand(right, currency, monetary)} minus "
                    f"{_operand(left, currency, monetary)}",
                    value,
                    currency=currency,
                )
            if operation == "subtract":
                value = ArithmeticOperations.subtract(left, right)
                return _value_result(
                    "arithmetic",
                    "subtraction",
                    f"{_operand(left, currency, monetary)} minus "
                    f"{_operand(right, currency, monetary)}",
                    value,
                    currency=currency,
                )
            value = ArithmeticOperations.divide(left, right)
            return _value_result(
                "arithmetic",
                "division",
                f"{_operand(left, currency, monetary)} divided by "
                f"{_operand(right, currency, monetary)}",
                value,
                currency=currency,
            )

        list_match = re.fullmatch(r"(?P<operation>add|sum(?:\s+of)?|multiply|product(?:\s+of)?)\s+(?P<values>.+)", core, re.I)
        if not list_match:
            return None
        values = _numbers(list_match.group("values"))
        if len(values) < 2:
            raise MathOperationError("At least two numbers are required.")
        operation = list_match.group("operation").lower()
        listed = ", ".join(_operand(number, currency, monetary) for number in values)
        if operation.startswith(("add", "sum")):
            return _value_result(
                "arithmetic",
                "addition",
                f"The sum of {listed}",
                ArithmeticOperations.add(values),
                currency=currency,
            )
        return _value_result(
            "arithmetic",
            "multiplication",
            f"The product of {listed}",
            ArithmeticOperations.multiply(values),
            currency=currency,
        )

    def _try_symbolic_arithmetic(self, query: str, currency: str | None = None) -> CalculationResult | None:
        expression = to_operator_symbols(self._strip_prefixes(query))
        expression = expression.strip().rstrip("?.!").strip()
        expression = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", expression)
        if _DATE_LIKE_RE.fullmatch(expression):
            return None
        # A dangling leading operator means the left operand was never supplied.
        if LEADING_OPERATOR_RE.match(expression):
            return None
        if not re.fullmatch(r"[\d.\s()+\-*/]+", expression):
            return None
        if not re.search(r"[+\-*/]", expression) or not re.search(r"\d", expression):
            return None
        value = _ExpressionEvaluator.evaluate(expression)
        spoken_expression = expression.replace("*", " times ").replace("/", " divided by ").replace("+", " plus ")
        spoken_expression = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", spoken_expression)
        spoken_expression = " ".join(spoken_expression.split())
        return _value_result("arithmetic", "expression", spoken_expression, value, currency=currency)

    @staticmethod
    def _strip_prefixes(query: str) -> str:
        core = _CURRENCY_RE.sub(" ", query)
        core = _REQUEST_PREFIX_RE.sub("", core.strip()).strip()
        core = _ACTION_PREFIX_RE.sub("", core).strip()
        core = _ARTICLE_PREFIX_RE.sub("", core).strip()
        return core.rstrip("?.!").strip()


def _resolve_reference(
    query: str,
    previous_value: Decimal,
    currency: str | None = None,
) -> str:
    """Rewrite a follow-up so it names the number the previous turn produced."""
    # The symbol keeps the carried value marked as money without rounding it to cents.
    rendered = f"{currency or ''}{_format_decimal(previous_value)[0]}"
    named = REFERENCE_RE.sub(rendered, query, count=1)
    if named != query:
        return named

    core = MathQueryRouter._strip_prefixes(query)
    if _PERCENT_MARKER_RE.search(core):
        # The percentage lane already falls back to the previous value for its base.
        return query
    operators = to_operator_symbols(core)
    if LEADING_OPERATOR_RE.match(operators):
        return f"{rendered} {operators}"
    return query


_ROUTER = MathQueryRouter()


def solve_math_query(
    user_query: str,
    prior_turn: dict[str, object] | None = None,
) -> CalculationResult | None:
    """Return a deterministic answer for supported math, otherwise ``None``."""
    return _ROUTER.solve(user_query, prior_turn)


def previous_calculation_value(turn: dict[str, object] | None) -> Decimal | None:
    """Recover the verified numeric result a previous turn produced, if any."""
    metadata = (turn or {}).get("calculation")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("value")
    if not raw:
        return None
    try:
        return to_decimal(str(raw))
    except MathOperationError:
        return None


def suggest_math_correction(
    user_query: str,
    prior_turn: dict[str, object] | None = None,
) -> tuple[str, str, CalculationResult] | None:
    """Return the misspelled operator, its spelling, and the result it unlocks."""
    if not any(character.isdigit() for character in user_query or ""):
        return None
    # Only the respelling that makes the query calculable counts as a suggestion.
    for corrected, word, suggestion in suggest_operator_corrections(user_query):
        calculation = _ROUTER.solve(corrected, prior_turn)
        if calculation is not None and calculation.value is not None:
            return word, suggestion, calculation
    return None


def _prior_context(turn: dict[str, object] | None) -> tuple[Decimal | None, str | None]:
    metadata = (turn or {}).get("calculation")
    if not isinstance(metadata, dict):
        return None, None
    currency = metadata.get("currency")
    return previous_calculation_value(turn), currency if isinstance(currency, str) else None
