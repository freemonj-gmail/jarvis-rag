"""Deterministic operations supported by Jarvis's calculator lane."""

from __future__ import annotations

from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from math import comb
from statistics import (
    StatisticsError,
    mean,
    median,
    mode,
    pstdev,
    pvariance,
    stdev,
    variance,
)
from typing import Iterable


Number = Decimal | int | float | str


class MathOperationError(ValueError):
    """Raised when a requested calculation cannot be completed."""


def to_decimal(value: Number) -> Decimal:
    """Convert a supported numeric value without inheriting float artifacts."""
    if isinstance(value, bool):
        raise MathOperationError("Boolean values are not supported.")
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MathOperationError(f"Invalid number: {value!r}.") from exc


class ArithmeticOperations:
    """Basic decimal arithmetic: addition, subtraction, multiplication, division."""

    @staticmethod
    def add(values: Iterable[Number]) -> Decimal:
        numbers = [to_decimal(value) for value in values]
        if not numbers:
            raise MathOperationError("Addition requires at least one number.")
        return sum(numbers, Decimal(0))

    @staticmethod
    def subtract(minuend: Number, subtrahend: Number) -> Decimal:
        return to_decimal(minuend) - to_decimal(subtrahend)

    @staticmethod
    def multiply(values: Iterable[Number]) -> Decimal:
        numbers = [to_decimal(value) for value in values]
        if not numbers:
            raise MathOperationError("Multiplication requires at least one number.")
        product = Decimal(1)
        for number in numbers:
            product *= number
        return product

    @staticmethod
    def divide(dividend: Number, divisor: Number) -> Decimal:
        try:
            with localcontext() as context:
                context.prec = 28
                return to_decimal(dividend) / to_decimal(divisor)
        except (DivisionByZero, ZeroDivisionError) as exc:
            raise MathOperationError("Division by zero is undefined.") from exc

    @classmethod
    def percentage_of(cls, rate: Number, whole: Number) -> Decimal:
        return cls.multiply([cls.divide(rate, 100), whole])

    @classmethod
    def percentage_rate(cls, part: Number, whole: Number) -> Decimal:
        return cls.multiply([cls.divide(part, whole), 100])

    @classmethod
    def percentage_change(cls, original: Number, updated: Number) -> Decimal:
        """Signed percent change from ``original`` to ``updated``."""
        starting = to_decimal(original)
        if starting == 0:
            raise MathOperationError("Percent change from zero is undefined.")
        return cls.percentage_rate(cls.subtract(updated, starting), starting)

    @classmethod
    def apply_percentage(cls, base: Number, rate: Number, increase: bool = True) -> Decimal:
        """Raise or lower ``base`` by ``rate`` percent."""
        delta = cls.percentage_of(rate, base)
        return cls.add([base, delta]) if increase else cls.subtract(base, delta)


class StatisticsOperations:
    """Basic descriptive statistics over decimal values."""

    @staticmethod
    def _values(values: Iterable[Number], minimum: int = 1) -> list[Decimal]:
        numbers = [to_decimal(value) for value in values]
        if len(numbers) < minimum:
            noun = "number" if minimum == 1 else "numbers"
            raise MathOperationError(f"This statistic requires at least {minimum} {noun}.")
        return numbers

    @classmethod
    def mean(cls, values: Iterable[Number]) -> Decimal:
        return mean(cls._values(values))

    @classmethod
    def median(cls, values: Iterable[Number]) -> Decimal:
        return median(cls._values(values))

    @classmethod
    def mode(cls, values: Iterable[Number]) -> Decimal:
        numbers = cls._values(values)
        try:
            modes = set(numbers)
            result = mode(numbers)
            highest_frequency = numbers.count(result)
            tied = [number for number in modes if numbers.count(number) == highest_frequency]
            if highest_frequency == 1 or len(tied) != 1:
                raise MathOperationError("The data has no single mode.")
            return result
        except StatisticsError as exc:
            raise MathOperationError(str(exc)) from exc

    @classmethod
    def population_variance(cls, values: Iterable[Number]) -> Decimal:
        return pvariance(cls._values(values, minimum=2))

    @classmethod
    def population_standard_deviation(cls, values: Iterable[Number]) -> Decimal:
        return pstdev(cls._values(values, minimum=2))

    @classmethod
    def sample_variance(cls, values: Iterable[Number]) -> Decimal:
        return variance(cls._values(values, minimum=2))

    @classmethod
    def sample_standard_deviation(cls, values: Iterable[Number]) -> Decimal:
        return stdev(cls._values(values, minimum=2))


class ProbabilityOperations:
    """Basic probability for finite outcomes and exact binomial events."""

    @staticmethod
    def favorable_over_total(favorable: Number, total: Number) -> Decimal:
        favorable_count = to_decimal(favorable)
        total_count = to_decimal(total)
        if total_count <= 0:
            raise MathOperationError("Total outcomes must be greater than zero.")
        if favorable_count < 0 or favorable_count > total_count:
            raise MathOperationError("Favorable outcomes must be between zero and total outcomes.")
        return ArithmeticOperations.divide(favorable_count, total_count)

    @staticmethod
    def exact_binomial(successes: int, trials: int, success_probability: Number) -> Decimal:
        probability = to_decimal(success_probability)
        if trials < 0 or successes < 0 or successes > trials:
            raise MathOperationError("Successes must be between zero and the number of trials.")
        if probability < 0 or probability > 1:
            raise MathOperationError("Success probability must be between zero and one.")
        return Decimal(comb(trials, successes)) * probability**successes * (Decimal(1) - probability) ** (trials - successes)


class FinancialOperations:
    """Compound interest and fixed-rate loan amortization over decimal money."""

    @staticmethod
    def compound_amount(
        principal: Number,
        annual_rate_percent: Number,
        years: Number,
        periods_per_year: Number = 1,
    ) -> Decimal:
        """Future value of ``principal`` under periodic compounding."""
        starting = to_decimal(principal)
        rate = to_decimal(annual_rate_percent)
        term_years = to_decimal(years)
        periods = to_decimal(periods_per_year)
        if starting < 0:
            raise MathOperationError("The principal must not be negative.")
        if term_years < 0:
            raise MathOperationError("The number of years must not be negative.")
        if periods <= 0:
            raise MathOperationError("Compounding periods per year must be greater than zero.")
        with localcontext() as context:
            context.prec = 34
            growth = Decimal(1) + (rate / 100) / periods
            if growth <= 0:
                raise MathOperationError("The interest rate makes the growth factor non-positive.")
            return starting * (growth ** (periods * term_years))

    @classmethod
    def compound_interest(
        cls,
        principal: Number,
        annual_rate_percent: Number,
        years: Number,
        periods_per_year: Number = 1,
    ) -> Decimal:
        """Interest earned only, i.e. future value minus the principal."""
        starting = to_decimal(principal)
        return cls.compound_amount(principal, annual_rate_percent, years, periods_per_year) - starting

    @staticmethod
    def amortized_payment(
        principal: Number,
        annual_rate_percent: Number,
        years: Number,
        periods_per_year: Number = 12,
    ) -> Decimal:
        """Level payment that retires a fixed-rate loan over its term."""
        loan = to_decimal(principal)
        rate = to_decimal(annual_rate_percent)
        term_years = to_decimal(years)
        periods = to_decimal(periods_per_year)
        if loan < 0:
            raise MathOperationError("The loan principal must not be negative.")
        if term_years <= 0:
            raise MathOperationError("The loan term must be greater than zero.")
        if periods <= 0:
            raise MathOperationError("Payments per year must be greater than zero.")
        if rate < 0:
            raise MathOperationError("The interest rate must not be negative.")
        with localcontext() as context:
            context.prec = 34
            total_payments = periods * term_years
            periodic_rate = (rate / 100) / periods
            if periodic_rate == 0:
                return loan / total_payments
            growth = (Decimal(1) + periodic_rate) ** total_payments
            return loan * (periodic_rate * growth) / (growth - Decimal(1))

    @classmethod
    def amortization_summary(
        cls,
        principal: Number,
        annual_rate_percent: Number,
        years: Number,
        periods_per_year: Number = 12,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Return (payment, total paid, total interest, number of payments)."""
        loan = to_decimal(principal)
        payment = cls.amortized_payment(principal, annual_rate_percent, years, periods_per_year)
        number_of_payments = to_decimal(periods_per_year) * to_decimal(years)
        total_paid = payment * number_of_payments
        return payment, total_paid, total_paid - loan, number_of_payments
