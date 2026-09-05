"""Deterministic operations supported by Jarvis's calculator lane."""

from __future__ import annotations

from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from math import comb, gcd
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


class RatioOperations:
    """Simplify ratios, solve proportions, and divide a quantity by a ratio."""

    @staticmethod
    def _scaled_integers(values: list[Decimal]) -> list[int]:
        """Scale decimal ratio terms to whole numbers without losing precision."""
        places = 0
        for value in values:
            exponent = value.normalize().as_tuple().exponent
            if isinstance(exponent, int) and exponent < 0:
                places = max(places, -exponent)
        scale = Decimal(10) ** places
        return [int((value * scale).to_integral_value()) for value in values]

    @classmethod
    def simplify(cls, terms: Iterable[Number]) -> list[Decimal]:
        """Reduce a ratio to its lowest whole-number terms."""
        values = [to_decimal(term) for term in terms]
        if len(values) < 2:
            raise MathOperationError("A ratio needs at least two terms.")
        if any(value < 0 for value in values):
            raise MathOperationError("Ratio terms must not be negative.")
        if all(value == 0 for value in values):
            raise MathOperationError("A ratio cannot be all zeros.")
        scaled = cls._scaled_integers(values)
        divisor = 0
        for number in scaled:
            divisor = gcd(divisor, abs(number))
        divisor = divisor or 1
        return [Decimal(number // divisor) for number in scaled]

    @staticmethod
    def solve_proportion(terms: list[Number | None]) -> Decimal:
        """Solve a:b = c:d (given as [a, b, c, d]) for its single unknown (None)."""
        if len(terms) != 4:
            raise MathOperationError("A proportion has four terms: a to b as c to d.")
        unknowns = [index for index, term in enumerate(terms) if term is None]
        if len(unknowns) != 1:
            raise MathOperationError("A proportion needs exactly one unknown term.")
        a, b, c, d = (to_decimal(term) if term is not None else None for term in terms)
        # Cross-multiplication: a*d = b*c.
        position = unknowns[0]
        if position == 0:
            return ArithmeticOperations.divide(b * c, d)
        if position == 1:
            return ArithmeticOperations.divide(a * d, c)
        if position == 2:
            return ArithmeticOperations.divide(a * d, b)
        return ArithmeticOperations.divide(b * c, a)

    @staticmethod
    def divide_in_ratio(total: Number, parts: Iterable[Number]) -> list[Decimal]:
        """Split ``total`` into shares proportional to ``parts``."""
        amount = to_decimal(total)
        weights = [to_decimal(part) for part in parts]
        if len(weights) < 2:
            raise MathOperationError("A ratio needs at least two terms.")
        if any(weight < 0 for weight in weights):
            raise MathOperationError("Ratio terms must not be negative.")
        divisor = sum(weights, Decimal(0))
        if divisor == 0:
            raise MathOperationError("The ratio terms must not sum to zero.")
        return [ArithmeticOperations.divide(amount * weight, divisor) for weight in weights]


class BusinessOperations:
    """Everyday business, pricing, and interest-theory calculations."""

    # --- Pricing: markup and margin ---
    @staticmethod
    def markup_percent(cost: Number, price: Number) -> Decimal:
        cost_value = to_decimal(cost)
        if cost_value == 0:
            raise MathOperationError("Cost must not be zero.")
        return ArithmeticOperations.percentage_rate(to_decimal(price) - cost_value, cost_value)

    @staticmethod
    def margin_percent(cost: Number, price: Number) -> Decimal:
        price_value = to_decimal(price)
        if price_value == 0:
            raise MathOperationError("Price must not be zero.")
        return ArithmeticOperations.percentage_rate(price_value - to_decimal(cost), price_value)

    @staticmethod
    def price_from_markup(cost: Number, markup_percent: Number) -> Decimal:
        return ArithmeticOperations.apply_percentage(cost, markup_percent, increase=True)

    @staticmethod
    def price_from_margin(cost: Number, margin_percent: Number) -> Decimal:
        margin = to_decimal(margin_percent)
        if margin >= 100:
            raise MathOperationError("Margin must be less than 100 percent.")
        return ArithmeticOperations.divide(to_decimal(cost), Decimal(1) - margin / 100)

    # --- Simple interest ---
    @staticmethod
    def simple_interest(principal: Number, annual_rate_percent: Number, years: Number) -> Decimal:
        return ArithmeticOperations.multiply(
            [to_decimal(principal), to_decimal(annual_rate_percent) / 100, to_decimal(years)]
        )

    @classmethod
    def simple_interest_amount(
        cls, principal: Number, annual_rate_percent: Number, years: Number
    ) -> Decimal:
        return to_decimal(principal) + cls.simple_interest(principal, annual_rate_percent, years)

    # --- Present value (time value of money) ---
    @staticmethod
    def present_value(
        future_value: Number,
        annual_rate_percent: Number,
        years: Number,
        periods_per_year: Number = 1,
    ) -> Decimal:
        amount = to_decimal(future_value)
        rate = to_decimal(annual_rate_percent)
        term_years = to_decimal(years)
        periods = to_decimal(periods_per_year)
        if periods <= 0:
            raise MathOperationError("Compounding periods per year must be greater than zero.")
        with localcontext() as context:
            context.prec = 34
            growth = Decimal(1) + (rate / 100) / periods
            if growth <= 0:
                raise MathOperationError("The interest rate makes the growth factor non-positive.")
            return amount / (growth ** (periods * term_years))

    # --- Return on investment ---
    @staticmethod
    def roi_percent(net_gain: Number, cost: Number) -> Decimal:
        cost_value = to_decimal(cost)
        if cost_value == 0:
            raise MathOperationError("The cost basis must not be zero.")
        return ArithmeticOperations.percentage_rate(to_decimal(net_gain), cost_value)

    @classmethod
    def roi_from_final(cls, final_value: Number, cost: Number) -> Decimal:
        return cls.roi_percent(to_decimal(final_value) - to_decimal(cost), cost)

    # --- Compound annual growth rate ---
    @staticmethod
    def cagr_percent(begin: Number, end: Number, years: Number) -> Decimal:
        start = to_decimal(begin)
        finish = to_decimal(end)
        term_years = to_decimal(years)
        if start <= 0:
            raise MathOperationError("The beginning value must be greater than zero.")
        if finish <= 0:
            raise MathOperationError("The ending value must be greater than zero.")
        if term_years <= 0:
            raise MathOperationError("The number of years must be greater than zero.")
        with localcontext() as context:
            context.prec = 34
            growth = (finish / start) ** (Decimal(1) / term_years)
            return (growth - Decimal(1)) * 100

    # --- Sales tax / tip ---
    @staticmethod
    def rate_total(base: Number, rate_percent: Number) -> tuple[Decimal, Decimal]:
        """Return (amount added, total) for a tax/tip rate applied to ``base``."""
        base_value = to_decimal(base)
        amount = ArithmeticOperations.percentage_of(rate_percent, base_value)
        return amount, base_value + amount

    # --- Discounts ---
    @staticmethod
    def sale_price(original: Number, discount_percent: Number) -> Decimal:
        return ArithmeticOperations.apply_percentage(original, discount_percent, increase=False)

    @staticmethod
    def original_from_sale(sale: Number, discount_percent: Number) -> Decimal:
        discount = to_decimal(discount_percent)
        if discount >= 100:
            raise MathOperationError("The discount must be less than 100 percent.")
        return ArithmeticOperations.divide(to_decimal(sale), Decimal(1) - discount / 100)

    @classmethod
    def chained_discount_price(cls, base: Number, discounts: Iterable[Number]) -> Decimal:
        price = to_decimal(base)
        rates = [to_decimal(discount) for discount in discounts]
        if not rates:
            raise MathOperationError("At least one discount is required.")
        for discount in rates:
            price = cls.sale_price(price, discount)
        return price

    # --- Break-even ---
    @staticmethod
    def break_even_units(
        fixed_costs: Number, price_per_unit: Number, variable_cost_per_unit: Number
    ) -> Decimal:
        price = to_decimal(price_per_unit)
        variable = to_decimal(variable_cost_per_unit)
        if price <= variable:
            raise MathOperationError("Price per unit must exceed variable cost per unit.")
        return ArithmeticOperations.divide(to_decimal(fixed_costs), price - variable)

    # --- Straight-line depreciation ---
    @staticmethod
    def straight_line_depreciation(
        cost: Number, salvage: Number, useful_life_years: Number
    ) -> Decimal:
        life = to_decimal(useful_life_years)
        if life <= 0:
            raise MathOperationError("The useful life must be greater than zero.")
        depreciable = to_decimal(cost) - to_decimal(salvage)
        if depreciable < 0:
            raise MathOperationError("The salvage value must not exceed the cost.")
        return ArithmeticOperations.divide(depreciable, life)

    # --- Effective annual rate ---
    @staticmethod
    def effective_annual_rate(nominal_percent: Number, periods_per_year: Number) -> Decimal:
        periods = to_decimal(periods_per_year)
        if periods <= 0:
            raise MathOperationError("Compounding periods per year must be greater than zero.")
        with localcontext() as context:
            context.prec = 34
            growth = (Decimal(1) + (to_decimal(nominal_percent) / 100) / periods) ** periods
            return (growth - Decimal(1)) * 100

    # --- Doubling time (rule of 70 / 72) ---
    @staticmethod
    def doubling_time(growth_percent: Number, rule: Number = 72) -> Decimal:
        rate = to_decimal(growth_percent)
        if rate <= 0:
            raise MathOperationError("The growth rate must be greater than zero.")
        return ArithmeticOperations.divide(to_decimal(rule), rate)

