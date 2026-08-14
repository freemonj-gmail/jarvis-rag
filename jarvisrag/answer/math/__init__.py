"""Deterministic arithmetic, probability, and statistics for Jarvis answers."""

from .operations import (
    ArithmeticOperations,
    MathOperationError,
    ProbabilityOperations,
    StatisticsOperations,
    FinancialOperations,
)
from .router import (
    CalculationResult,
    MathQueryRouter,
    previous_calculation_value,
    solve_math_query,
    suggest_math_correction,
)

__all__ = [
    "ArithmeticOperations",
    "CalculationResult",
    "FinancialOperations",
    "MathOperationError",
    "MathQueryRouter",
    "ProbabilityOperations",
    "StatisticsOperations",
    "previous_calculation_value",
    "solve_math_query",
    "suggest_math_correction",
]
