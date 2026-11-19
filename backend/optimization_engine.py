
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.2 — Stage 4.2.1: Core Engine Foundation

Provides:
- Covariance matrix validation
- Positive-semidefinite matrix verification
- Portfolio allocation validation
- Expected portfolio return
- Annualized portfolio variance
- Annualized portfolio volatility
- Sharpe ratio
- Standardized portfolio metrics

This stage does not execute numerical optimization
or generate an efficient frontier.
"""

from __future__ import annotations

import math
from typing import Sequence

from .optimization_models import (
    OptimizationRequest,
    PortfolioMetrics,
)


CALCULATION_VERSION = "1.0.0"

WEIGHT_TOLERANCE = 1e-8
MATRIX_TOLERANCE = 1e-8
PSD_TOLERANCE = 1e-10


class OptimizationEngineError(ValueError):
    """Raised when portfolio calculations are invalid."""


def _require_finite(value: float, name: str) -> float:
    """Convert a number to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizationEngineError(
            f"{name} must be a finite number."
        ) from exc

    if not math.isfinite(result):
        raise OptimizationEngineError(
            f"{name} must be a finite number."
        )

    return result


def _validate_vector(
    values: Sequence[float],
    name: str,
) -> list[float]:
    """Validate a nonempty numerical vector."""
    if not values:
        raise OptimizationEngineError(
            f"{name} cannot be empty."
        )

    return [
        _require_finite(value, f"{name}[{i}]")
        for i, value in enumerate(values)
    ]


def _minimum_eigenvalue(
    matrix: list[list[float]],
) -> float:
    """
    Estimate the smallest eigenvalue of a symmetric
    matrix using Jacobi rotations.

    Suitable for the small educational portfolio
    matrices used in the initial application.
    """
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    a = [row.copy() for row in matrix]

    scale = max(
        1.0,
        max(abs(value) for row in a for value in row),
    )

    tolerance = 1e-13 * scale
    maximum_rotations = max(100, 20 * n * n * n)

    for _ in range(maximum_rotations):
        p = 0
        q = 1
        largest = abs(a[p][q])

        for i in range(n):
            for j in range(i + 1, n):
                value = abs(a[i][j])

                if value > largest:
                    largest = value
                    p, q = i, j

        if largest <= tolerance:
            return min(a[i][i] for i in range(n))

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        tau = (aqq - app) / (2.0 * apq)

        t = math.copysign(
            1.0, tau
        ) / (abs(tau) + math.sqrt(1.0 + tau * tau))

        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq

        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            if k == p or k == q:
                continue

            akp = a[k][p]
            akq = a[k][q]

            a[k][p] = c * akp - s * akq
            a[p][k] = a[k][p]

            a[k][q] = s * akp + c * akq
            a[q][k] = a[k][q]

    raise OptimizationEngineError(
        "Covariance eigenvalue validation "
        "did not converge."
    )


def validate_covariance_matrix(
    covariance_matrix: Sequence[Sequence[float]],
    asset_count: int,
) -> list[list[float]]:
    """
    Validate a covariance matrix.

    Requirements:
    - Square
    - Matches asset count
    - Finite values
    - Symmetric
    - Nonnegative diagonal
    - Positive semidefinite

    Returns a validated numerical matrix.
    """
    if asset_count < 1:
        raise OptimizationEngineError(
            "Asset count must be positive."
        )

    if len(covariance_matrix) != asset_count:
        raise OptimizationEngineError(
            "Covariance matrix dimensions "
            "must match asset count."
        )

    matrix = []

    for i, row in enumerate(covariance_matrix):
        if len(row) != asset_count:
            raise OptimizationEngineError(
                "Covariance matrix must be square."
            )

        matrix.append([
            _require_finite(
                value,
                f"covariance[{i}][{j}]",
            )
            for j, value in enumerate(row)
        ])

    for i in range(asset_count):
        if matrix[i][i] < 0:
            raise OptimizationEngineError(
                "Covariance diagonal cannot be negative."
            )

        for j in range(i + 1, asset_count):
            if not math.isclose(
                matrix[i][j],
                matrix[j][i],
                rel_tol=MATRIX_TOLERANCE,
                abs_tol=MATRIX_TOLERANCE,
            ):
                raise OptimizationEngineError(
                    "Covariance matrix must be symmetric."
                )

            # Remove tiny numerical asymmetry.
            average = (
                matrix[i][j] + matrix[j][i]
            ) / 2.0

            matrix[i][j] = average
            matrix[j][i] = average

    scale = max(
        1.0,
        max(abs(value) for row in matrix for value in row),
    )

    minimum_eigenvalue = _minimum_eigenvalue(matrix)

    if minimum_eigenvalue < -PSD_TOLERANCE * scale:
        raise OptimizationEngineError(
            "Covariance matrix must be "
            "positive semidefinite."
        )

    return matrix


def validate_weights(
    weights: Sequence[float],
    request: OptimizationRequest,
) -> list[float]:
    """
    Validate a fully invested, long-only portfolio.

    Checks individual asset bounds and the
    requirement that all weights sum to one.
    """
    values = _validate_vector(weights, "weights")

    if len(values) != len(request.assets):
        raise OptimizationEngineError(
            "Number of weights must match assets."
        )

    for weight, asset in zip(values, request.assets):
        if weight < -WEIGHT_TOLERANCE:
            raise OptimizationEngineError(
                f"Negative weight is not allowed: "
                f"{asset.ticker}"
            )

        if (
            weight < asset.minimum_weight - WEIGHT_TOLERANCE
            or weight > asset.maximum_weight + WEIGHT_TOLERANCE
        ):
            raise OptimizationEngineError(
                f"Weight violates allocation bounds: "
                f"{asset.ticker}"
            )

    if not math.isclose(
        sum(values),
        1.0,
        rel_tol=WEIGHT_TOLERANCE,
        abs_tol=WEIGHT_TOLERANCE,
    ):
        raise OptimizationEngineError(
            "Portfolio weights must sum to 1.0."
        )

    return values


def calculate_expected_return(
    weights: Sequence[float],
    expected_returns: Sequence[float],
) -> float:
    """
    Calculate annualized expected portfolio return.

    E(Rp) = sum(weight_i * expected_return_i)
    """
    w = _validate_vector(weights, "weights")
    r = _validate_vector(
        expected_returns,
        "expected_returns",
    )

    if len(w) != len(r):
        raise OptimizationEngineError(
            "Weights and expected returns "
            "must have equal lengths."
        )

    return math.fsum(
        weight * expected_return
        for weight, expected_return in zip(w, r)
    )


def calculate_portfolio_variance(
    weights: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate annualized portfolio variance.

    Variance = w_transpose * covariance * w
    """
    w = _validate_vector(weights, "weights")

    matrix = validate_covariance_matrix(
        covariance_matrix,
        len(w),
    )

    variance = math.fsum(
        w[i] * matrix[i][j] * w[j]
        for i in range(len(w))
        for j in range(len(w))
    )

    if variance < -PSD_TOLERANCE:
        raise OptimizationEngineError(
            "Calculated portfolio variance "
            "cannot be negative."
        )

    return max(0.0, variance)


def calculate_portfolio_volatility(
    weights: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """Calculate annualized portfolio volatility."""
    variance = calculate_portfolio_variance(
        weights,
        covariance_matrix,
    )

    return math.sqrt(variance)


def calculate_sharpe_ratio(
    expected_annual_return: float,
    annualized_volatility: float,
    risk_free_annual_rate: float = 0.0,
) -> float | None:
    """
    Calculate the annualized Sharpe ratio.

    Sharpe = (expected return - risk-free rate)
             / annualized volatility

    Returns None for zero volatility.
    """
    expected_return = _require_finite(
        expected_annual_return,
        "expected_annual_return",
    )

    volatility = _require_finite(
        annualized_volatility,
        "annualized_volatility",
    )

    risk_free_rate = _require_finite(
        risk_free_annual_rate,
        "risk_free_annual_rate",
    )

    if volatility < 0:
        raise OptimizationEngineError(
            "Volatility cannot be negative."
        )

    if volatility == 0:
        return None

    return (
        expected_return - risk_free_rate
    ) / volatility


def calculate_portfolio_metrics(
    request: OptimizationRequest,
    weights: Sequence[float],
) -> PortfolioMetrics:
    """
    Calculate standardized metrics for a portfolio.

    Uses validated annualized Market Intelligence
    inputs and verifies allocation constraints.

    Also enforces any portfolio-level minimum
    return or maximum volatility limits.
    """
    w = validate_weights(weights, request)

    expected_returns = [
        asset.expected_annual_return
        for asset in request.assets
    ]

    expected_return = calculate_expected_return(
        w,
        expected_returns,
    )

    variance = calculate_portfolio_variance(
        w,
        request.covariance_matrix,
    )

    volatility = math.sqrt(variance)

    constraints = request.constraints

    maximum_volatility = (
        constraints.maximum_annual_volatility
    )

    if maximum_volatility is not None:
        if volatility > (
            maximum_volatility + WEIGHT_TOLERANCE
        ):
            raise OptimizationEngineError(
                "Portfolio exceeds maximum "
                "annual volatility."
            )

    minimum_return = (
        constraints.minimum_expected_annual_return
    )

    if minimum_return is not None:
        if expected_return < (
            minimum_return - WEIGHT_TOLERANCE
        ):
            raise OptimizationEngineError(
                "Portfolio does not meet minimum "
                "expected annual return."
            )

    sharpe = calculate_sharpe_ratio(
        expected_return,
        volatility,
        request.config.risk_free_annual_rate,
    )

    return PortfolioMetrics(
        expected_annual_return=expected_return,
        annualized_volatility=volatility,
        annualized_variance=variance,
        sharpe_ratio=sharpe,
    )


def equal_weight_portfolio(
    request: OptimizationRequest,
) -> list[float]:
    """
    Generate an equal-weight baseline portfolio.

    This is a benchmark, not an optimized portfolio.
    Raises an error if equal weights violate
    asset allocation constraints.
    """
    count = len(request.assets)

    weights = [1.0 / count] * count

    return validate_weights(weights, request)
