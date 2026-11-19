
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.2.2 — Constrained Optimization Solver

Implements:
- Minimum-variance optimization
- Maximum-Sharpe optimization
- Long-only, fully invested allocations
- Individual asset weight limits
- Minimum expected return constraints
- Maximum annual volatility constraints
- Covariance matrix validation
- Solver convergence and feasibility checks
- Deterministic multistart optimization

SciPy is imported when the solver is executed.
This allows the module to be imported before
the project's SciPy dependency is installed.

Efficient frontier generation and standardized
OptimizationResponse integration are reserved
for subsequent stages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .optimization_models import (
    OptimizationObjective,
    OptimizationRequest,
    PortfolioMetrics,
)
from .optimization_engine import (
    OptimizationEngineError,
    calculate_portfolio_metrics,
    validate_covariance_matrix,
    validate_weights,
)


FEASIBILITY_TOLERANCE = 1e-6
SHARPE_VOLATILITY_FLOOR = 1e-12


class OptimizationSolverError(OptimizationEngineError):
    """Raised when optimization cannot produce a valid result."""


@dataclass(frozen=True)
class SolverResult:
    """
    Validated result of one optimization objective.

    This is an internal solver result, not the
    final public OptimizationResponse contract.
    """

    objective: OptimizationObjective
    weights: tuple[float, ...]
    metrics: PortfolioMetrics
    success: bool
    solver_message: str
    iterations: int
    objective_value: float
    maximum_constraint_violation: float
    starting_points_attempted: int


def _load_scipy_minimize():
    """
    Load SciPy only when optimization is requested.

    A missing dependency produces an actionable
    error rather than breaking module imports.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise OptimizationSolverError(
            "SciPy is required for constrained "
            "portfolio optimization. Install "
            "the scipy Python package."
        ) from exc

    return minimize


def _finite_float(value: Any, label: str) -> float:
    """Require a finite numerical value."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizationSolverError(
            f"{label} must be finite."
        ) from exc

    if not math.isfinite(result):
        raise OptimizationSolverError(
            f"{label} must be finite."
        )

    return result


def _portfolio_return(
    weights: Sequence[float],
    expected_returns: Sequence[float],
) -> float:
    """Calculate expected annual portfolio return."""
    return math.fsum(
        float(weight) * expected_return
        for weight, expected_return
        in zip(weights, expected_returns)
    )


def _portfolio_variance(
    weights: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> float:
    """Calculate portfolio variance using w-transpose Sigma w."""
    count = len(weights)

    return math.fsum(
        float(weights[i])
        * covariance[i][j]
        * float(weights[j])
        for i in range(count)
        for j in range(count)
    )


def _portfolio_volatility(
    weights: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> float:
    """Calculate nonnegative annual portfolio volatility."""
    variance = _portfolio_variance(weights, covariance)

    return math.sqrt(max(0.0, variance))


def _bounded_start(
    lower: Sequence[float],
    upper: Sequence[float],
    preferred: Sequence[float],
) -> list[float]:
    """
    Build a fully invested starting portfolio
    within individual asset weight bounds.

    This does not guarantee satisfaction of
    portfolio-level return or risk constraints.
    """
    count = len(lower)

    weights = [float(value) for value in lower]

    remaining = 1.0 - math.fsum(weights)

    if remaining < -FEASIBILITY_TOLERANCE:
        raise OptimizationSolverError(
            "Minimum allocation weights exceed 100%."
        )

    capacity = math.fsum(
        upper[i] - lower[i]
        for i in range(count)
    )

    if capacity < remaining - FEASIBILITY_TOLERANCE:
        raise OptimizationSolverError(
            "Maximum allocation weights cannot "
            "support a fully invested portfolio."
        )

    remaining = max(0.0, remaining)

    # Allocate remaining weight toward the
    # preferred portfolio while respecting bounds.
    for i in range(count):
        desired = max(
            0.0,
            float(preferred[i]) - weights[i],
        )

        addition = min(
            desired,
            upper[i] - weights[i],
            remaining,
        )

        weights[i] += addition
        remaining -= addition

    # Distribute any remaining weight across
    # assets with available capacity.
    for i in range(count):
        if remaining <= 1e-12:
            break

        addition = min(
            upper[i] - weights[i],
            remaining,
        )

        weights[i] += addition
        remaining -= addition

    if abs(math.fsum(weights) - 1.0) > 1e-8:
        raise OptimizationSolverError(
            "Unable to construct a fully "
            "invested starting portfolio."
        )

    return weights


def _starting_portfolios(
    request: OptimizationRequest,
) -> list[list[float]]:
    """
    Generate deterministic starting allocations.

    Multiple starts reduce dependence on one
    initial guess, particularly for Sharpe
    maximization and constrained problems.
    """
    count = len(request.assets)

    lower = [
        asset.minimum_weight
        for asset in request.assets
    ]

    upper = [
        asset.maximum_weight
        for asset in request.assets
    ]

    preferred_vectors = [
        [1.0 / count] * count,
    ]

    # Include one asset-focused starting point
    # for each asset, subject to allocation bounds.
    for i in range(count):
        preferred = [0.0] * count
        preferred[i] = 1.0
        preferred_vectors.append(preferred)

    starts = []
    seen = set()

    for preferred in preferred_vectors:
        weights = _bounded_start(
            lower,
            upper,
            preferred,
        )

        key = tuple(
            round(weight, 12)
            for weight in weights
        )

        if key not in seen:
            seen.add(key)
            starts.append(weights)

    return starts


def _constraint_violation(
    weights: Sequence[float],
    request: OptimizationRequest,
    covariance: Sequence[Sequence[float]],
    expected_returns: Sequence[float],
) -> float:
    """
    Return the maximum violation across
    investment and portfolio constraints.
    """
    values = [
        _finite_float(weight, "portfolio weight")
        for weight in weights
    ]

    violations = [
        abs(math.fsum(values) - 1.0),
    ]

    for weight, asset in zip(
        values,
        request.assets,
    ):
        violations.append(
            max(0.0, asset.minimum_weight - weight)
        )

        violations.append(
            max(0.0, weight - asset.maximum_weight)
        )

    portfolio_return = _portfolio_return(
        values,
        expected_returns,
    )

    portfolio_volatility = _portfolio_volatility(
        values,
        covariance,
    )

    minimum_return = (
        request.constraints.minimum_expected_annual_return
    )

    maximum_volatility = (
        request.constraints.maximum_annual_volatility
    )

    if minimum_return is not None:
        violations.append(
            max(
                0.0,
                minimum_return - portfolio_return,
            )
        )

    if maximum_volatility is not None:
        violations.append(
            max(
                0.0,
                portfolio_volatility - maximum_volatility,
            )
        )

    return max(violations)


def _make_constraints(
    request: OptimizationRequest,
    covariance: Sequence[Sequence[float]],
    expected_returns: Sequence[float],
) -> list[dict[str, Any]]:
    """Construct SciPy SLSQP constraints."""
    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda w: math.fsum(w) - 1.0,
        }
    ]

    minimum_return = (
        request.constraints.minimum_expected_annual_return
    )

    if minimum_return is not None:
        constraints.append({
            "type": "ineq",
            "fun": (
                lambda w, target=minimum_return:
                _portfolio_return(w, expected_returns)
                - target
            ),
        })

    maximum_volatility = (
        request.constraints.maximum_annual_volatility
    )

    if maximum_volatility is not None:
        # Variance form avoids the square-root
        # derivative singularity at zero volatility.
        maximum_variance = maximum_volatility ** 2

        constraints.append({
            "type": "ineq",
            "fun": (
                lambda w, limit=maximum_variance:
                limit
                - _portfolio_variance(w, covariance)
            ),
        })

    return constraints


def _objective_function(
    objective: OptimizationObjective,
    covariance: Sequence[Sequence[float]],
    expected_returns: Sequence[float],
    risk_free_rate: float,
):
    """Return the requested numerical objective."""

    if objective == OptimizationObjective.MINIMUM_VARIANCE:
        return lambda w: _portfolio_variance(
            w,
            covariance,
        )

    if objective == OptimizationObjective.MAXIMUM_SHARPE:

        def negative_sharpe(weights):
            volatility = _portfolio_volatility(
                weights,
                covariance,
            )

            excess_return = (
                _portfolio_return(
                    weights,
                    expected_returns,
                )
                - risk_free_rate
            )

            if volatility <= SHARPE_VOLATILITY_FLOOR:
                # A zero-volatility portfolio has
                # undefined Sharpe ratio.
                # Use a finite penalty to guide
                # the numerical search away from it.
                return 1e12

            return -(excess_return / volatility)

        return negative_sharpe

    raise OptimizationSolverError(
        f"Unsupported objective: {objective}"
    )


def solve_portfolio(
    request: OptimizationRequest,
    objective: OptimizationObjective | str | None = None,
) -> SolverResult:
    """
    Solve a constrained portfolio optimization problem.

    Parameters
    ----------
    request:
        Validated OptimizationRequest containing
        assets, annualized covariance, constraints,
        configuration, and provenance.

    objective:
        Optional override for the configured
        optimization objective.

    Returns
    -------
    SolverResult:
        Validated optimal candidate with allocation,
        metrics, convergence information, and
        numerical constraint diagnostics.

    Raises
    ------
    OptimizationSolverError:
        If dependencies are missing, inputs are
        invalid, or no acceptable candidate
        can be established.

    Notes
    -----
    SLSQP is a local numerical optimization method.
    A successful result does not constitute proof
    of global optimality for every possible
    maximum-Sharpe problem.
    """
    if not isinstance(request, OptimizationRequest):
        raise OptimizationSolverError(
            "request must be an OptimizationRequest."
        )

    try:
        selected_objective = (
            OptimizationObjective(objective)
            if objective is not None
            else request.config.objective
        )
    except ValueError as exc:
        raise OptimizationSolverError(
            "Unsupported optimization objective."
        ) from exc

    count = len(request.assets)

    try:
        covariance = validate_covariance_matrix(
            request.covariance_matrix,
            count,
        )
    except OptimizationEngineError as exc:
        raise OptimizationSolverError(str(exc)) from exc

    expected_returns = [
        _finite_float(
            asset.expected_annual_return,
            f"{asset.ticker} expected return",
        )
        for asset in request.assets
    ]

    risk_free_rate = _finite_float(
        request.config.risk_free_annual_rate,
        "risk-free rate",
    )

    bounds = [
        (
            asset.minimum_weight,
            asset.maximum_weight,
        )
        for asset in request.assets
    ]

    starts = _starting_portfolios(request)

    scipy_constraints = _make_constraints(
        request,
        covariance,
        expected_returns,
    )

    objective_function = _objective_function(
        selected_objective,
        covariance,
        expected_returns,
        risk_free_rate,
    )

    minimize = _load_scipy_minimize()

    accepted_candidates = []
    failure_messages = []

    for start in starts:
        try:
            result = minimize(
                objective_function,
                x0=start,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints,
                options={
                    "maxiter": (
                        request.config.maximum_iterations
                    ),
                    "ftol": (
                        request.config.solver_tolerance
                    ),
                    "disp": False,
                },
            )
        except (
            ValueError,
            ArithmeticError,
            FloatingPointError,
        ) as exc:
            failure_messages.append(str(exc))
            continue

        if not result.success:
            failure_messages.append(
                str(result.message)
            )
            continue

        try:
            weights = [
                _finite_float(
                    value,
                    "optimized weight",
                )
                for value in result.x
            ]

            if len(weights) != count:
                raise OptimizationSolverError(
                    "Solver returned an incorrect "
                    "number of weights."
                )

            violation = _constraint_violation(
                weights,
                request,
                covariance,
                expected_returns,
            )

            if violation > FEASIBILITY_TOLERANCE:
                raise OptimizationSolverError(
                    "Solver result violates "
                    "portfolio constraints."
                )

            # Verify the stricter allocation
            # contract before calculating metrics.
            validate_weights(
                weights,
                request,
            )

            metrics = calculate_portfolio_metrics(
                request,
                weights,
            )

            if (
                selected_objective
                == OptimizationObjective.MAXIMUM_SHARPE
                and metrics.sharpe_ratio is None
            ):
                raise OptimizationSolverError(
                    "Maximum-Sharpe result has "
                    "undefined Sharpe ratio."
                )

            value = _finite_float(
                objective_function(weights),
                "objective value",
            )

            accepted_candidates.append({
                "weights": weights,
                "metrics": metrics,
                "value": value,
                "iterations": int(result.nit),
                "message": str(result.message),
                "violation": violation,
            })

        except (
            OptimizationEngineError,
            ValueError,
            ArithmeticError,
        ) as exc:
            failure_messages.append(str(exc))

    if not accepted_candidates:
        detail = (
            failure_messages[0]
            if failure_messages
            else "No feasible candidate was found."
        )

        raise OptimizationSolverError(
            "Portfolio optimization did not produce "
            f"an acceptable solution. {detail}"
        )

    # Both objectives are formulated as minimization.
    best = min(
        accepted_candidates,
        key=lambda candidate: candidate["value"],
    )

    return SolverResult(
        objective=selected_objective,
        weights=tuple(best["weights"]),
        metrics=best["metrics"],
        success=True,
        solver_message=best["message"],
        iterations=best["iterations"],
        objective_value=best["value"],
        maximum_constraint_violation=best["violation"],
        starting_points_attempted=len(starts),
    )
