
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.2.3 — Efficient Frontier Generation

Generate minimum-variance portfolios across
a feasible range of expected annual returns.

Synthetic or user-supplied validated inputs.
No external market data or AI credentials required.
"""

from __future__ import annotations

import math

from .optimization_models import (
    EfficientFrontierPoint,
    OptimizationRequest,
    OptimizedAllocation,
)
from .optimization_solver import (
    OptimizationSolverError,
    solve_portfolio,
)


class EfficientFrontierError(OptimizationSolverError):
    """Raised when a valid frontier cannot be generated."""


def _maximum_feasible_return(
    request: OptimizationRequest,
) -> float:
    """
    Maximize expected return under asset weight
    bounds and the fully-invested constraint.

    This linear allocation calculation does not
    incorporate the portfolio volatility cap.
    The numerical solver handles that constraint.
    """
    assets = request.assets

    weights = [
        asset.minimum_weight
        for asset in assets
    ]

    remaining = 1.0 - math.fsum(weights)

    if remaining < -1e-8:
        raise EfficientFrontierError(
            "Minimum weights exceed full investment."
        )

    order = sorted(
        range(len(assets)),
        key=lambda i: (
            -assets[i].expected_annual_return,
            i,
        ),
    )

    for i in order:
        if remaining <= 1e-12:
            break

        capacity = (
            assets[i].maximum_weight - weights[i]
        )

        addition = min(
            max(0.0, capacity),
            remaining,
        )

        weights[i] += addition
        remaining -= addition

    if remaining > 1e-8:
        raise EfficientFrontierError(
            "Asset bounds cannot support full investment."
        )

    return math.fsum(
        weight * asset.expected_annual_return
        for weight, asset in zip(weights, assets)
    )


def _request_for_target(
    request: OptimizationRequest,
    target_return: float,
) -> OptimizationRequest:
    """
    Create an independent request with an updated
    minimum expected annual return.

    Existing risk limits and allocation bounds
    are preserved.
    """
    constraints = request.constraints.model_copy(
        update={
            "minimum_expected_annual_return": (
                target_return
            ),
        }
    )

    return request.model_copy(
        update={"constraints": constraints}
    )


def _make_frontier_point(
    request: OptimizationRequest,
    target_return: float,
    result,
) -> EfficientFrontierPoint:
    """Convert a validated solver result to a point."""
    allocations = [
        OptimizedAllocation(
            ticker=asset.ticker,
            weight=float(weight),
        )
        for asset, weight in zip(
            request.assets,
            result.weights,
        )
    ]

    return EfficientFrontierPoint(
        target_annual_return=target_return,
        expected_annual_return=(
            result.metrics.expected_annual_return
        ),
        annualized_volatility=(
            result.metrics.annualized_volatility
        ),
        allocations=allocations,
    )


def generate_efficient_frontier(
    request: OptimizationRequest,
    point_count: int | None = None,
) -> list[EfficientFrontierPoint]:
    """
    Generate a validated long-only efficient frontier.

    The lower endpoint is the minimum-variance
    portfolio under the request's constraints.

    The upper endpoint is the maximum expected
    return permitted by the asset bounds, unless
    a volatility cap prevents reaching it.

    For each target return, the existing solver
    minimizes variance subject to that return.

    Returns points ordered by increasing target
    expected annual return.

    Raises EfficientFrontierError if the frontier
    cannot be fully generated.
    """
    if not isinstance(request, OptimizationRequest):
        raise EfficientFrontierError(
            "request must be an OptimizationRequest."
        )

    count = (
        request.config.efficient_frontier_points
        if point_count is None
        else point_count
    )

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 2 <= count <= 200
    ):
        raise EfficientFrontierError(
            "point_count must be an integer "
            "between 2 and 200."
        )

    try:
        minimum_result = solve_portfolio(
            request,
            objective="minimum_variance",
        )
    except OptimizationSolverError as exc:
        raise EfficientFrontierError(
            "Cannot establish the minimum-variance "
            f"frontier endpoint: {exc}"
        ) from exc

    lower_return = (
        minimum_result.metrics.expected_annual_return
    )

    upper_return = _maximum_feasible_return(request)

    if upper_return < lower_return - 1e-7:
        raise EfficientFrontierError(
            "Maximum feasible return is below "
            "the minimum-variance portfolio return."
        )

    # If all feasible allocations have effectively
    # the same expected return, there is only one
    # distinct efficient portfolio to report.
    if math.isclose(
        lower_return,
        upper_return,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        return [
            _make_frontier_point(
                request,
                lower_return,
                minimum_result,
            )
        ]

    frontier = []

    for index in range(count):
        fraction = index / (count - 1)

        target = (
            lower_return
            + fraction * (upper_return - lower_return)
        )

        # Avoid numerical differences at the lower
        # endpoint by reusing its validated solution.
        if index == 0:
            result = minimum_result
        else:
            target_request = _request_for_target(
                request,
                target,
            )

            try:
                result = solve_portfolio(
                    target_request,
                    objective="minimum_variance",
                )
            except OptimizationSolverError as exc:
                raise EfficientFrontierError(
                    "Efficient frontier could not "
                    f"reach target return {target:.8f}: "
                    f"{exc}"
                ) from exc

        if (
            result.metrics.expected_annual_return
            < target - 1e-6
        ):
            raise EfficientFrontierError(
                "Frontier point failed its "
                "minimum-return constraint."
            )

        if (
            abs(math.fsum(result.weights) - 1.0)
            > 1e-6
        ):
            raise EfficientFrontierError(
                "Frontier weights do not sum to one."
            )

        frontier.append(
            _make_frontier_point(
                request,
                target,
                result,
            )
        )

    return frontier
