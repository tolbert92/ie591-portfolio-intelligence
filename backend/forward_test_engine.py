"""IE 591 Skill #7: calculate a buy-and-hold paper portfolio."""

from __future__ import annotations

from datetime import date
from math import fsum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .forward_test_models import ForwardTestRequest, PositiveAmount
from .visualization_models import FiniteNumber, Nonblank

AccountValue = Annotated[FiniteNumber, Field(ge=0)]


class ForwardTestPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    date: date
    portfolio_value: AccountValue
    portfolio_cumulative_return: FiniteNumber
    benchmark_value: AccountValue | None = None
    benchmark_cumulative_return: FiniteNumber | None = None


class ForwardTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    test_id: Nonblank
    portfolio_id: Nonblank
    initial_value: PositiveAmount
    is_synthetic: bool
    benchmark_id: Nonblank | None
    points: tuple[ForwardTestPoint, ...] = Field(min_length=1)


def run_forward_test(request: ForwardTestRequest) -> ForwardTestResult:
    """Compound each starting holding independently, without rebalancing."""
    initial_value = request.initial_value
    allocations = request.portfolio.allocations
    total_weight = fsum(row.weight for row in allocations)
    holdings = {
        row.ticker: initial_value * (row.weight / total_weight)
        for row in allocations
    }
    benchmark_value = initial_value if request.benchmark_id is not None else None
    points = []

    for observation in request.observations:
        for row in observation.asset_returns:
            holdings[row.ticker] *= 1 + row.daily_return
        portfolio_value = fsum(holdings.values())

        if benchmark_value is not None and observation.benchmark_return is not None:
            benchmark_value *= 1 + observation.benchmark_return

        points.append(
            ForwardTestPoint(
                date=observation.date,
                portfolio_value=portfolio_value,
                portfolio_cumulative_return=portfolio_value / initial_value - 1,
                benchmark_value=benchmark_value,
                benchmark_cumulative_return=(
                    benchmark_value / initial_value - 1 if benchmark_value is not None else None
                ),
            )
        )

    return ForwardTestResult(
        test_id=request.test_id,
        portfolio_id=request.portfolio.portfolio_id,
        initial_value=initial_value,
        is_synthetic=(
            request.returns_source.is_synthetic or request.portfolio.provenance.is_synthetic
        ),
        benchmark_id=request.benchmark_id,
        points=tuple(points),
    )


__all__ = ["ForwardTestPoint", "ForwardTestResult", "run_forward_test"]
