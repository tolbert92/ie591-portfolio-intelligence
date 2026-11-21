"""IE 591 Skill #8: simulate terminal values under geometric Brownian motion."""

from __future__ import annotations

from math import exp, isfinite, log, log1p, sqrt
from random import Random
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .monte_carlo_models import MonteCarloAssumptions, MonteCarloRequest, PositiveAmount
from .visualization_models import Nonblank, Nonnegative, Weight


class TerminalPercentiles(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    p05: Nonnegative
    p25: Nonnegative
    p50: Nonnegative
    p75: Nonnegative
    p95: Nonnegative


class MonteCarloResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    simulation_id: Nonblank
    model: Literal["geometric_brownian_motion"]
    initial_value: PositiveAmount
    horizon_years: int
    path_count: int
    seed: int
    assumptions: MonteCarloAssumptions
    goal_value: PositiveAmount | None
    goal_probability: Weight | None
    terminal_percentiles: TerminalPercentiles


def _percentile(ordered_values: list[float], probability: float) -> float:
    index = probability * (len(ordered_values) - 1)
    lower_index = int(index)
    lower = ordered_values[lower_index]
    upper = ordered_values[min(lower_index + 1, len(ordered_values) - 1)]
    return lower + (upper - lower) * (index - lower_index)


def run_monte_carlo(request: MonteCarloRequest) -> MonteCarloResult:
    """Sample the exact terminal distribution without stepping through each year."""
    volatility = request.assumptions.annualized_volatility
    annual_growth = log1p(request.assumptions.expected_annual_return)
    horizon = request.horizon_years
    drift = (annual_growth - (0.5 * volatility) * volatility) * horizon
    center = log(request.initial_value) + drift
    scale = volatility * sqrt(horizon)
    if not isfinite(center) or not isfinite(scale):
        raise ValueError("Simulation assumptions exceed the supported numeric range")

    random = Random(request.seed)
    try:
        if volatility == 0:
            try:
                growth = (1 + request.assumptions.expected_annual_return) ** horizon
            except OverflowError:
                growth = float('inf')
            terminal_value = request.initial_value * growth
            if not isfinite(terminal_value) or terminal_value == 0:
                terminal_value = exp(center)
            values = [terminal_value] * request.path_count
        else:
            values = sorted(
                exp(center + scale * random.gauss(0, 1))
                for _ in range(request.path_count)
            )
    except OverflowError as exc:
        raise ValueError("A simulated ending value exceeds the supported numeric range") from exc
    if not isfinite(values[-1]):
        raise ValueError("A simulated ending value exceeds the supported numeric range")

    goal = request.goal_value
    return MonteCarloResult(
        simulation_id=request.simulation_id,
        model=request.model,
        initial_value=request.initial_value,
        horizon_years=horizon,
        path_count=request.path_count,
        seed=request.seed,
        assumptions=request.assumptions,
        goal_value=goal,
        goal_probability=(
            sum(value >= goal for value in values) / request.path_count
            if goal is not None else None
        ),
        terminal_percentiles=TerminalPercentiles(
            p05=_percentile(values, 0.05),
            p25=_percentile(values, 0.25),
            p50=_percentile(values, 0.50),
            p75=_percentile(values, 0.75),
            p95=_percentile(values, 0.95),
        ),
    )


__all__ = ["MonteCarloResult", "TerminalPercentiles", "run_monte_carlo"]
