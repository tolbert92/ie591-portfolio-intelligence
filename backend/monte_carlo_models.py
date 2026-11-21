"""IE 591 Skill #8: request contract for Monte Carlo portfolio simulation.

Annual return and volatility use decimals. The expected annual return is
the assumed mean one-year simple return. Volatility is the annualized
standard deviation of log returns for the geometric Brownian motion model.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .visualization_models import FiniteNumber, Nonblank, Nonnegative

PositiveAmount = Annotated[FiniteNumber, Field(gt=0)]
ExpectedAnnualReturn = Annotated[FiniteNumber, Field(gt=-1)]


class MonteCarloAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_annual_return: ExpectedAnnualReturn
    annualized_volatility: Nonnegative
    source_name: Nonblank
    is_synthetic: bool
    source_analysis_id: Nonblank | None = None


class MonteCarloRequest(BaseModel):
    """Simulate an initial balance without contributions or withdrawals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    simulation_id: Nonblank
    model: Literal["geometric_brownian_motion"] = "geometric_brownian_motion"
    initial_value: PositiveAmount
    horizon_years: int = Field(ge=1, le=50, strict=True)
    path_count: int = Field(default=1000, ge=100, le=10000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    assumptions: MonteCarloAssumptions
    goal_value: PositiveAmount | None = None


__all__ = ["MonteCarloAssumptions", "MonteCarloRequest"]
