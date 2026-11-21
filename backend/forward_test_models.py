"""IE 591 Skill #7: contracts for a buy-and-hold paper portfolio.

The starting allocations apply immediately before the first observation.
Each dated observation contains simple daily returns as decimals.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .visualization_models import FiniteNumber, Nonblank, PortfolioAllocation

SimpleReturn = Annotated[FiniteNumber, Field(ge=-1)]
PositiveAmount = Annotated[FiniteNumber, Field(gt=0)]


class _ForwardTestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetReturn(_ForwardTestModel):
    ticker: Nonblank
    daily_return: SimpleReturn


class ForwardObservation(_ForwardTestModel):
    date: date
    asset_returns: tuple[AssetReturn, ...] = Field(min_length=1)
    benchmark_return: SimpleReturn | None = None

    @model_validator(mode="after")
    def unique_tickers(self) -> ForwardObservation:
        if len({row.ticker for row in self.asset_returns}) != len(self.asset_returns):
            raise ValueError("Observation tickers must be unique")
        return self


class ReturnSeriesSource(_ForwardTestModel):
    source_name: Nonblank
    is_synthetic: bool
    dataset_id: Nonblank | None = None


class ForwardTestRequest(_ForwardTestModel):
    contract_version: Literal["1.0.0"] = "1.0.0"
    test_id: Nonblank
    portfolio: PortfolioAllocation
    initial_value: PositiveAmount
    returns_source: ReturnSeriesSource
    benchmark_id: Nonblank | None = None
    observations: tuple[ForwardObservation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observations(self) -> ForwardTestRequest:
        tickers = {allocation.ticker for allocation in self.portfolio.allocations}
        for observation in self.observations:
            if {row.ticker for row in observation.asset_returns} != tickers:
                raise ValueError(
                    f"Observation {observation.date} must cover exactly the portfolio tickers"
                )
            if (self.benchmark_id is not None) != (observation.benchmark_return is not None):
                raise ValueError(
                    "Benchmark returns are required on every observation when benchmark_id is set"
                )
        if any(
            first.date >= second.date
            for first, second in zip(self.observations, self.observations[1:])
        ):
            raise ValueError("Observation dates must be strictly increasing")
        return self


__all__ = ["AssetReturn", "ForwardObservation", "ReturnSeriesSource", "ForwardTestRequest"]
