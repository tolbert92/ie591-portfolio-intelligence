"""IE 591 Skill #9: request contract for comparing portfolio strategies.

Return and volatility are annualized decimals. Candidates in one request
must use the same comparison assumptions. A volatility limit is used for
eligibility before scoring in the comparison engine.
"""

from __future__ import annotations

from math import isclose
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .visualization_models import Nonblank, Nonnegative, PortfolioMetrics, Weight


class StrategyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: Nonblank
    label: Nonblank
    metrics: PortfolioMetrics
    source_analysis_id: Nonblank | None = None


class ComparisonCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_return_weight: Weight = 0.5
    volatility_weight: Weight = 0.5
    maximum_annualized_volatility: Nonnegative | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> ComparisonCriteria:
        if not isclose(
            self.expected_return_weight + self.volatility_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("Return and volatility scoring weights must sum to 1")
        return self


class StrategyComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    comparison_id: Nonblank
    assumptions_label: Nonblank
    is_synthetic: bool
    candidates: tuple[StrategyCandidate, ...] = Field(min_length=2)
    criteria: ComparisonCriteria = Field(default_factory=ComparisonCriteria)

    @model_validator(mode="after")
    def unique_strategies(self) -> StrategyComparisonRequest:
        if len({candidate.strategy_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Strategy IDs must be unique")
        return self


__all__ = ["ComparisonCriteria", "StrategyCandidate", "StrategyComparisonRequest"]
