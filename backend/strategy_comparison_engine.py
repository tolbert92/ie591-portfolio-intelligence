"""IE 591 Skill #9: score eligible portfolio strategies on common criteria."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .strategy_comparison_models import ComparisonCriteria, StrategyComparisonRequest
from .visualization_models import Nonblank, PortfolioMetrics, Weight


class RankedStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: Nonblank
    label: Nonblank
    metrics: PortfolioMetrics
    source_analysis_id: Nonblank | None = None
    eligible: bool
    exclusion_reason: str | None = None
    expected_return_score: Weight | None = None
    volatility_score: Weight | None = None
    weighted_score: Weight | None = None
    rank: int | None = Field(default=None, ge=1)


class StrategyComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    comparison_id: Nonblank
    assumptions_label: Nonblank
    is_synthetic: bool
    criteria: ComparisonCriteria
    status: Literal["ranked", "no_eligible_strategies"]
    strategies: tuple[RankedStrategy, ...] = Field(min_length=2)


def _scaled_score(value: float, low: float, high: float) -> float:
    if low == high:
        return 0.5
    if high - low == float("inf"):
        return (value / 2 - low / 2) / (high / 2 - low / 2)
    return (value - low) / (high - low)


def compare_strategies(request: StrategyComparisonRequest) -> StrategyComparisonResult:
    """Normalize metrics across eligible candidates and apply preference weights.

    A metric shared by all eligible strategies receives a neutral score
    of 0.5. Weighted scores rounded to 12 decimal places determine rank.
    Ties share a rank and are displayed in strategy ID order.
    """
    criteria = request.criteria
    limit = criteria.maximum_annualized_volatility
    eligible = []
    excluded = []
    for candidate in request.candidates:
        if limit is not None and candidate.metrics.annualized_volatility > limit:
            excluded.append(
                RankedStrategy(
                    **candidate.model_dump(),
                    eligible=False,
                    exclusion_reason="Annualized volatility exceeds the specified limit",
                )
            )
        else:
            eligible.append(candidate)

    ranked = []
    if eligible:
        returns = [candidate.metrics.expected_annual_return for candidate in eligible]
        volatilities = [candidate.metrics.annualized_volatility for candidate in eligible]
        lowest_return, highest_return = min(returns), max(returns)
        lowest_volatility, highest_volatility = min(volatilities), max(volatilities)
        total_weight = criteria.expected_return_weight + criteria.volatility_weight

        scored = []
        for candidate in eligible:
            return_score = _scaled_score(
                candidate.metrics.expected_annual_return, lowest_return, highest_return
            )
            volatility_score = 1 - _scaled_score(
                candidate.metrics.annualized_volatility, lowest_volatility, highest_volatility
            )
            weighted_score = round(
                (
                    criteria.expected_return_weight * return_score
                    + criteria.volatility_weight * volatility_score
                ) / total_weight,
                12,
            )
            scored.append((weighted_score, candidate, return_score, volatility_score))

        previous_score = None
        rank = 1
        for position, (score, candidate, return_score, volatility_score) in enumerate(
            sorted(scored, key=lambda row: (-row[0], row[1].strategy_id)), start=1
        ):
            if score != previous_score:
                rank = position
                previous_score = score
            ranked.append(
                RankedStrategy(
                    **candidate.model_dump(),
                    eligible=True,
                    expected_return_score=return_score,
                    volatility_score=volatility_score,
                    weighted_score=score,
                    rank=rank,
                )
            )

    return StrategyComparisonResult(
        comparison_id=request.comparison_id,
        assumptions_label=request.assumptions_label,
        is_synthetic=request.is_synthetic,
        criteria=criteria,
        status="ranked" if ranked else "no_eligible_strategies",
        strategies=tuple(ranked + excluded),
    )


__all__ = ["RankedStrategy", "StrategyComparisonResult", "compare_strategies"]
