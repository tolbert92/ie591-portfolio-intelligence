"""IE 591 Skill #10: summarize comparison and simulation evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .decision_support_models import DecisionSupportRequest, StrategySimulationEvidence
from .strategy_comparison_engine import StrategyComparisonResult
from .visualization_models import Nonblank

Outcome = Literal["leading_strategy", "tied_leaders", "sole_eligible", "no_eligible_strategies"]


class DecisionSupportResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    decision_id: Nonblank
    outcome: Outcome
    summary: Nonblank
    comparison: StrategyComparisonResult
    leading_strategy_ids: tuple[Nonblank, ...]
    simulations: tuple[StrategySimulationEvidence, ...]
    eligible_without_simulation: tuple[Nonblank, ...]
    limitations: tuple[Nonblank, ...]


def build_decision_support(request: DecisionSupportRequest) -> DecisionSupportResult:
    """Explain the supplied ranking and preserve the evidence behind it."""
    comparison = request.comparison
    eligible = tuple(row for row in comparison.strategies if row.eligible)
    leaders = tuple(row for row in eligible if row.rank == 1)
    if any(
        row.rank is None or row.weighted_score is None
        for row in eligible
    ) or (eligible and not leaders):
        raise ValueError("Eligible strategies must have scores and a first-ranked strategy")
    if any(
        row.rank is not None or row.weighted_score is not None
        for row in comparison.strategies if not row.eligible
    ):
        raise ValueError("Excluded strategies must have no rank or weighted score")

    outcome: Outcome
    if not eligible:
        outcome = "no_eligible_strategies"
        summary = "No strategy met the comparison eligibility criteria."
    elif len(eligible) == 1:
        outcome = "sole_eligible"
        summary = f"{eligible[0].label} was the only strategy that met the eligibility criteria."
    elif len(leaders) > 1:
        outcome = "tied_leaders"
        labels = ", ".join(row.label for row in leaders)
        summary = f"{labels} tied for first under the selected return and volatility weights."
    else:
        outcome = "leading_strategy"
        summary = f"{leaders[0].label} ranked first under the selected return and volatility weights."

    linked_ids = {row.strategy_id for row in request.simulations}
    missing = tuple(row.strategy_id for row in eligible if row.strategy_id not in linked_ids)
    limitations = [
        "Scores are relative to the eligible strategies and depend on the selected criteria."
    ]
    if comparison.is_synthetic:
        limitations.append("The strategy comparison uses synthetic assumptions.")
    if request.simulations:
        limitations.append("Simulation outcomes depend on the assumptions and horizon of each run.")
    if missing:
        limitations.append("Some eligible strategies have no linked simulation evidence.")

    return DecisionSupportResult(
        decision_id=request.decision_id,
        outcome=outcome,
        summary=summary,
        comparison=comparison,
        leading_strategy_ids=tuple(row.strategy_id for row in leaders),
        simulations=request.simulations,
        eligible_without_simulation=missing,
        limitations=tuple(limitations),
    )


__all__ = ["DecisionSupportResult", "build_decision_support"]
