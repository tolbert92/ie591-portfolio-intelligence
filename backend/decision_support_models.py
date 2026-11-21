"""IE 591 Skill #10: inputs for evidence-based decision support.

Simulation evidence is explicitly associated with a strategy by the caller.
The contract preserves the underlying results and their source assumptions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .monte_carlo_engine import MonteCarloResult
from .strategy_comparison_engine import StrategyComparisonResult
from .visualization_models import Nonblank


class StrategySimulationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: Nonblank
    simulation: MonteCarloResult


class DecisionSupportRequest(BaseModel):
    """Provide a completed comparison and optional simulation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    decision_id: Nonblank
    comparison: StrategyComparisonResult
    simulations: tuple[StrategySimulationEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_simulation_references(self) -> DecisionSupportRequest:
        candidate_ids = {strategy.strategy_id for strategy in self.comparison.strategies}
        linked_ids = {evidence.strategy_id for evidence in self.simulations}
        unknown = linked_ids - candidate_ids
        if unknown:
            raise ValueError(f"Simulation evidence refers to unknown strategies: {sorted(unknown)}")
        if len(linked_ids) != len(self.simulations):
            raise ValueError("Provide at most one simulation per strategy")
        simulation_ids = {evidence.simulation.simulation_id for evidence in self.simulations}
        if len(simulation_ids) != len(self.simulations):
            raise ValueError("Simulation IDs must be unique within a decision")
        return self


__all__ = ["DecisionSupportRequest", "StrategySimulationEvidence"]
