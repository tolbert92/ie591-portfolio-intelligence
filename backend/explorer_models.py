"""IE 591 Skill #6: data contract for the 3D Portfolio Explorer.

All three coordinates use decimals. Each point comes from one validated
Skill #5 efficient frontier point.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .visualization_models import (
    EfficientFrontier,
    FiniteNumber,
    Nonblank,
    Nonnegative,
    VisualizationProvenance,
    Weight,
)


class ExplorerPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annualized_volatility: Nonnegative
    expected_annual_return: FiniteNumber
    selected_asset_weight: Weight


class ExplorerDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = "1.0.0"
    portfolio_id: Nonblank
    selected_ticker: Nonblank
    provenance: VisualizationProvenance
    points: tuple[ExplorerPoint, ...] = Field(min_length=1)

    @classmethod
    def from_frontier(
        cls, frontier: EfficientFrontier, selected_ticker: str
    ) -> ExplorerDataset:
        """Project the frontier onto risk, return, and one asset's weight.

        Require an explicit weight at every point so missing data stays
        distinguishable from an allocation of zero.
        """
        selected_ticker = selected_ticker.strip()
        if not selected_ticker:
            raise ValueError("Select an asset ticker")
        points = []
        for index, point in enumerate(frontier.points, start=1):
            weights = {allocation.ticker: allocation.weight for allocation in point.allocations}
            if selected_ticker not in weights:
                raise ValueError(
                    f"Frontier point {index} has no allocation for {selected_ticker!r}"
                )
            points.append(
                ExplorerPoint(
                    annualized_volatility=point.annualized_volatility,
                    expected_annual_return=point.expected_annual_return,
                    selected_asset_weight=weights[selected_ticker],
                )
            )
        return cls(
            portfolio_id=frontier.portfolio_id,
            selected_ticker=selected_ticker,
            provenance=frontier.provenance,
            points=tuple(points),
        )


__all__ = ["ExplorerDataset", "ExplorerPoint"]
