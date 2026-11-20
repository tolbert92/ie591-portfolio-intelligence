"""IE 591 Skill #5, Stage 5.1: portfolio visualization data contracts.

Returns, volatilities, weights, and shares of total risk use decimals.
Portfolio performance and frontier metrics are annualized. Stress returns
describe hypothetical scenarios. Risk measures carry their own horizon and unit.
"""

from __future__ import annotations

from enum import Enum
from math import fsum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONTRACT_VERSION = "1.0.0"
# Accept the wider of the two upstream weight tolerances.
WEIGHT_TOLERANCE = 1e-6

Nonblank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[FiniteNumber, Field(ge=0)]
Weight = Annotated[FiniteNumber, Field(ge=0, le=1)]
StressReturn = Annotated[FiniteNumber, Field(ge=-1)]
ConfidenceLevel = Annotated[FiniteNumber, Field(gt=0, lt=1)]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VisualizationDiagnostic(_ContractModel):
    severity: DiagnosticSeverity
    code: Nonblank
    message: Nonblank
    field_path: Nonblank | None = None


class VisualizationProvenance(_ContractModel):
    source_skill: Literal["Portfolio Optimization", "Risk Assessment"]
    source_type: Nonblank
    source_name: Nonblank
    analysis_id: Nonblank
    calculation_version: Nonblank
    is_synthetic: bool
    dataset_id: Nonblank | None = None
    upstream_analysis_id: Nonblank | None = None
    generated_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_synthetic_source(self) -> VisualizationProvenance:
        if self.source_type == "synthetic" and not self.is_synthetic:
            raise ValueError("Synthetic source_type requires is_synthetic=true")
        return self


class AssetAllocation(_ContractModel):
    ticker: Nonblank
    weight: Weight
    asset_name: Nonblank | None = None
    asset_class: Nonblank | None = None


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _validate_allocations(allocations: tuple[AssetAllocation, ...]) -> None:
    _require_unique(tuple(row.ticker for row in allocations), "Allocation tickers")
    total = fsum(row.weight for row in allocations)
    if not 1.0 - WEIGHT_TOLERANCE <= total <= 1.0 + WEIGHT_TOLERANCE:
        raise ValueError("Allocation weights must sum to 1 within 1e-6")


class PortfolioMetrics(_ContractModel):
    expected_annual_return: FiniteNumber
    annualized_volatility: Nonnegative
    annualized_variance: Nonnegative | None = None
    sharpe_ratio: FiniteNumber | None = None


class PortfolioAllocation(_ContractModel):
    portfolio_id: Nonblank
    label: Nonblank | None = None
    allocations: tuple[AssetAllocation, ...] = Field(min_length=1)
    portfolio_metrics: PortfolioMetrics | None = None
    provenance: VisualizationProvenance

    @model_validator(mode="after")
    def validate_weights(self) -> PortfolioAllocation:
        _validate_allocations(self.allocations)
        return self


class EfficientFrontierPoint(_ContractModel):
    target_annual_return: FiniteNumber
    expected_annual_return: FiniteNumber
    annualized_volatility: Nonnegative
    allocations: tuple[AssetAllocation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_weights(self) -> EfficientFrontierPoint:
        _validate_allocations(self.allocations)
        return self


class EfficientFrontier(_ContractModel):
    portfolio_id: Nonblank
    points: tuple[EfficientFrontierPoint, ...] = Field(min_length=1)
    provenance: VisualizationProvenance


class RiskMeasure(_ContractModel):
    value: FiniteNumber
    unit: Literal["decimal", "currency"]
    method: Literal["parametric", "historical", "auto"]
    horizon: Literal["1d", "10d", "30d", "1y"]
    confidence_level: ConfidenceLevel | None = None


class AssetRiskContribution(_ContractModel):
    """Contributions can be negative or exceed the total for individual assets."""

    ticker: Nonblank
    weight: Weight
    marginal_volatility: FiniteNumber
    component_volatility: FiniteNumber
    percent_total_risk: FiniteNumber


class StressContribution(_ContractModel):
    ticker: Nonblank
    contribution: FiniteNumber


class StressScenarioResult(_ContractModel):
    scenario_id: Nonblank
    name: Nonblank | None = None
    scenario_type: Literal["predefined", "custom"] | None = None
    portfolio_shock_return: StressReturn
    hypothetical: Literal[True] = True
    contributions: tuple[StressContribution, ...] = ()

    @model_validator(mode="after")
    def validate_tickers(self) -> StressScenarioResult:
        _require_unique(tuple(row.ticker for row in self.contributions), "Stress contribution tickers")
        return self


class PortfolioRisk(_ContractModel):
    portfolio_id: Nonblank
    status: Literal["completed", "partial", "blocked"]
    annualized_volatility: Nonnegative | None = None
    value_at_risk: RiskMeasure | None = None
    conditional_value_at_risk: RiskMeasure | None = None
    asset_risk_contributions: tuple[AssetRiskContribution, ...] = ()
    stress_results: tuple[StressScenarioResult, ...] = ()
    provenance: VisualizationProvenance
    diagnostics: tuple[VisualizationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def validate_identifiers(self) -> PortfolioRisk:
        _require_unique(tuple(row.ticker for row in self.asset_risk_contributions), "Risk contribution tickers")
        _require_unique(tuple(row.scenario_id for row in self.stress_results), "Stress scenario IDs")
        return self


class PortfolioComparison(_ContractModel):
    comparison_id: Nonblank
    label: Nonblank | None = None
    portfolio_ids: tuple[Nonblank, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_portfolio_ids(self) -> PortfolioComparison:
        _require_unique(self.portfolio_ids, "Comparison portfolio IDs")
        return self


class VisualizationDataset(_ContractModel):
    """Versioned payload for charts and comparisons.

    Frontier, risk, and stress series may omit assets with no reported value.
    Each supplied ticker must resolve to an allocation in the referenced portfolio.
    """

    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    portfolios: tuple[PortfolioAllocation, ...] = Field(min_length=1)
    efficient_frontiers: tuple[EfficientFrontier, ...] = ()
    risk_results: tuple[PortfolioRisk, ...] = ()
    comparisons: tuple[PortfolioComparison, ...] = ()
    diagnostics: tuple[VisualizationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> VisualizationDataset:
        _require_unique(tuple(row.portfolio_id for row in self.portfolios), "Portfolio IDs")
        _require_unique(tuple(row.comparison_id for row in self.comparisons), "Comparison IDs")
        weights_by_portfolio = {
            row.portfolio_id: {allocation.ticker: allocation.weight for allocation in row.allocations}
            for row in self.portfolios
        }

        for frontier in self.efficient_frontiers:
            weights = _portfolio_weights(weights_by_portfolio, frontier.portfolio_id)
            for point in frontier.points:
                _validate_tickers({row.ticker for row in point.allocations}, weights, frontier.portfolio_id)

        for result in self.risk_results:
            weights = _portfolio_weights(weights_by_portfolio, result.portfolio_id)
            _validate_tickers(
                {row.ticker for row in result.asset_risk_contributions}, weights, result.portfolio_id
            )
            for row in result.asset_risk_contributions:
                if abs(row.weight - weights[row.ticker]) > WEIGHT_TOLERANCE:
                    raise ValueError(f"Risk contribution weight does not match allocation for {row.ticker!r}")
            for scenario in result.stress_results:
                _validate_tickers({row.ticker for row in scenario.contributions}, weights, result.portfolio_id)

        for comparison in self.comparisons:
            for portfolio_id in comparison.portfolio_ids:
                _portfolio_weights(weights_by_portfolio, portfolio_id)

        return self


def _portfolio_weights(
    weights_by_portfolio: dict[str, dict[str, float]], portfolio_id: str
) -> dict[str, float]:
    if portfolio_id not in weights_by_portfolio:
        raise ValueError(f"Unknown portfolio ID: {portfolio_id!r}")
    return weights_by_portfolio[portfolio_id]


def _validate_tickers(tickers: set[str], weights: dict[str, float], portfolio_id: str) -> None:
    unknown = tickers.difference(weights)
    if unknown:
        raise ValueError(f"Unknown tickers for portfolio {portfolio_id!r}: {sorted(unknown)}")


__all__ = [
    "CONTRACT_VERSION",
    "WEIGHT_TOLERANCE",
    "DiagnosticSeverity",
    "VisualizationDiagnostic",
    "VisualizationProvenance",
    "AssetAllocation",
    "PortfolioMetrics",
    "PortfolioAllocation",
    "EfficientFrontierPoint",
    "EfficientFrontier",
    "RiskMeasure",
    "AssetRiskContribution",
    "StressContribution",
    "StressScenarioResult",
    "PortfolioRisk",
    "PortfolioComparison",
    "VisualizationDataset",
]
