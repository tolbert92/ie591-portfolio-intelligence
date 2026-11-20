"""IE 591 Skill #4 — versioned risk-assessment contracts (Stage 4.1).

Pydantic v2 models only. No risk calculations, external calls, or API routes.
All returns, weights, and shocks use decimals (0.05 means 5%).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "1.0.0"
WEIGHT_TOLERANCE = 1e-6
MATRIX_TOLERANCE = 1e-9


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RiskMethod(str, Enum):
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    AUTO = "auto"


class RiskHorizon(str, Enum):
    ONE_DAY = "1d"
    TEN_DAYS = "10d"
    THIRTY_DAYS = "30d"
    ONE_YEAR = "1y"


class RiskStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    INDETERMINATE = "indeterminate"


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    POTENTIAL_MISMATCH = "potential_mismatch"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Provenance(StrictModel):
    source_type: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    dataset_id: str | None = None
    upstream_analysis_id: str | None = None
    is_synthetic: bool


class RiskAsset(StrictModel):
    ticker: str = Field(min_length=1)
    asset_name: str | None = None
    asset_class: str = Field(default="unclassified", min_length=1)
    weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    expected_annual_return: float = Field(allow_inf_nan=False)


class HistoricalReturns(StrictModel):
    """Rows are ordered oldest-to-newest; each row follows assets order.

    Only daily, simple, non-overlapping returns are supported by this contract.
    Historical simulation sufficiency is determined by the calculation engine.
    """
    frequency: Literal["daily"] = "daily"
    observations: list[list[float]] = Field(min_length=2)
    observation_dates: list[str] | None = None

    @field_validator("observations")
    @classmethod
    def validate_observations(cls, rows: list[list[float]]) -> list[list[float]]:
        if any(not row or any(not isfinite(v) or v < -1 for v in row) for row in rows):
            raise ValueError("Historical returns must be finite, nonempty rows with returns >= -1")
        if len({len(row) for row in rows}) != 1:
            raise ValueError("Historical observations must have a consistent asset count")
        return rows

    @model_validator(mode="after")
    def validate_dates(self) -> HistoricalReturns:
        if self.observation_dates is not None:
            if len(self.observation_dates) != len(self.observations):
                raise ValueError("observation_dates must match observations length")
            parsed = []
            for value in self.observation_dates:
                try:
                    parsed.append(datetime.strptime(value, "%Y-%m-%d").date())
                except ValueError as exc:
                    raise ValueError("observation_dates must use YYYY-MM-DD") from exc
            if any(a >= b for a, b in zip(parsed, parsed[1:])):
                raise ValueError("observation_dates must be strictly increasing")
        return self


def _is_positive_semidefinite(matrix: list[list[float]], tol: float) -> bool:
    """Semidefinite Cholesky check, including zero-variance assets."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            residual = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if residual < -tol:
                    return False
                lower[i][j] = max(residual, 0.0) ** 0.5
            elif lower[j][j] > tol ** 0.5:
                lower[i][j] = residual / lower[j][j]
            elif abs(residual) > tol:
                return False
    return True


class RiskPortfolio(StrictModel):
    portfolio_id: str = Field(min_length=1)
    assets: list[RiskAsset] = Field(min_length=1)
    covariance_matrix: list[list[float]] = Field(min_length=1)
    historical_returns: HistoricalReturns | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def validate_portfolio(self) -> RiskPortfolio:
        n = len(self.assets)
        tickers = [asset.ticker for asset in self.assets]
        if len(set(tickers)) != n:
            raise ValueError("Asset tickers must be unique")
        if abs(sum(asset.weight for asset in self.assets) - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError("Portfolio weights must sum to 1 within tolerance")
        matrix = self.covariance_matrix
        if len(matrix) != n or any(len(row) != n for row in matrix):
            raise ValueError("Covariance matrix must be square and match the asset count")
        if any(not isfinite(value) for row in matrix for value in row):
            raise ValueError("Covariance matrix must contain finite numbers")
        if any(matrix[i][i] < 0 for i in range(n)):
            raise ValueError("Covariance matrix diagonal must be nonnegative")
        if any(abs(matrix[i][j] - matrix[j][i]) > MATRIX_TOLERANCE
               for i in range(n) for j in range(i)):
            raise ValueError("Covariance matrix must be symmetric")
        if not _is_positive_semidefinite(matrix, MATRIX_TOLERANCE):
            raise ValueError("Covariance matrix must be positive semidefinite")
        if self.historical_returns is not None:
            if any(len(row) != n for row in self.historical_returns.observations):
                raise ValueError("Historical observation width must match asset count")
        return self


class AssetShock(StrictModel):
    ticker: str = Field(min_length=1)
    shock_return: float = Field(ge=-1, allow_inf_nan=False)


class StressScenario(StrictModel):
    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scenario_type: Literal["predefined", "custom"]
    shocks: list[AssetShock] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tickers(self) -> StressScenario:
        tickers = [shock.ticker for shock in self.shocks]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Stress scenario shock tickers must be unique")
        return self


class RiskThresholds(StrictModel):
    """Illustrative, user-configurable thresholds, not investment standards."""
    volatility_moderate: float = Field(default=0.10, gt=0, allow_inf_nan=False)
    volatility_high: float = Field(default=0.20, gt=0, allow_inf_nan=False)
    volatility_very_high: float = Field(default=0.30, gt=0, allow_inf_nan=False)
    concentration_high: float = Field(default=0.50, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> RiskThresholds:
        if not (self.volatility_moderate < self.volatility_high < self.volatility_very_high):
            raise ValueError("Volatility thresholds must increase strictly")
        return self


class InvestorRiskProfile(StrictModel):
    """Optional normalized subset of Skill #1; never invent missing fields."""
    profile_id: str | None = None
    risk_tolerance: str | None = None
    investment_horizon_years: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    financial_objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RiskConfig(StrictModel):
    method: RiskMethod = RiskMethod.AUTO
    horizon: RiskHorizon = RiskHorizon.ONE_DAY
    confidence_level: Literal[0.90, 0.95, 0.975, 0.99] = 0.95
    trading_days_per_year: int = Field(default=252, ge=200, le=366)
    thresholds: RiskThresholds = Field(default_factory=RiskThresholds)


class RiskAssessmentRequest(StrictModel):
    assessment_id: str = Field(min_length=1)
    portfolio: RiskPortfolio
    config: RiskConfig = Field(default_factory=RiskConfig)
    investor_profile: InvestorRiskProfile | None = None
    stress_scenarios: list[StressScenario] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scenarios(self) -> RiskAssessmentRequest:
        valid = {asset.ticker for asset in self.portfolio.assets}
        ids = [scenario.scenario_id for scenario in self.stress_scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Stress scenario IDs must be unique")
        for scenario in self.stress_scenarios:
            unknown = {shock.ticker for shock in scenario.shocks} - valid
            if unknown:
                raise ValueError(f"Unknown scenario asset tickers: {sorted(unknown)}")
        return self


class ValidationIssue(StrictModel):
    severity: IssueSeverity
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    field_path: str | None = None


class ValidationReport(StrictModel):
    status: Literal["pass", "warning", "fail"]
    issues: list[ValidationIssue] = Field(default_factory=list)
    disabled_calculations: list[str] = Field(default_factory=list)


class RiskMeasure(StrictModel):
    value: float = Field(allow_inf_nan=False)
    unit: Literal["decimal", "currency"] = "decimal"
    method: RiskMethod
    horizon: RiskHorizon
    confidence_level: float | None = None
    assumptions: list[str] = Field(default_factory=list)


class AssetRiskContribution(StrictModel):
    ticker: str
    weight: float = Field(allow_inf_nan=False)
    marginal_volatility: float = Field(allow_inf_nan=False)
    component_volatility: float = Field(allow_inf_nan=False)
    percent_total_risk: float = Field(allow_inf_nan=False)


class StressResult(StrictModel):
    scenario_id: str
    portfolio_shock_return: float = Field(allow_inf_nan=False)
    hypothetical: Literal[True] = True
    contributions: dict[str, float] = Field(default_factory=dict)


class RiskClassification(StrictModel):
    overall: RiskLevel
    dimensions: dict[str, RiskLevel] = Field(default_factory=dict)
    explanations: list[str] = Field(default_factory=list)
    thresholds: RiskThresholds


class InvestorAlignment(StrictModel):
    status: AlignmentStatus
    explanations: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    suitability_determination: Literal[False] = False


class RiskAssessmentResult(StrictModel):
    assessment_id: str
    status: RiskStatus
    portfolio_id: str
    method_requested: RiskMethod
    method_used: RiskMethod | None = None
    method_fallback_reason: str | None = None
    horizon: RiskHorizon
    confidence_level: float
    annualized_volatility: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    value_at_risk: RiskMeasure | None = None
    conditional_value_at_risk: RiskMeasure | None = None
    asset_risk_contributions: list[AssetRiskContribution] = Field(default_factory=list)
    stress_results: list[StressResult] = Field(default_factory=list)
    classification: RiskClassification | None = None
    investor_alignment: InvestorAlignment | None = None
    validation: ValidationReport
    provenance: Provenance
    calculation_version: str = CONTRACT_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    downstream_ready: bool = False


class RiskExportPackage(StrictModel):
    project: Literal["IE 591 Portfolio Intelligence"] = "IE 591 Portfolio Intelligence"
    skill: Literal["Risk Assessment"] = "Risk Assessment"
    contract_version: str = CONTRACT_VERSION
    synthetic: bool
    input: RiskAssessmentRequest
    assessment: RiskAssessmentResult

    @model_validator(mode="after")
    def consistent_identity(self) -> RiskExportPackage:
        if self.synthetic != self.input.portfolio.provenance.is_synthetic:
            raise ValueError("Export synthetic flag must match portfolio provenance")
        if self.input.assessment_id != self.assessment.assessment_id:
            raise ValueError("Export assessment IDs must match")
        if self.input.portfolio.portfolio_id != self.assessment.portfolio_id:
            raise ValueError("Export portfolio IDs must match")
        return self


class OptimizationExportAdapter:
    """Validate and normalize the Skill #3 JSON export without recalculating.

    This is an explicit integration boundary; frontend auto-handoff comes later.
    """

    @staticmethod
    def from_export(data: dict[str, Any], *, assessment_id: str = "IE591-RISK-DEMO") -> RiskAssessmentRequest:
        if data.get("project") != "IE 591 Portfolio Intelligence" or data.get("skill") != "Portfolio Optimization":
            raise ValueError("Expected an IE 591 Portfolio Optimization export")
        source = data.get("input")
        result = data.get("optimization")
        if not isinstance(source, dict) or not isinstance(result, dict):
            raise ValueError("Optimization export requires input and optimization objects")
        if result.get("status") != "completed" or result.get("downstream_ready") is not True:
            raise ValueError("Optimization must be completed and downstream-ready")
        source_assets = source.get("assets")
        allocations = result.get("allocations")
        if not isinstance(source_assets, list) or not isinstance(allocations, list):
            raise ValueError("Optimization export requires assets and allocations arrays")
        allocation_map: dict[str, float] = {}
        for allocation in allocations:
            if not isinstance(allocation, dict) or not isinstance(allocation.get("ticker"), str):
                raise ValueError("Each allocation requires a ticker")
            ticker = allocation["ticker"]
            if ticker in allocation_map:
                raise ValueError("Duplicate allocation ticker")
            allocation_map[ticker] = allocation.get("weight")
        if {asset.get("ticker") for asset in source_assets} != set(allocation_map):
            raise ValueError("Optimization assets and allocations must match exactly")
        assets = [RiskAsset(
            ticker=asset["ticker"],
            asset_name=asset.get("asset_name"),
            asset_class=asset.get("asset_class") or "unclassified",
            weight=allocation_map[asset["ticker"]],
            expected_annual_return=asset["expected_annual_return"],
        ) for asset in source_assets]
        provenance = Provenance.model_validate(source.get("provenance"))
        portfolio = RiskPortfolio(
            portfolio_id=source.get("optimization_id") or result.get("optimization_id"),
            assets=assets,
            covariance_matrix=source.get("covariance_matrix"),
            provenance=provenance,
        )
        return RiskAssessmentRequest(assessment_id=assessment_id, portfolio=portfolio)


__all__ = [
    "CONTRACT_VERSION", "RiskMethod", "RiskHorizon", "RiskStatus", "RiskLevel",
    "AlignmentStatus", "IssueSeverity", "Provenance", "RiskAsset", "HistoricalReturns",
    "RiskPortfolio", "AssetShock", "StressScenario", "RiskThresholds",
    "InvestorRiskProfile", "RiskConfig", "RiskAssessmentRequest", "ValidationIssue",
    "ValidationReport", "RiskMeasure", "AssetRiskContribution", "StressResult",
    "RiskClassification", "InvestorAlignment", "RiskAssessmentResult",
    "RiskExportPackage", "OptimizationExportAdapter",
]
