
"""IE 591 Skill #5 — Tiered portfolio visualization validation.

Validates visualization datasets without changing source calculations.

Validation policy:
- ERROR: Block the affected dataset or portfolio.
- WARNING: Disable only the affected optional visualization.
- INFO: Record nonblocking data limitations.

All portfolio weights and returns use decimal units.
This module does not recalculate optimization or risk results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ValidationError

from .visualization_models import VisualizationDataset


VALIDATION_VERSION = "1.0.0"
WEIGHT_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-8

Severity = Literal["error", "warning", "info"]
ValidationStatus = Literal["pass", "warning", "fail"]

CHART_ALLOCATION = "allocation"
CHART_OVERVIEW = "overview"
CHART_FRONTIER = "efficient_frontier"
CHART_RISK = "risk_attribution"
CHART_STRESS = "stress_scenarios"
CHART_COMPARISON = "portfolio_comparison"

ALL_CHARTS = (
    CHART_ALLOCATION,
    CHART_OVERVIEW,
    CHART_FRONTIER,
    CHART_RISK,
    CHART_STRESS,
    CHART_COMPARISON,
)


@dataclass(frozen=True)
class ValidationFinding:
    """One machine-readable validation finding."""

    severity: Severity
    code: str
    message: str
    field_path: str | None = None
    portfolio_id: str | None = None
    chart: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "field_path": self.field_path,
            "portfolio_id": self.portfolio_id,
            "chart": self.chart,
        }


@dataclass
class VisualizationValidationReport:
    """Aggregate validation results and chart availability."""

    version: str = VALIDATION_VERSION
    status: ValidationStatus = "pass"
    findings: list[ValidationFinding] = field(default_factory=list)
    disabled_charts: dict[str, list[str]] = field(default_factory=dict)
    blocked_portfolios: list[str] = field(default_factory=list)
    validated_portfolios: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status != "fail"

    @property
    def has_errors(self) -> bool:
        return any(
            finding.severity == "error"
            for finding in self.findings
        )

    @property
    def has_warnings(self) -> bool:
        return any(
            finding.severity == "warning"
            for finding in self.findings
        )

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        field_path: str | None = None,
        portfolio_id: str | None = None,
        chart: str | None = None,
    ) -> None:
        self.findings.append(
            ValidationFinding(
                severity=severity,
                code=code,
                message=message,
                field_path=field_path,
                portfolio_id=portfolio_id,
                chart=chart,
            )
        )

        if severity == "error":
            self.status = "fail"
        elif severity == "warning" and self.status == "pass":
            self.status = "warning"

    def disable_chart(
        self,
        chart: str,
        reason: str,
        *,
        portfolio_id: str | None = None,
    ) -> None:
        key = (
            f"{portfolio_id}:{chart}"
            if portfolio_id
            else chart
        )

        reasons = self.disabled_charts.setdefault(key, [])

        if reason not in reasons:
            reasons.append(reason)

    def block_portfolio(
        self,
        portfolio_id: str,
        reason: str,
        *,
        code: str = "PORTFOLIO_BLOCKED",
        field_path: str | None = None,
    ) -> None:
        if portfolio_id not in self.blocked_portfolios:
            self.blocked_portfolios.append(portfolio_id)

        self.add(
            "error",
            code,
            reason,
            field_path=field_path,
            portfolio_id=portfolio_id,
        )

        for chart in ALL_CHARTS:
            self.disable_chart(
                chart,
                reason,
                portfolio_id=portfolio_id,
            )

    def chart_available(
        self,
        chart: str,
        *,
        portfolio_id: str | None = None,
    ) -> bool:
        if portfolio_id in self.blocked_portfolios:
            return False

        if chart in self.disabled_charts:
            return False

        if portfolio_id:
            key = f"{portfolio_id}:{chart}"
            if key in self.disabled_charts:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "valid": self.valid,
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "disabled_charts": self.disabled_charts,
            "blocked_portfolios": self.blocked_portfolios,
            "validated_portfolios": self.validated_portfolios,
        }


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert a Pydantic model or mapping into a dictionary."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(
        "Expected a Pydantic model or mapping"
    )


def _rows(value: Any) -> list[Any]:
    """Normalize optional sequences without treating strings as rows."""

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return []


def _finite_number(value: Any) -> bool:
    """Accept finite real numbers but reject booleans."""

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return isfinite(value)


def _nonempty_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
    )


def _validate_provenance(
    provenance: Any,
    report: VisualizationValidationReport,
    *,
    portfolio_id: str,
    path: str,
) -> None:
    """Check the minimum metadata needed for traceability."""

    if provenance is None:
        report.add(
            "warning",
            "MISSING_PROVENANCE",
            "Source provenance is unavailable.",
            field_path=path,
            portfolio_id=portfolio_id,
        )
        return

    data = _as_dict(provenance)

    for name in (
        "source_type",
        "source_name",
        "source_skill",
        "analysis_id",
    ):
        if not _nonempty_identifier(data.get(name)):
            report.add(
                "warning",
                "INCOMPLETE_PROVENANCE",
                f"Provenance field '{name}' is missing.",
                field_path=f"{path}.{name}",
                portfolio_id=portfolio_id,
            )

    if not isinstance(data.get("is_synthetic"), bool):
        report.add(
            "warning",
            "UNKNOWN_SYNTHETIC_STATUS",
            "Synthetic-data status cannot be confirmed.",
            field_path=f"{path}.is_synthetic",
            portfolio_id=portfolio_id,
        )


def _validate_allocations(
    allocations: Any,
    report: VisualizationValidationReport,
    *,
    portfolio_id: str,
    path: str,
    require_fully_invested: bool = True,
) -> bool:
    """Validate tickers, weights, uniqueness and investment total."""

    rows = _rows(allocations)

    if not rows:
        report.block_portfolio(
            portfolio_id,
            "Portfolio has no asset allocations.",
            code="EMPTY_ALLOCATION",
            field_path=path,
        )
        return False

    tickers: set[str] = set()
    total_weight = 0.0
    valid = True

    for index, item in enumerate(rows):
        asset = _as_dict(item)
        ticker = asset.get("ticker")
        weight = asset.get("weight")
        item_path = f"{path}[{index}]"

        if not _nonempty_identifier(ticker):
            report.block_portfolio(
                portfolio_id,
                "Every allocation must have a nonempty ticker.",
                code="INVALID_TICKER",
                field_path=f"{item_path}.ticker",
            )
            valid = False
        elif ticker in tickers:
            report.block_portfolio(
                portfolio_id,
                f"Duplicate allocation ticker: {ticker}",
                code="DUPLICATE_TICKER",
                field_path=f"{item_path}.ticker",
            )
            valid = False
        else:
            tickers.add(ticker)

        if (
            not _finite_number(weight)
            or weight < 0
            or weight > 1
        ):
            report.block_portfolio(
                portfolio_id,
                "Allocation weight must be finite and between 0 and 1.",
                code="INVALID_WEIGHT",
                field_path=f"{item_path}.weight",
            )
            valid = False
        else:
            total_weight += float(weight)

    if (
        require_fully_invested
        and abs(total_weight - 1.0) > WEIGHT_TOLERANCE
    ):
        report.block_portfolio(
            portfolio_id,
            (
                "Portfolio weights must sum to 1.0; "
                f"received {total_weight:.10f}."
            ),
            code="WEIGHTS_NOT_NORMALIZED",
            field_path=path,
        )
        valid = False

    return valid


def _validate_metrics(
    metrics: Any,
    report: VisualizationValidationReport,
    *,
    portfolio_id: str,
    path: str,
) -> None:
    """Check optional portfolio metrics without blocking allocation charts."""

    if metrics is None:
        report.add(
            "info",
            "METRICS_UNAVAILABLE",
            "Portfolio metrics are not available.",
            field_path=path,
            portfolio_id=portfolio_id,
        )
        report.disable_chart(
            CHART_OVERVIEW,
            "Portfolio metrics are unavailable.",
            portfolio_id=portfolio_id,
        )
        return

    data = _as_dict(metrics)

    for name in (
        "expected_annual_return",
        "annualized_volatility",
        "annualized_variance",
        "sharpe_ratio",
    ):
        value = data.get(name)

        if value is None:
            continue

        if not _finite_number(value):
            report.add(
                "warning",
                "INVALID_PORTFOLIO_METRIC",
                f"Metric '{name}' is not finite.",
                field_path=f"{path}.{name}",
                portfolio_id=portfolio_id,
                chart=CHART_OVERVIEW,
            )
            report.disable_chart(
                CHART_OVERVIEW,
                f"Invalid metric: {name}",
                portfolio_id=portfolio_id,
            )
            continue

        if (
            name in (
                "annualized_volatility",
                "annualized_variance",
            )
            and value < 0
        ):
            report.add(
                "warning",
                "NEGATIVE_RISK_METRIC",
                f"Metric '{name}' cannot be negative.",
                field_path=f"{path}.{name}",
                portfolio_id=portfolio_id,
                chart=CHART_OVERVIEW,
            )
            report.disable_chart(
                CHART_OVERVIEW,
                f"Negative metric: {name}",
                portfolio_id=portfolio_id,
            )

    variance = data.get("annualized_variance")
    volatility = data.get("annualized_volatility")

    if (
        _finite_number(variance)
        and _finite_number(volatility)
        and variance >= 0
        and volatility >= 0
    ):
        difference = abs(
            float(variance) - float(volatility) ** 2
        )

        if difference > METRIC_TOLERANCE:
            report.add(
                "warning",
                "METRIC_INCONSISTENCY",
                "Annualized variance and volatility are inconsistent.",
                field_path=path,
                portfolio_id=portfolio_id,
                chart=CHART_OVERVIEW,
            )


def _validate_frontier(
    frontier: Any,
    report: VisualizationValidationReport,
    *,
    portfolio_ids: set[str],
    path: str,
) -> None:
    """Validate efficient-frontier plotting requirements."""

    data = _as_dict(frontier)
    portfolio_id = data.get("portfolio_id")

    if portfolio_id not in portfolio_ids:
        report.add(
            "error",
            "UNKNOWN_FRONTIER_PORTFOLIO",
            "Efficient frontier references an unknown portfolio.",
            field_path=f"{path}.portfolio_id",
            chart=CHART_FRONTIER,
        )
        report.disable_chart(
            CHART_FRONTIER,
            "Frontier references an unknown portfolio.",
        )
        return

    points = _rows(data.get("points"))

    if not points:
        report.add(
            "warning",
            "EMPTY_FRONTIER",
            "No efficient-frontier points are available.",
            field_path=f"{path}.points",
            portfolio_id=portfolio_id,
            chart=CHART_FRONTIER,
        )
        report.disable_chart(
            CHART_FRONTIER,
            "Efficient frontier has no points.",
            portfolio_id=portfolio_id,
        )
        return

    for index, item in enumerate(points):
        point = _as_dict(item)
        point_path = f"{path}.points[{index}]"

        for name in (
            "target_annual_return",
            "expected_annual_return",
            "annualized_volatility",
        ):
            value = point.get(name)

            if not _finite_number(value):
                report.add(
                    "warning",
                    "INVALID_FRONTIER_POINT",
                    f"Frontier field '{name}' must be finite.",
                    field_path=f"{point_path}.{name}",
                    portfolio_id=portfolio_id,
                    chart=CHART_FRONTIER,
                )
                report.disable_chart(
                    CHART_FRONTIER,
                    "Frontier contains invalid numerical data.",
                    portfolio_id=portfolio_id,
                )

        volatility = point.get("annualized_volatility")

        if _finite_number(volatility) and volatility < 0:
            report.add(
                "warning",
                "NEGATIVE_FRONTIER_VOLATILITY",
                "Frontier volatility cannot be negative.",
                field_path=f"{point_path}.annualized_volatility",
                portfolio_id=portfolio_id,
                chart=CHART_FRONTIER,
            )
            report.disable_chart(
                CHART_FRONTIER,
                "Frontier contains negative volatility.",
                portfolio_id=portfolio_id,
            )

        point_allocations = _rows(point.get("allocations"))

        if point_allocations:
            weights = []
            tickers = []

            for allocation in point_allocations:
                asset = _as_dict(allocation)
                weights.append(asset.get("weight"))
                tickers.append(asset.get("ticker"))

            valid_weights = all(
                _finite_number(weight)
                and 0 <= weight <= 1
                for weight in weights
            )

            valid_tickers = (
                all(_nonempty_identifier(t) for t in tickers)
                and len(set(tickers)) == len(tickers)
            )

            if (
                not valid_weights
                or not valid_tickers
                or abs(sum(weights) - 1.0) > WEIGHT_TOLERANCE
            ):
                report.add(
                    "warning",
                    "INVALID_FRONTIER_ALLOCATIONS",
                    "Frontier point allocations are invalid.",
                    field_path=f"{point_path}.allocations",
                    portfolio_id=portfolio_id,
                    chart=CHART_FRONTIER,
                )
                report.disable_chart(
                    CHART_FRONTIER,
                    "Invalid frontier allocations.",
                    portfolio_id=portfolio_id,
                )


def _validate_risk(
    risk: Any,
    report: VisualizationValidationReport,
    *,
    portfolio_ids: set[str],
    allocation_tickers: dict[str, set[str]],
    path: str,
) -> None:
    """Validate optional risk attribution and stress data."""

    data = _as_dict(risk)
    portfolio_id = data.get("portfolio_id")

    if portfolio_id not in portfolio_ids:
        report.add(
            "error",
            "UNKNOWN_RISK_PORTFOLIO",
            "Risk result references an unknown portfolio.",
            field_path=f"{path}.portfolio_id",
            chart=CHART_RISK,
        )
        report.disable_chart(
            CHART_RISK,
            "Risk result references an unknown portfolio.",
        )
        report.disable_chart(
            CHART_STRESS,
            "Risk result references an unknown portfolio.",
        )
        return

    status = data.get("status")

    if status == "blocked":
        report.add(
            "warning",
            "UPSTREAM_RISK_BLOCKED",
            "Upstream risk calculations were blocked.",
            field_path=f"{path}.status",
            portfolio_id=portfolio_id,
            chart=CHART_RISK,
        )
        report.disable_chart(
            CHART_RISK,
            "Upstream risk calculations were blocked.",
            portfolio_id=portfolio_id,
        )
    elif status == "partial":
        report.add(
            "warning",
            "UPSTREAM_RISK_PARTIAL",
            "Upstream risk calculations are partially available.",
            field_path=f"{path}.status",
            portfolio_id=portfolio_id,
        )
    elif status != "completed":
        report.add(
            "warning",
            "UNKNOWN_RISK_STATUS",
            "Risk assessment status is not recognized.",
            field_path=f"{path}.status",
            portfolio_id=portfolio_id,
        )
        report.disable_chart(
            CHART_RISK,
            "Unknown risk assessment status.",
            portfolio_id=portfolio_id,
        )

    volatility = data.get("annualized_volatility")

    if volatility is not None and (
        not _finite_number(volatility)
        or volatility < 0
    ):
        report.add(
            "warning",
            "INVALID_RISK_VOLATILITY",
            "Risk volatility must be finite and nonnegative.",
            field_path=f"{path}.annualized_volatility",
            portfolio_id=portfolio_id,
            chart=CHART_RISK,
        )
        report.disable_chart(
            CHART_RISK,
            "Invalid risk volatility.",
            portfolio_id=portfolio_id,
        )

    for name in (
        "value_at_risk",
        "conditional_value_at_risk",
    ):
        measure = data.get(name)

        if measure is None:
            continue

        value = _as_dict(measure).get("value")

        if not _finite_number(value):
            report.add(
                "warning",
                "INVALID_RISK_MEASURE",
                f"Risk measure '{name}' is not finite.",
                field_path=f"{path}.{name}.value",
                portfolio_id=portfolio_id,
                chart=CHART_RISK,
            )
            report.disable_chart(
                CHART_RISK,
                "Invalid risk measure.",
                portfolio_id=portfolio_id,
            )

    contributions = _rows(
        data.get("asset_risk_contributions")
    )

    if not contributions:
        report.add(
            "info",
            "RISK_ATTRIBUTION_UNAVAILABLE",
            "Asset risk contributions are unavailable.",
            portfolio_id=portfolio_id,
            chart=CHART_RISK,
        )
        report.disable_chart(
            CHART_RISK,
            "No asset risk contributions are available.",
            portfolio_id=portfolio_id,
        )

    seen_tickers: set[str] = set()

    for index, item in enumerate(contributions):
        contribution = _as_dict(item)
        ticker = contribution.get("ticker")
        contribution_path = (
            f"{path}.asset_risk_contributions[{index}]"
        )

        if (
            not _nonempty_identifier(ticker)
            or ticker in seen_tickers
            or ticker not in allocation_tickers[portfolio_id]
        ):
            report.add(
                "warning",
                "INVALID_RISK_CONTRIBUTION_TICKER",
                "Risk contribution ticker is missing, duplicated, or unknown.",
                field_path=f"{contribution_path}.ticker",
                portfolio_id=portfolio_id,
                chart=CHART_RISK,
            )
            report.disable_chart(
                CHART_RISK,
                "Risk attribution tickers are inconsistent.",
                portfolio_id=portfolio_id,
            )

        if _nonempty_identifier(ticker):
            seen_tickers.add(ticker)

        for name in (
            "weight",
            "marginal_volatility",
            "component_volatility",
            "percent_total_risk",
        ):
            value = contribution.get(name)

            if value is not None and not _finite_number(value):
                report.add(
                    "warning",
                    "INVALID_RISK_CONTRIBUTION",
                    f"Risk contribution '{name}' is not finite.",
                    field_path=f"{contribution_path}.{name}",
                    portfolio_id=portfolio_id,
                    chart=CHART_RISK,
                )
                report.disable_chart(
                    CHART_RISK,
                    "Invalid risk contribution.",
                    portfolio_id=portfolio_id,
                )

    stress_results = _rows(data.get("stress_results"))

    if not stress_results:
        report.add(
            "info",
            "STRESS_DATA_UNAVAILABLE",
            "No stress scenarios are available.",
            portfolio_id=portfolio_id,
            chart=CHART_STRESS,
        )
        report.disable_chart(
            CHART_STRESS,
            "No stress scenarios are available.",
            portfolio_id=portfolio_id,
        )

    scenario_ids: set[str] = set()

    for index, item in enumerate(stress_results):
        scenario = _as_dict(item)
        scenario_id = scenario.get("scenario_id")
        scenario_path = f"{path}.stress_results[{index}]"

        if (
            not _nonempty_identifier(scenario_id)
            or scenario_id in scenario_ids
        ):
            report.add(
                "warning",
                "INVALID_STRESS_SCENARIO_ID",
                "Stress scenario ID is missing or duplicated.",
                field_path=f"{scenario_path}.scenario_id",
                portfolio_id=portfolio_id,
                chart=CHART_STRESS,
            )
            report.disable_chart(
                CHART_STRESS,
                "Invalid stress scenario identifier.",
                portfolio_id=portfolio_id,
            )

        if _nonempty_identifier(scenario_id):
            scenario_ids.add(scenario_id)

        shock = scenario.get("portfolio_shock_return")

        if not _finite_number(shock):
            report.add(
                "warning",
                "INVALID_STRESS_RETURN",
                "Stress scenario return must be finite.",
                field_path=f"{scenario_path}.portfolio_shock_return",
                portfolio_id=portfolio_id,
                chart=CHART_STRESS,
            )
            report.disable_chart(
                CHART_STRESS,
                "Invalid stress scenario return.",
                portfolio_id=portfolio_id,
            )

        if scenario.get("hypothetical") is not True:
            report.add(
                "warning",
                "STRESS_NOT_HYPOTHETICAL",
                "Stress scenarios must be labeled hypothetical.",
                field_path=f"{scenario_path}.hypothetical",
                portfolio_id=portfolio_id,
                chart=CHART_STRESS,
            )
            report.disable_chart(
                CHART_STRESS,
                "Stress scenario is not labeled hypothetical.",
                portfolio_id=portfolio_id,
            )

        for j, contribution in enumerate(
            _rows(scenario.get("contributions"))
        ):
            item_data = _as_dict(contribution)
            ticker = item_data.get("ticker")
            value = item_data.get("contribution")

            if (
                ticker not in allocation_tickers[portfolio_id]
                or not _finite_number(value)
            ):
                report.add(
                    "warning",
                    "INVALID_STRESS_CONTRIBUTION",
                    "Stress contribution has an unknown ticker or invalid value.",
                    field_path=(
                        f"{scenario_path}.contributions[{j}]"
                    ),
                    portfolio_id=portfolio_id,
                    chart=CHART_STRESS,
                )
                report.disable_chart(
                    CHART_STRESS,
                    "Invalid stress contribution.",
                    portfolio_id=portfolio_id,
                )


def validate_visualization_dataset(
    dataset: VisualizationDataset | Mapping[str, Any],
) -> VisualizationValidationReport:
    """Validate a Skill #5 dataset and determine chart availability.

    Structural Pydantic failures are critical.
    Missing optional frontier/risk data disables only relevant charts.
    The source dataset is never mutated.
    """

    report = VisualizationValidationReport()

    try:
        validated = VisualizationDataset.model_validate(dataset)
    except (ValidationError, TypeError, ValueError) as exc:
        report.add(
            "error",
            "DATASET_SCHEMA_INVALID",
            f"Visualization dataset failed schema validation: {exc}",
        )

        for chart in ALL_CHARTS:
            report.disable_chart(
                chart,
                "Dataset schema validation failed.",
            )

        return report

    data = validated.model_dump(mode="python")
    portfolios = _rows(data.get("portfolios"))

    if not portfolios:
        report.add(
            "error",
            "EMPTY_DATASET",
            "Visualization dataset contains no portfolios.",
            field_path="portfolios",
        )

        for chart in ALL_CHARTS:
            report.disable_chart(
                chart,
                "No portfolios are available.",
            )

        return report

    portfolio_ids: set[str] = set()
    allocation_tickers: dict[str, set[str]] = {}

    for index, item in enumerate(portfolios):
        portfolio = _as_dict(item)
        portfolio_id = portfolio.get("portfolio_id")
        path = f"portfolios[{index}]"

        if not _nonempty_identifier(portfolio_id):
            report.add(
                "error",
                "INVALID_PORTFOLIO_ID",
                "Portfolio ID must be a nonempty string.",
                field_path=f"{path}.portfolio_id",
            )
            continue

        if portfolio_id in portfolio_ids:
            report.block_portfolio(
                portfolio_id,
                "Duplicate portfolio ID detected.",
                code="DUPLICATE_PORTFOLIO_ID",
                field_path=f"{path}.portfolio_id",
            )
            continue

        portfolio_ids.add(portfolio_id)

        allocations = _rows(portfolio.get("allocations"))
        allocation_tickers[portfolio_id] = {
            _as_dict(row).get("ticker")
            for row in allocations
            if _nonempty_identifier(
                _as_dict(row).get("ticker")
            )
        }

        allocation_valid = _validate_allocations(
            allocations,
            report,
            portfolio_id=portfolio_id,
            path=f"{path}.allocations",
        )

        if not allocation_valid:
            continue

        report.validated_portfolios.append(portfolio_id)

        _validate_metrics(
            portfolio.get("portfolio_metrics"),
            report,
            portfolio_id=portfolio_id,
            path=f"{path}.portfolio_metrics",
        )

        _validate_provenance(
            portfolio.get("provenance"),
            report,
            portfolio_id=portfolio_id,
            path=f"{path}.provenance",
        )

    frontiers = _rows(data.get("efficient_frontiers"))

    frontier_ids: set[str] = set()

    for index, item in enumerate(frontiers):
        frontier = _as_dict(item)
        portfolio_id = frontier.get("portfolio_id")

        if portfolio_id in frontier_ids:
            report.add(
                "warning",
                "DUPLICATE_FRONTIER",
                "Multiple efficient frontiers reference the same portfolio.",
                portfolio_id=portfolio_id,
                chart=CHART_FRONTIER,
            )
            report.disable_chart(
                CHART_FRONTIER,
                "Duplicate frontier datasets.",
                portfolio_id=portfolio_id,
            )

        if _nonempty_identifier(portfolio_id):
            frontier_ids.add(portfolio_id)

        _validate_frontier(
            frontier,
            report,
            portfolio_ids=portfolio_ids,
            path=f"efficient_frontiers[{index}]",
        )

    for portfolio_id in portfolio_ids - frontier_ids:
        report.disable_chart(
            CHART_FRONTIER,
            "Efficient frontier data is unavailable.",
            portfolio_id=portfolio_id,
        )

    risks = _rows(data.get("risk_results"))
    risk_ids: set[str] = set()

    for index, item in enumerate(risks):
        risk = _as_dict(item)
        portfolio_id = risk.get("portfolio_id")

        if portfolio_id in risk_ids:
            report.add(
                "warning",
                "DUPLICATE_RISK_RESULT",
                "Multiple risk results reference the same portfolio.",
                portfolio_id=portfolio_id,
                chart=CHART_RISK,
            )
            report.disable_chart(
                CHART_RISK,
                "Duplicate risk datasets.",
                portfolio_id=portfolio_id,
            )
            report.disable_chart(
                CHART_STRESS,
                "Duplicate risk datasets.",
                portfolio_id=portfolio_id,
            )

        if _nonempty_identifier(portfolio_id):
            risk_ids.add(portfolio_id)

        _validate_risk(
            risk,
            report,
            portfolio_ids=portfolio_ids,
            allocation_tickers=allocation_tickers,
            path=f"risk_results[{index}]",
        )

    for portfolio_id in portfolio_ids - risk_ids:
        report.disable_chart(
            CHART_RISK,
            "Risk attribution data is unavailable.",
            portfolio_id=portfolio_id,
        )
        report.disable_chart(
            CHART_STRESS,
            "Stress scenario data is unavailable.",
            portfolio_id=portfolio_id,
        )

    if len(report.validated_portfolios) < 2:
        report.disable_chart(
            CHART_COMPARISON,
            "At least two valid portfolios are required.",
        )

    # Preserve upstream diagnostics without changing their severity.
    for index, item in enumerate(
        _rows(data.get("diagnostics"))
    ):
        diagnostic = _as_dict(item)
        severity = diagnostic.get("severity")

        if severity not in ("error", "warning", "info"):
            severity = "warning"

        report.add(
            severity,
            str(
                diagnostic.get("code")
                or "UPSTREAM_DIAGNOSTIC"
            ),
            str(
                diagnostic.get("message")
                or "Upstream diagnostic reported."
            ),
            field_path=diagnostic.get("field_path"),
        )

    return report


def validate_visualization_payload(
    payload: Mapping[str, Any],
) -> VisualizationValidationReport:
    """Validate a JSON-compatible visualization payload."""

    return validate_visualization_dataset(payload)


__all__ = [
    "VALIDATION_VERSION",
    "WEIGHT_TOLERANCE",
    "METRIC_TOLERANCE",
    "ValidationFinding",
    "VisualizationValidationReport",
    "validate_visualization_dataset",
    "validate_visualization_payload",
    "ALL_CHARTS",
    "CHART_ALLOCATION",
    "CHART_OVERVIEW",
    "CHART_FRONTIER",
    "CHART_RISK",
    "CHART_STRESS",
    "CHART_COMPARISON",
]
