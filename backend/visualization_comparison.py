"""IE 591 Skill #5 — validated, renderer-neutral portfolio comparisons.

All weights, returns, and volatility values are decimals. Differences use
percentage-point-equivalent decimal subtraction (0.02 means two percentage
points). Comparisons never imply that differently sourced data are equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from .visualization_models import VisualizationDataset
from .visualization_validation import (
    VisualizationValidationReport,
    validate_visualization_dataset,
)

COMPARISON_VERSION = "1.0.0"
METRICS = (
    "expected_annual_return",
    "annualized_volatility",
    "annualized_variance",
    "sharpe_ratio",
)


class VisualizationComparisonError(ValueError):
    """Portfolio comparison cannot be constructed safely."""


@dataclass(frozen=True)
class ComparisonWarning:
    code: str
    message: str
    portfolio_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "portfolio_ids": list(self.portfolio_ids),
        }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    raise VisualizationComparisonError("Expected a Pydantic model or mapping")


def _rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise VisualizationComparisonError("Expected a sequence of records")
    return [_mapping(item) for item in value]


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _metric(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _provenance(portfolio: dict[str, Any]) -> dict[str, Any]:
    source = portfolio.get("provenance")
    return _mapping(source) if source is not None else {}


def _allocation_map(portfolio: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in _rows(portfolio.get("allocations")):
        ticker = row.get("ticker")
        weight = row.get("weight")
        if not isinstance(ticker, str) or not ticker.strip():
            raise VisualizationComparisonError("Allocation ticker is invalid")
        if ticker in result:
            raise VisualizationComparisonError(f"Duplicate ticker: {ticker}")
        if not _finite(weight) or not 0 <= weight <= 1:
            raise VisualizationComparisonError(f"Invalid weight for {ticker}")
        result[ticker] = float(weight)
    return result


def _risk_map(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for risk in _rows(dataset.get("risk_results")):
        portfolio_id = risk.get("portfolio_id")
        if portfolio_id in result:
            raise VisualizationComparisonError("Duplicate risk result identity")
        result[portfolio_id] = risk
    return result


def _source_warnings(
    portfolios: Sequence[dict[str, Any]],
    ids: tuple[str, ...],
) -> list[ComparisonWarning]:
    warnings: list[ComparisonWarning] = []
    provenance = [_provenance(portfolio) for portfolio in portfolios]
    for field, code, message in (
        (
            "is_synthetic",
            "MIXED_SYNTHETIC_STATUS",
            "Synthetic and nonsynthetic portfolios are being compared.",
        ),
        (
            "source_type",
            "MIXED_SOURCE_TYPES",
            "Portfolio source types differ; review comparability.",
        ),
        (
            "dataset_id",
            "DIFFERENT_DATASETS",
            "Portfolios reference different datasets; assumptions may differ.",
        ),
        (
            "calculation_version",
            "DIFFERENT_CALCULATION_VERSIONS",
            "Portfolio calculation versions differ.",
        ),
    ):
        values = [item.get(field) for item in provenance]
        if any(value != values[0] for value in values[1:]):
            warnings.append(ComparisonWarning(code, message, ids))
    return warnings


def compare_visualization_portfolios(
    dataset: VisualizationDataset | Mapping[str, Any],
    portfolio_ids: Sequence[str],
    *,
    report: VisualizationValidationReport | None = None,
) -> dict[str, Any]:
    """Compare two or more distinct, valid portfolios without mutating input.

    Returns chart-ready allocation and metric tables. Missing optional metrics
    remain null; no estimates are substituted. Metric deltas are calculated
    relative to the first selected portfolio and preserve the source units.
    """
    if not isinstance(portfolio_ids, (list, tuple)):
        raise VisualizationComparisonError("portfolio_ids must be a list or tuple")
    ids = tuple(portfolio_ids)
    if len(ids) < 2:
        raise VisualizationComparisonError("Select at least two portfolios")
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise VisualizationComparisonError("Portfolio IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise VisualizationComparisonError("Portfolio IDs must be unique")

    if report is None:
        report = validate_visualization_dataset(dataset)
    if report.has_errors:
        raise VisualizationComparisonError(
            "Dataset has critical validation errors: "
            + "; ".join(
                finding.message
                for finding in report.findings
                if finding.severity == "error"
            )
        )

    try:
        validated = VisualizationDataset.model_validate(dataset)
    except (TypeError, ValueError) as exc:
        raise VisualizationComparisonError("Invalid visualization dataset") from exc
    source = validated.model_dump(mode="python")
    portfolios_by_id = {
        portfolio["portfolio_id"]: portfolio
        for portfolio in _rows(source.get("portfolios"))
    }
    unknown = [item for item in ids if item not in portfolios_by_id]
    if unknown:
        raise VisualizationComparisonError(f"Unknown portfolio IDs: {unknown}")
    blocked = [item for item in ids if item in report.blocked_portfolios]
    if blocked:
        raise VisualizationComparisonError(f"Blocked portfolio IDs: {blocked}")

    portfolios = [portfolios_by_id[item] for item in ids]
    allocations = [_allocation_map(portfolio) for portfolio in portfolios]
    tickers = sorted(set().union(*(set(row) for row in allocations)))
    risk_by_id = _risk_map(source)
    warnings = _source_warnings(portfolios, ids)

    metric_rows: list[dict[str, Any]] = []
    baseline_metrics = _mapping(portfolios[0]["portfolio_metrics"]) if portfolios[0].get("portfolio_metrics") is not None else {}
    for name in METRICS:
        values = []
        for portfolio in portfolios:
            metrics = portfolio.get("portfolio_metrics")
            metrics_data = _mapping(metrics) if metrics is not None else {}
            values.append(_metric(metrics_data.get(name)))
        baseline = _metric(baseline_metrics.get(name))
        metric_rows.append({
            "metric": name,
            "unit": "ratio" if name == "sharpe_ratio" else "decimal_annual",
            "values": {portfolio_id: value for portfolio_id, value in zip(ids, values)},
            "delta_from_baseline": {
                portfolio_id: (
                    value - baseline if value is not None and baseline is not None else None
                )
                for portfolio_id, value in zip(ids, values)
            },
        })
        if any(value is None for value in values):
            warnings.append(ComparisonWarning(
                "METRIC_UNAVAILABLE",
                f"Metric '{name}' is unavailable for one or more portfolios.",
                ids,
            ))

    risk_rows: list[dict[str, Any]] = []
    for portfolio_id in ids:
        risk = risk_by_id.get(portfolio_id)
        risk_rows.append({
            "portfolio_id": portfolio_id,
            "status": risk.get("status") if risk else None,
            "annualized_volatility": _metric(risk.get("annualized_volatility")) if risk else None,
            "value_at_risk": risk.get("value_at_risk") if risk else None,
            "conditional_value_at_risk": risk.get("conditional_value_at_risk") if risk else None,
            "provenance": risk.get("provenance") if risk else None,
        })
        if risk is None or risk.get("status") != "completed":
            warnings.append(ComparisonWarning(
                "RISK_NOT_FULLY_AVAILABLE",
                f"Complete risk results are unavailable for '{portfolio_id}'.",
                (portfolio_id,),
            ))

    allocation_rows = [
        {
            "ticker": ticker,
            "weights": {
                portfolio_id: allocations[index].get(ticker, 0.0)
                for index, portfolio_id in enumerate(ids)
            },
            "delta_from_baseline": {
                portfolio_id: allocations[index].get(ticker, 0.0) - allocations[0].get(ticker, 0.0)
                for index, portfolio_id in enumerate(ids)
            },
        }
        for ticker in tickers
    ]

    return {
        "comparison_version": COMPARISON_VERSION,
        "portfolio_ids": list(ids),
        "baseline_portfolio_id": ids[0],
        "units": {
            "allocation": "decimal_weight",
            "returns": "decimal_annual",
            "volatility": "decimal_annual",
            "delta": "arithmetic_difference_in_source_units",
        },
        "portfolios": [
            {
                "portfolio_id": portfolio_id,
                "provenance": portfolio.get("provenance"),
                "asset_count": len(allocations[index]),
            }
            for index, (portfolio_id, portfolio) in enumerate(zip(ids, portfolios))
        ],
        "allocation_comparison": allocation_rows,
        "metric_comparison": metric_rows,
        "risk_comparison": risk_rows,
        "warnings": [warning.to_dict() for warning in warnings],
        "hypothetical": False,
    }


__all__ = [
    "COMPARISON_VERSION",
    "METRICS",
    "VisualizationComparisonError",
    "ComparisonWarning",
    "compare_visualization_portfolios",
]
