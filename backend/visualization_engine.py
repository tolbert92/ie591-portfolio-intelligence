"""IE 591 Skill #5 — chart-ready portfolio visualization engine.

Pure transformation of validated visualization data. No API calls, investment
recommendations, optimization, or risk recalculation. All rates use decimals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from pydantic import BaseModel

from .visualization_models import VisualizationDataset
from .visualization_validation import (
    VisualizationValidationReport,
    validate_visualization_dataset,
    CHART_ALLOCATION,
    CHART_OVERVIEW,
    CHART_FRONTIER,
    CHART_RISK,
    CHART_STRESS,
)

ENGINE_VERSION = "1.0.0"


class VisualizationEngineError(ValueError):
    """A requested chart cannot be safely produced."""


@dataclass(frozen=True)
class ChartDataset:
    """Serializable, renderer-neutral chart specification."""

    chart: str
    portfolio_id: str
    available: bool
    data: dict[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart": self.chart,
            "portfolio_id": self.portfolio_id,
            "available": self.available,
            "data": self.data,
            "reasons": list(self.reasons),
        }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    raise VisualizationEngineError("Expected a model or mapping")


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (tuple, list)):
        return []
    return [_mapping(item) for item in value]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _clean(value: Any) -> Any:
    """Convert nested models to JSON-friendly values; never emit NaN/Infinity."""
    from datetime import date, datetime
    from enum import Enum

    if isinstance(value, BaseModel):
        return _clean(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise VisualizationEngineError("Chart data must contain finite numbers")
    return value


def _reasons(report: VisualizationValidationReport, chart: str, portfolio_id: str) -> tuple[str, ...]:
    reasons = list(report.disabled_charts.get(chart, []))
    reasons.extend(report.disabled_charts.get(f"{portfolio_id}:{chart}", []))
    if portfolio_id in report.blocked_portfolios:
        reasons.append("Portfolio failed critical validation")
    return tuple(dict.fromkeys(reasons))


def _chart(
    report: VisualizationValidationReport,
    chart: str,
    portfolio_id: str,
    data: dict[str, Any],
) -> ChartDataset:
    reasons = _reasons(report, chart, portfolio_id)
    available = report.chart_available(chart, portfolio_id=portfolio_id)
    return ChartDataset(
        chart=chart,
        portfolio_id=portfolio_id,
        available=available,
        data=_clean(data) if available else {},
        reasons=reasons if not available else (),
    )


def _allocation(portfolio: dict[str, Any], report: VisualizationValidationReport) -> ChartDataset:
    portfolio_id = portfolio["portfolio_id"]
    allocations = sorted(
        _rows(portfolio.get("allocations")),
        key=lambda row: (-row["weight"], row["ticker"]),
    )
    by_class: dict[str, float] = {}
    for asset in allocations:
        asset_class = asset.get("asset_class") or "unclassified"
        by_class[asset_class] = by_class.get(asset_class, 0.0) + asset["weight"]
    return _chart(report, CHART_ALLOCATION, portfolio_id, {
        "units": "decimal_weights",
        "assets": [
            {
                "ticker": a["ticker"],
                "asset_name": a.get("asset_name"),
                "asset_class": a.get("asset_class") or "unclassified",
                "weight": a["weight"],
            }
            for a in allocations
        ],
        "asset_classes": [
            {"asset_class": name, "weight": weight}
            for name, weight in sorted(by_class.items())
        ],
        "total_weight": sum(a["weight"] for a in allocations),
    })


def _overview(portfolio: dict[str, Any], report: VisualizationValidationReport) -> ChartDataset:
    portfolio_id = portfolio["portfolio_id"]
    metrics = portfolio.get("portfolio_metrics")
    return _chart(report, CHART_OVERVIEW, portfolio_id, {
        "metrics": _mapping(metrics) if metrics is not None else {},
        "units": {"returns": "decimal_annual", "volatility": "decimal_annual"},
        "provenance": portfolio.get("provenance"),
    })


def _frontier(frontier: dict[str, Any], report: VisualizationValidationReport) -> ChartDataset:
    portfolio_id = frontier["portfolio_id"]
    points = _rows(frontier.get("points"))
    points = sorted(points, key=lambda p: (p["annualized_volatility"], p["expected_annual_return"]))
    return _chart(report, CHART_FRONTIER, portfolio_id, {
        "x_axis": "annualized_volatility",
        "y_axis": "expected_annual_return",
        "units": "decimal_annual",
        "points": [
            {
                "target_annual_return": p.get("target_annual_return"),
                "expected_annual_return": p["expected_annual_return"],
                "annualized_volatility": p["annualized_volatility"],
                "allocations": _rows(p.get("allocations")),
            }
            for p in points
        ],
        "provenance": frontier.get("provenance"),
    })


def _risk(risk: dict[str, Any], report: VisualizationValidationReport) -> ChartDataset:
    portfolio_id = risk["portfolio_id"]
    rows = _rows(risk.get("asset_risk_contributions"))
    return _chart(report, CHART_RISK, portfolio_id, {
        "status": risk.get("status"),
        "annualized_volatility": risk.get("annualized_volatility"),
        "value_at_risk": risk.get("value_at_risk"),
        "conditional_value_at_risk": risk.get("conditional_value_at_risk"),
        "contributions": sorted(rows, key=lambda r: r["ticker"]),
        "contribution_unit": "source_defined_percent_total_risk",
        "provenance": risk.get("provenance"),
        "diagnostics": risk.get("diagnostics", []),
    })


def _stress(risk: dict[str, Any], report: VisualizationValidationReport) -> ChartDataset:
    portfolio_id = risk["portfolio_id"]
    scenarios = _rows(risk.get("stress_results"))
    return _chart(report, CHART_STRESS, portfolio_id, {
        "units": "decimal_return",
        "hypothetical": True,
        "scenarios": sorted(scenarios, key=lambda s: s["scenario_id"]),
        "provenance": risk.get("provenance"),
    })


def build_visualization_charts(
    dataset: VisualizationDataset | Mapping[str, Any],
    *,
    report: VisualizationValidationReport | None = None,
) -> dict[str, Any]:
    """Produce chart-ready data for all portfolios in a validated dataset.

    A critical schema error raises VisualizationEngineError. Optional chart
    limitations are returned as unavailable chart specifications, allowing
    unaffected charts to render. The input is never mutated.
    """
    if report is None:
        report = validate_visualization_dataset(dataset)

    if report.has_errors:
        raise VisualizationEngineError(
            "Visualization input contains critical validation errors: "
            + "; ".join(f.message for f in report.findings if f.severity == "error")
        )

    validated = VisualizationDataset.model_validate(dataset)
    source = validated.model_dump(mode="python")
    portfolios = _rows(source.get("portfolios"))
    frontiers = {row["portfolio_id"]: row for row in _rows(source.get("efficient_frontiers"))}
    risks = {row["portfolio_id"]: row for row in _rows(source.get("risk_results"))}

    result: dict[str, dict[str, Any]] = {}
    for portfolio in portfolios:
        portfolio_id = portfolio["portfolio_id"]
        charts: dict[str, ChartDataset] = {
            CHART_ALLOCATION: _allocation(portfolio, report),
            CHART_OVERVIEW: _overview(portfolio, report),
        }
        frontier = frontiers.get(portfolio_id)
        if frontier is not None:
            charts[CHART_FRONTIER] = _frontier(frontier, report)
        else:
            charts[CHART_FRONTIER] = _chart(report, CHART_FRONTIER, portfolio_id, {})
        risk = risks.get(portfolio_id)
        if risk is not None:
            charts[CHART_RISK] = _risk(risk, report)
            charts[CHART_STRESS] = _stress(risk, report)
        else:
            charts[CHART_RISK] = _chart(report, CHART_RISK, portfolio_id, {})
            charts[CHART_STRESS] = _chart(report, CHART_STRESS, portfolio_id, {})
        result[portfolio_id] = {name: spec.to_dict() for name, spec in charts.items()}

    return _clean({
        "engine_version": ENGINE_VERSION,
        "validation": report.to_dict(),
        "portfolios": result,
    })


__all__ = [
    "ENGINE_VERSION",
    "VisualizationEngineError",
    "ChartDataset",
    "build_visualization_charts",
]
