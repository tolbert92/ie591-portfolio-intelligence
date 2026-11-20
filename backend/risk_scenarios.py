"""IE 591 Skill #4 — deterministic portfolio stress scenarios (Stage 4.3.1).

All shocks are hypothetical one-period simple returns, expressed as decimals.
This module never estimates probabilities, forecasts returns, or modifies inputs.
It uses the validated Pydantic v2 contracts from backend/risk_models.py.
"""
from __future__ import annotations

from math import fsum, isfinite
from typing import Iterable

try:  # Support both package imports and backend working-directory execution.
    from .risk_models import (
        AssetShock, RiskAssessmentRequest, RiskPortfolio, StressResult,
        StressScenario,
    )
except ImportError:
    from risk_models import (
        AssetShock, RiskAssessmentRequest, RiskPortfolio, StressResult,
        StressScenario,
    )

SCENARIO_VERSION = "1.0.0"

# Illustrative scenarios, NOT historical episodes, forecasts, or risk probabilities.
# Each scenario explicitly specifies shocks for broad asset-class categories.
# Unknown classes receive a 0% shock and are reported in scenario metadata.
_EQUITY_CLASSES = frozenset({"equity", "equities", "stock", "stocks"})
_BOND_CLASSES = frozenset({"fixed_income", "fixed income", "bond", "bonds"})
_CASH_CLASSES = frozenset({"cash", "cash_equivalent", "cash equivalents", "money_market"})

_PREDEFINED = (
    (
        "equity_drawdown",
        "Hypothetical equity drawdown",
        {"equity": -0.20, "fixed_income": 0.02, "cash": 0.0},
    ),
    (
        "bond_selloff",
        "Hypothetical bond selloff",
        {"equity": -0.05, "fixed_income": -0.10, "cash": 0.0},
    ),
    (
        "combined_market_shock",
        "Hypothetical combined market shock",
        {"equity": -0.30, "fixed_income": -0.12, "cash": 0.0},
    ),
)


def _asset_category(asset_class: str) -> str | None:
    normalized = asset_class.strip().lower().replace("-", "_")
    if normalized in _EQUITY_CLASSES:
        return "equity"
    if normalized in _BOND_CLASSES:
        return "fixed_income"
    if normalized in _CASH_CLASSES:
        return "cash"
    return None


def predefined_scenarios(portfolio: RiskPortfolio) -> list[StressScenario]:
    """Build reproducible illustrative scenarios for this portfolio.

    Assets in unrecognized classes are assigned an explicit zero shock; callers
    can inspect ``unmapped_asset_classes`` to decide whether that is useful.
    """
    scenarios: list[StressScenario] = []
    for scenario_id, name, category_shocks in _PREDEFINED:
        shocks = [
            AssetShock(
                ticker=asset.ticker,
                shock_return=category_shocks.get(_asset_category(asset.asset_class), 0.0),
            )
            for asset in portfolio.assets
        ]
        scenarios.append(StressScenario(
            scenario_id=scenario_id,
            name=name,
            scenario_type="predefined",
            shocks=shocks,
        ))
    return scenarios


def unmapped_asset_classes(portfolio: RiskPortfolio) -> list[str]:
    """Return tickers whose class has no predefined scenario mapping."""
    return [asset.ticker for asset in portfolio.assets if _asset_category(asset.asset_class) is None]


def calculate_stress_result(
    portfolio: RiskPortfolio,
    scenario: StressScenario,
    *,
    require_complete: bool = False,
) -> StressResult:
    """Calculate a static buy-and-hold, one-period weighted portfolio shock.

    Unspecified asset shocks are 0% (unchanged), never silently reweighted.
    Set ``require_complete=True`` to reject incomplete custom scenarios.
    Unknown tickers and duplicate shocks are always rejected, even when
    unvalidated objects are supplied. Values use decimal units, e.g. -0.20.
    """
    assets = {asset.ticker: asset for asset in portfolio.assets}
    shocks: dict[str, float] = {}
    for item in scenario.shocks:
        if item.ticker not in assets:
            raise ValueError(f"Unknown stress scenario ticker: {item.ticker}")
        if item.ticker in shocks:
            raise ValueError(f"Duplicate stress scenario ticker: {item.ticker}")
        if not isfinite(item.shock_return) or item.shock_return < -1:
            raise ValueError(f"Invalid stress shock for ticker: {item.ticker}")
        shocks[item.ticker] = item.shock_return
    missing = set(assets) - set(shocks)
    if require_complete and missing:
        raise ValueError(f"Stress scenario missing asset shocks: {sorted(missing)}")

    contributions = {
        asset.ticker: asset.weight * shocks.get(asset.ticker, 0.0)
        for asset in portfolio.assets
    }
    total = fsum(contributions.values())
    return StressResult(
        scenario_id=scenario.scenario_id,
        portfolio_shock_return=total,
        hypothetical=True,
        contributions=contributions,
    )


def evaluate_stress_scenarios(
    request: RiskAssessmentRequest,
    *,
    include_predefined: bool = True,
    require_complete_custom: bool = False,
) -> list[StressResult]:
    """Evaluate predefined scenarios followed by request-provided scenarios.

    Scenario IDs must be unique across both sets; collisions raise errors.
    Custom scenarios may specify only affected assets (others remain at 0%).
    This is independent of the VaR/CVaR horizon and confidence level.
    """
    scenarios: list[StressScenario] = []
    if include_predefined:
        scenarios.extend(predefined_scenarios(request.portfolio))
    scenarios.extend(request.stress_scenarios)
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Stress scenario IDs must be unique across predefined and custom scenarios")
    return [
        calculate_stress_result(
            request.portfolio,
            scenario,
            require_complete=require_complete_custom and scenario.scenario_type == "custom",
        )
        for scenario in scenarios
    ]


def compare_stress_results(results: Iterable[StressResult]) -> list[StressResult]:
    """Order scenario outcomes from worst to best portfolio return.

    This is a deterministic comparison, not a probability ranking.
    """
    return sorted(results, key=lambda result: (result.portfolio_shock_return, result.scenario_id))


__all__ = [
    "SCENARIO_VERSION",
    "predefined_scenarios",
    "unmapped_asset_classes",
    "calculate_stress_result",
    "evaluate_stress_scenarios",
    "compare_stress_results",
]
