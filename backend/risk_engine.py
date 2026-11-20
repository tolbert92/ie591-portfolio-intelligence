
"""IE 591 Skill #4 — Stage 4.2: deterministic portfolio risk calculation engine.

Inputs and outputs use backend.risk_models (Pydantic v2). No API, I/O, network,
random simulation, stress scenarios, classification, or investor recommendations.
Returns and loss measures are decimal fractions, not percentages or currency.

Parametric VaR/CVaR: normal simple-return approximation with annual mean and
covariance scaled by trading days; loss = -return (may be negative for positive
expected returns). Historical VaR/CVaR: overlapping compounded rolling portfolio
returns from observed daily asset returns, using fixed-weight daily rebalancing.
Quantiles use linear interpolation. Historical CVaR uses the empirical expected
loss above the quantile with fractional boundary mass, not a simple tail mean.
"""
from __future__ import annotations

from math import exp, isfinite, pi, sqrt
from statistics import NormalDist
from typing import Sequence

try:
    from .risk_models import (
        AssetRiskContribution, IssueSeverity, RiskAssessmentRequest,
        RiskAssessmentResult, RiskHorizon, RiskMeasure, RiskMethod, RiskStatus,
        ValidationIssue, ValidationReport,
    )
except ImportError:  # Permit `python backend/risk_engine.py` from repo root.
    from risk_models import (
        AssetRiskContribution, IssueSeverity, RiskAssessmentRequest,
        RiskAssessmentResult, RiskHorizon, RiskMeasure, RiskMethod, RiskStatus,
        ValidationIssue, ValidationReport,
    )

CALCULATION_VERSION = "1.0.0"
MIN_HISTORICAL_WINDOWS = 20
NORMAL = NormalDist()


def horizon_days(horizon: RiskHorizon, trading_days_per_year: int = 252) -> int:
    """Map approved horizons to trading days (30d means 30 trading days)."""
    mapping = {
        RiskHorizon.ONE_DAY: 1,
        RiskHorizon.TEN_DAYS: 10,
        RiskHorizon.THIRTY_DAYS: 30,
        RiskHorizon.ONE_YEAR: trading_days_per_year,
    }
    return mapping[RiskHorizon(horizon)]


def portfolio_moments(request: RiskAssessmentRequest) -> tuple[float, float, float]:
    """Annual expected return, annual variance, annual volatility."""
    assets = request.portfolio.assets
    weights = [asset.weight for asset in assets]
    covariance = request.portfolio.covariance_matrix
    mean = sum(asset.weight * asset.expected_annual_return for asset in assets)
    variance = sum(
        weights[i] * covariance[i][j] * weights[j]
        for i in range(len(weights)) for j in range(len(weights))
    )
    # The input PSD check uses numerical tolerance; tiny negatives are roundoff.
    if variance < -1e-9:
        raise ValueError("Portfolio variance is materially negative")
    variance = max(variance, 0.0)
    return mean, variance, sqrt(variance)


def risk_attribution(request: RiskAssessmentRequest) -> list[AssetRiskContribution]:
    """Euler volatility attribution; zero-volatility portfolios return zeroes."""
    _, _, volatility = portfolio_moments(request)
    assets = request.portfolio.assets
    covariance = request.portfolio.covariance_matrix
    weights = [asset.weight for asset in assets]
    out: list[AssetRiskContribution] = []
    for i, asset in enumerate(assets):
        covariance_with_portfolio = sum(covariance[i][j] * weights[j] for j in range(len(assets)))
        marginal = covariance_with_portfolio / volatility if volatility > 1e-14 else 0.0
        component = asset.weight * marginal
        share = component / volatility if volatility > 1e-14 else 0.0
        out.append(AssetRiskContribution(
            ticker=asset.ticker,
            weight=asset.weight,
            marginal_volatility=marginal,
            component_volatility=component,
            percent_total_risk=share,
        ))
    return out


def parametric_tail_risk(
    annual_mean: float,
    annual_volatility: float,
    confidence_level: float,
    days: int,
    trading_days_per_year: int = 252,
) -> tuple[float, float]:
    """Normal-distribution VaR and CVaR for positive-loss convention."""
    if not (0 < confidence_level < 1):




