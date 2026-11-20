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
        raise ValueError("Confidence level must be strictly between zero and one")
    if annual_volatility < 0 or days < 1 or trading_days_per_year < 1:
        raise ValueError("Volatility, horizon, or trading-day count is invalid")
    mu = annual_mean * days / trading_days_per_year
    sigma = annual_volatility * sqrt(days / trading_days_per_year)
    z = NORMAL.inv_cdf(confidence_level)
    var = -mu + z * sigma
    cvar = -mu + sigma * exp(-0.5 * z * z) / (sqrt(2 * pi) * (1 - confidence_level))
    return var, cvar


def historical_portfolio_losses(request: RiskAssessmentRequest, days: int) -> list[float]:
    """Compounded fixed-weight daily-rebalanced portfolio losses for each window.

    A day with a -100% portfolio return produces a full loss, never a negative
    wealth value. The windows overlap for horizons greater than one trading day.
    """
    history = request.portfolio.historical_returns
    if history is None:
        return []
    weights = [asset.weight for asset in request.portfolio.assets]
    daily_returns = [sum(weight * value for weight, value in zip(weights, row))
                     for row in history.observations]
    if len(daily_returns) < days:
        return []
    losses: list[float] = []
    for start in range(len(daily_returns) - days + 1):
        wealth = 1.0
        for daily_return in daily_returns[start:start + days]:
            wealth *= 1.0 + daily_return
        loss = 1.0 - wealth
        if not isfinite(loss):
            raise ValueError("Historical compounding overflowed; review observations")
        losses.append(loss)
    return losses


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Quantile requires at least one observation")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def historical_tail_risk(losses: Sequence[float], confidence_level: float) -> tuple[float, float]:
    """Interpolated empirical VaR and discrete expected shortfall.

    CVaR is the mean of the worst (1-confidence) probability mass, allowing
    fractional contribution from the cutoff observation. This guarantees CVaR
    >= VaR under the discrete empirical distribution.
    """
    if not losses or not 0 < confidence_level < 1:
        raise ValueError("Historical losses and a valid confidence level are required")
    if any(not isfinite(value) for value in losses):
        raise ValueError("Historical losses must be finite")
    ordered = sorted(losses)
    var = _linear_quantile(ordered, confidence_level)
    tail_count = len(ordered) * (1.0 - confidence_level)
    whole = int(tail_count)
    fraction = tail_count - whole
    tail_sum = sum(ordered[-whole:]) if whole else 0.0
    if fraction > 1e-12:
        tail_sum += fraction * ordered[-whole - 1]
    cvar = tail_sum / tail_count
    return var, max(cvar, var)


def assess_portfolio_risk(request: RiskAssessmentRequest) -> RiskAssessmentResult:
    """Compute Stage 4.2 metrics with explicit historical availability rules.

    `auto`: historical if >= max(20, ceil(1/(1-confidence))) rolling windows,
    otherwise parametric, with a recorded fallback reason. Explicit `historical`
    never silently falls back; it returns PARTIAL and disables VaR/CVaR.
    Classification, stress scenarios and investor alignment are intentionally
    left to Stage 4.3. The result is not downstream-ready until those stages.
    """
    config = request.config
    days = horizon_days(config.horizon, config.trading_days_per_year)
    annual_mean, _, annual_volatility = portfolio_moments(request)
    contributions = risk_attribution(request)
    issues: list[ValidationIssue] = []
    disabled: list[str] = []
    historical_losses = historical_portfolio_losses(request, days)
    # Require at least one expected tail observation and a base sample size.
    from math import ceil
    minimum = max(MIN_HISTORICAL_WINDOWS, ceil(1 / (1 - config.confidence_level)))
    enough_history = len(historical_losses) >= minimum
    selected: RiskMethod | None = None
    fallback_reason: str | None = None
    if config.method == RiskMethod.PARAMETRIC:
        selected = RiskMethod.PARAMETRIC
    elif config.method == RiskMethod.HISTORICAL:
        if enough_history:
            selected = RiskMethod.HISTORICAL
        else:
            disabled.extend(["value_at_risk", "conditional_value_at_risk"])
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="INSUFFICIENT_HISTORICAL_WINDOWS",
                field_path="portfolio.historical_returns",
                message=(f"Historical simulation requires at least {minimum} rolling "
                         f"{days}-day windows; received {len(historical_losses)}. "
                         "VaR/CVaR were not calculated."),
            ))
    else:
        if enough_history:
            selected = RiskMethod.HISTORICAL
        else:
            selected = RiskMethod.PARAMETRIC
            fallback_reason = (f"Historical simulation unavailable: {len(historical_losses)} "
                               f"rolling {days}-day windows; minimum {minimum}. "
                               "Parametric normal approximation selected.")
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="PARAMETRIC_FALLBACK",
                message=fallback_reason,
                field_path="config.method",
            ))

    var_measure = None
    cvar_measure = None
    if selected is not None:
        if selected == RiskMethod.PARAMETRIC:
            var, cvar = parametric_tail_risk(
                annual_mean, annual_volatility, config.confidence_level,
                days, config.trading_days_per_year,
            )
            assumptions = [
                "Normal distribution of portfolio simple returns",
                "Annual expected return scaled linearly by trading-day fraction",
                "Annual volatility scaled by square root of trading-day fraction",
                "Positive loss convention; negative VaR indicates a gain threshold",
            ]
        else:
            var, cvar = historical_tail_risk(historical_losses, config.confidence_level)
            assumptions = [
                "Observed daily simple returns, in chronological order",
                "Fixed portfolio weights rebalanced daily",
                f"{days}-trading-day compounded overlapping return windows",
                "Linear-interpolated empirical quantile; fractional-mass empirical CVaR",
                f"{len(historical_losses)} rolling windows; overlapping windows are dependent",
            ]
        var_measure = RiskMeasure(
            value=var, method=selected, horizon=config.horizon,
            confidence_level=config.confidence_level, assumptions=assumptions,
        )
        cvar_measure = RiskMeasure(
            value=cvar, method=selected, horizon=config.horizon,
            confidence_level=config.confidence_level, assumptions=assumptions,
        )

    if annual_volatility <= 1e-14:
        issues.append(ValidationIssue(
            severity=IssueSeverity.INFO,
            code="ZERO_PORTFOLIO_VOLATILITY",
            message="Portfolio volatility is zero; risk contributions are reported as zero.",
        ))
    # Stage 4.3 outputs are deliberately absent; this is an intermediate result.
    disabled.extend(["stress_results", "classification", "investor_alignment"])
    return RiskAssessmentResult(
        assessment_id=request.assessment_id,
        status=RiskStatus.PARTIAL,
        portfolio_id=request.portfolio.portfolio_id,
        method_requested=config.method,
        method_used=selected,
        method_fallback_reason=fallback_reason,
        horizon=config.horizon,
        confidence_level=config.confidence_level,
        annualized_volatility=annual_volatility,
        value_at_risk=var_measure,
        conditional_value_at_risk=cvar_measure,
        asset_risk_contributions=contributions,
        validation=ValidationReport(
            status="warning" if issues else "pass",
            issues=issues,
            disabled_calculations=disabled,
        ),
        provenance=request.portfolio.provenance,
        calculation_version=CALCULATION_VERSION,
        downstream_ready=False,
    )


__all__ = [
    "CALCULATION_VERSION", "MIN_HISTORICAL_WINDOWS", "horizon_days",
    "portfolio_moments", "risk_attribution", "parametric_tail_risk",
    "historical_portfolio_losses", "historical_tail_risk", "assess_portfolio_risk",
]
