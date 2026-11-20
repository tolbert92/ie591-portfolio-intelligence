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
