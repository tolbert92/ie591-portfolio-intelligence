
"""
IE 591 — Portfolio Intelligence
Skill #2: Market Intelligence
Gate 3.2 — Calculation Engine

Deterministic historical market analysis.

Calculations:
- Aligned historical prices
- Simple or logarithmic periodic returns
- Annualized arithmetic return estimates
- Sample standard deviation and volatility
- Sample covariance and Pearson correlation
- Cumulative returns
- Data quality and provenance reporting

No external data retrieval.
No OpenAI API key required.
Educational and research use.
"""

from collections import defaultdict
from datetime import datetime, timezone
from math import exp, isfinite, log, sqrt
from statistics import mean, stdev

from .market_models import (
    AssetStatistics,
    DataQualityIssue,
    DataQualityReport,
    MarketAnalysisRequest,
    MarketAnalysisResponse,
    MarketMatrix,
    MissingDataPolicy,
    QualityStatus,
    ReturnMethod,
    Severity,
)


CALCULATION_VERSION = "1.0.0"


# ------------------------------------------
# NUMERICAL UTILITIES
# ------------------------------------------

def sample_covariance(x, y):
    """Sample covariance using denominator n - 1."""
    if len(x) != len(y):
        raise ValueError("Series lengths must match.")

    if len(x) < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    mx = mean(x)
    my = mean(y)

    return sum(
        (a - mx) * (b - my)
        for a, b in zip(x, y)
    ) / (len(x) - 1)


def calculate_returns(prices, method):
    """Calculate periodic simple or log returns."""
    returns = []

    for previous, current in zip(
        prices[:-1], prices[1:]
    ):
        if previous <= 0 or current <= 0:
            raise ValueError(
                "Historical prices must be positive."
            )

        ratio = current / previous

        if method == ReturnMethod.LOG:
            value = log(ratio)
        else:
            value = ratio - 1.0

        if not isfinite(value):
            raise ValueError(
                "Non-finite return encountered."
            )

        returns.append(value)

    return returns


# ------------------------------------------
# QUALITY HELPERS
# ------------------------------------------

def make_issue(code, severity, message):
    return DataQualityIssue(
        code=code,
        severity=severity,
        message=message,
    )


def rejected_result(
    request,
    issues,
    common_price_dates=0,
    common_return_periods=0,
    excluded_dates=None,
):
    """Return a structured rejection, not fabricated stats."""
    return MarketAnalysisResponse(
        analysis_id=request.analysis_id,
        status="rejected",
        provenance=request.provenance,
        config=request.config,
        quality=DataQualityReport(
            status=QualityStatus.FAIL,
            source_observation_count=len(
                request.observations
            ),
            valid_common_price_dates=common_price_dates,
            valid_common_return_periods=common_return_periods,
            excluded_dates=excluded_dates or [],
            issues=issues,
        ),
        asset_statistics=[],
        correlation_matrix=None,
        covariance_matrix=None,
        calculation_version=CALCULATION_VERSION,
        generated_at=datetime.now(timezone.utc),
        downstream_ready=False,
    )


# ------------------------------------------
# MAIN MARKET ANALYSIS ENGINE
# ------------------------------------------

def analyze_market(
    request: MarketAnalysisRequest,
) -> MarketAnalysisResponse:
    """
    Analyze historical asset prices.

    The input request is validated by Pydantic.

    Price dates are aligned across all assets before
    returns are calculated.

    Returns are calculated between consecutive
    retained common dates. Therefore, dropping
    intermediate dates can create multi-period
    returns. Such gaps are rejected to avoid
    silently treating multi-period returns as
    single-period observations.
    """

    tickers = [
        asset.ticker for asset in request.assets
    ]

    issues = []

    # --------------------------------------
    # 1. COLLECT OBSERVATIONS
    # --------------------------------------

    prices_by_ticker = defaultdict(dict)

    for observation in request.observations:
        prices_by_ticker[
            observation.ticker
        ][observation.observation_date] = (
            observation.price
        )

    date_sets = [
        set(prices_by_ticker[ticker])
        for ticker in tickers
    ]

    common_dates = sorted(
        set.intersection(*date_sets)
    )

    all_dates = sorted(
        set.union(*date_sets)
    )

    excluded_dates = sorted(
        set(all_dates) - set(common_dates)
    )

    # --------------------------------------
    # 2. VALIDATE ALIGNMENT
    # --------------------------------------

    if excluded_dates:
        issues.append(
            make_issue(
                "MISSING_COMMON_DATES",
                Severity.WARNING,
                (
                    f"{len(excluded_dates)} dates are "
                    "not shared by every asset."
                ),
            )
        )

    if (
        request.config.missing_data_policy
        == MissingDataPolicy.STRICT
        and excluded_dates
    ):
        issues.append(
            make_issue(
                "STRICT_ALIGNMENT_FAILED",
                Severity.ERROR,
                "Strict policy requires identical dates.",
            )
        )

        return rejected_result(
            request,
            issues,
            len(common_dates),
            max(0, len(common_dates) - 1),
            excluded_dates,
        )

    # A gap inside the common-date sequence may
    # represent a longer holding period for one
    # or more assets. Without an explicit calendar
    # policy, the engine cannot safely annualize it.
    #
    # Missing dates at the beginning or end of
    # individual asset histories are permitted
    # under COMMON_DATES, but interior unmatched
    # dates are not.

    if len(common_dates) >= 2:
        first = common_dates[0]
        last = common_dates[-1]

        interior_exclusions = [
            day for day in excluded_dates
            if first < day < last
        ]

        if interior_exclusions:
            issues.append(
                make_issue(
                    "INTERIOR_ALIGNMENT_GAP",
                    Severity.ERROR,
                    (
                        "Unmatched dates inside the "
                        "common analysis window could "
                        "produce multi-period returns."
                    ),
                )
            )

            return rejected_result(
                request,
                issues,
                len(common_dates),
                max(0, len(common_dates) - 1),
                excluded_dates,
            )

    return_count = max(
        0, len(common_dates) - 1
    )

    if (
        return_count
        < request.config.minimum_common_observations
    ):
        issues.append(
            make_issue(
                "INSUFFICIENT_HISTORY",
                Severity.ERROR,
                (
                    "Insufficient aligned return "
                    "observations for analysis."
                ),
            )
        )

        return rejected_result(
            request,
            issues,
            len(common_dates),
            return_count,
            excluded_dates,
        )

    # --------------------------------------
    # 3. CALCULATE RETURNS
    # --------------------------------------

    price_series = {}
    return_series = {}

    try:
        for ticker in tickers:
            prices = [
                prices_by_ticker[ticker][day]
                for day in common_dates
            ]

            price_series[ticker] = prices

            return_series[ticker] = (
                calculate_returns(
                    prices,
                    request.config.return_method,
                )
            )

    except (ValueError, OverflowError) as exc:
        issues.append(
            make_issue(
                "RETURN_CALCULATION_FAILED",
                Severity.ERROR,
                str(exc),
            )
        )

        return rejected_result(
            request,
            issues,
            len(common_dates),
            return_count,
            excluded_dates,
        )

    # --------------------------------------
    # 4. ASSET STATISTICS
    # --------------------------------------

    annualization = (
        request.config.annualization_periods
    )

    statistics = []

    try:
        for ticker in tickers:
            returns = return_series[ticker]
            prices = price_series[ticker]

            periodic_mean = mean(returns)

            # For log returns, convert the mean
            # log return to an equivalent simple
            # periodic return before annualizing.
            if (
                request.config.return_method
                == ReturnMethod.LOG
            ):
                annual_return = (
                    exp(periodic_mean) - 1.0
                ) * annualization
            else:
                annual_return = (
                    periodic_mean * annualization
                )

            annual_volatility = (
                stdev(returns)
                * sqrt(annualization)
            )

            cumulative_return = (
                prices[-1] / prices[0] - 1.0
            )

            statistics.append(
                AssetStatistics(
                    ticker=ticker,
                    observation_count=len(returns),
                    mean_periodic_return=periodic_mean,
                    annualized_arithmetic_return=(
                        annual_return
                    ),
                    annualized_volatility=(
                        annual_volatility
                    ),
                    cumulative_return=(
                        cumulative_return
                    ),
                )
            )

    except (
        ValueError,
        OverflowError,
        ArithmeticError,
    ) as exc:
        issues.append(
            make_issue(
                "STATISTICS_FAILED",
                Severity.ERROR,
                str(exc),
            )
        )

        return rejected_result(
            request,
            issues,
            len(common_dates),
            return_count,
            excluded_dates,
        )

    # --------------------------------------
    # 5. COVARIANCE AND CORRELATION
    # --------------------------------------

    covariance_values = []
    correlation_values = []

    try:
        standard_deviations = {
            ticker: stdev(return_series[ticker])
            for ticker in tickers
        }

        zero_variance = [
            ticker
            for ticker in tickers
            if standard_deviations[ticker] == 0
        ]

        if zero_variance:
            issues.append(
                make_issue(
                    "ZERO_VARIANCE",
                    Severity.ERROR,
                    (
                        "Correlation is undefined for "
                        "constant-return assets: "
                        + ", ".join(zero_variance)
                    ),
                )
            )

            return rejected_result(
                request,
                issues,
                len(common_dates),
                return_count,
                excluded_dates,
            )

        for ticker_a in tickers:
            covariance_row = []
            correlation_row = []

            for ticker_b in tickers:
                covariance = sample_covariance(
                    return_series[ticker_a],
                    return_series[ticker_b],
                )

                denominator = (
                    standard_deviations[ticker_a]
                    * standard_deviations[ticker_b]
                )

                correlation = (
                    covariance / denominator
                )

                # Remove insignificant floating
                # point drift beyond +/-1.
                correlation = max(
                    -1.0,
                    min(1.0, correlation),
                )

                covariance_row.append(
                    covariance * annualization
                )

                correlation_row.append(
                    correlation
                )

            covariance_values.append(
                covariance_row
            )

            correlation_values.append(
                correlation_row
            )

        covariance_matrix = MarketMatrix(
            tickers=tickers,
            values=covariance_values,
        )

        correlation_matrix = MarketMatrix(
            tickers=tickers,
            values=correlation_values,
        )

    except (
        ValueError,
        OverflowError,
        ArithmeticError,
    ) as exc:
        issues.append(
            make_issue(
                "MATRIX_CALCULATION_FAILED",
                Severity.ERROR,
                str(exc),
            )
        )

        return rejected_result(
            request,
            issues,
            len(common_dates),
            return_count,
            excluded_dates,
        )

    # --------------------------------------
    # 6. QUALITY AND RESPONSE
    # --------------------------------------

    has_warnings = any(
        issue.severity == Severity.WARNING
        for issue in issues
    )

    quality_status = (
        QualityStatus.WARNING
        if has_warnings
        else QualityStatus.PASS
    )

    analysis_status = (
        "completed_with_warnings"
        if has_warnings
        else "completed"
    )

    return MarketAnalysisResponse(
        analysis_id=request.analysis_id,
        status=analysis_status,
        provenance=request.provenance,
        config=request.config,
        quality=DataQualityReport(
            status=quality_status,
            source_observation_count=len(
                request.observations
            ),
            valid_common_price_dates=len(
                common_dates
            ),
            valid_common_return_periods=(
                return_count
            ),
            excluded_dates=excluded_dates,
            issues=issues,
        ),
        asset_statistics=statistics,
        correlation_matrix=correlation_matrix,
        covariance_matrix=covariance_matrix,
        calculation_version=CALCULATION_VERSION,
        generated_at=datetime.now(timezone.utc),
        downstream_ready=True,
    )
