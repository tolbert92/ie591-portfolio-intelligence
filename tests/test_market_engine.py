
"""
IE 591 — Portfolio Intelligence
Skill #2: Market Intelligence

Dedicated analytical validation tests.

Run:
    python -m pytest tests/test_market_engine.py -v

Tests use deterministic synthetic prices.
No network, database, or OpenAI API required.
"""

from datetime import date
from math import exp, log, sqrt
from statistics import covariance, mean, stdev

import pytest
from pydantic import ValidationError

from backend.market_engine import (
    analyze_market,
    calculate_returns,
    sample_covariance,
)
from backend.market_models import (
    MarketAnalysisRequest,
    MarketAnalysisConfig,
    MissingDataPolicy,
    QualityStatus,
    ReturnMethod,
)


# ------------------------------------------
# SYNTHETIC TEST DATA
# ------------------------------------------

DATES = [
    date(2025, 1, 31),
    date(2025, 2, 28),
    date(2025, 3, 31),
    date(2025, 4, 30),
    date(2025, 5, 31),
]

PRICES = {
    "VTI": [100, 110, 105, 115, 120],
    "BND": [100, 102, 101, 104, 103],
    "SGOV": [100, 101, 102, 103, 105],
}


def make_request(
    prices=None,
    dates=None,
    config=None,
):
    """Create a valid synthetic market request."""
    prices = prices if prices is not None else PRICES
    dates = dates if dates is not None else DATES

    observations = []

    for ticker, values in prices.items():
        for day, price in zip(dates, values):
            observations.append(
                {
                    "observation_date": day,
                    "ticker": ticker,
                    "price": price,
                }
            )

    return MarketAnalysisRequest.model_validate(
        {
            "analysis_id": "IE591-SYNTHETIC-001",
            "assets": [
                {
                    "ticker": ticker,
                    "asset_name": (
                        f"Synthetic {ticker} Asset"
                    ),
                    "asset_class": "test_asset",
                    "currency": "USD",
                }
                for ticker in prices
            ],
            "observations": observations,
            "provenance": {
                "source_type": "synthetic",
                "source_name": "IE 591 Test Dataset",
                "dataset_id": "TEST-001",
                "is_synthetic": True,
            },
            "config": config or {
                "frequency": "monthly",
                "annualization_periods": 12,
                "minimum_common_observations": 3,
                "return_method": "simple",
                "missing_data_policy": "common_dates",
            },
        }
    )


def statistic_for(result, ticker):
    return next(
        item
        for item in result.asset_statistics
        if item.ticker == ticker
    )


# ------------------------------------------
# TEST 1: SIMPLE RETURNS
# ------------------------------------------

def test_simple_return_calculation():
    result = calculate_returns(
        [100, 110, 105],
        ReturnMethod.SIMPLE,
    )

    assert result[0] == pytest.approx(0.10)
    assert result[1] == pytest.approx(
        105 / 110 - 1
    )


# ------------------------------------------
# TEST 2: LOG RETURNS
# ------------------------------------------

def test_log_return_calculation():
    result = calculate_returns(
        [100, 110, 105],
        ReturnMethod.LOG,
    )

    assert result[0] == pytest.approx(
        log(110 / 100)
    )
    assert result[1] == pytest.approx(
        log(105 / 110)
    )


# ------------------------------------------
# TEST 3: SAMPLE COVARIANCE
# ------------------------------------------

def test_sample_covariance():
    x = [0.01, 0.03, -0.02, 0.04]
    y = [0.02, -0.01, 0.01, 0.05]

    actual = sample_covariance(x, y)

    assert actual == pytest.approx(
        covariance(x, y)
    )


# ------------------------------------------
# TEST 4: COMPLETE ANALYSIS
# ------------------------------------------

def test_complete_market_analysis():
    result = analyze_market(make_request())

    assert result.status == "completed"
    assert result.downstream_ready is True
    assert result.quality.status == QualityStatus.PASS

    assert len(result.asset_statistics) == 3
    assert result.correlation_matrix is not None
    assert result.covariance_matrix is not None


# ------------------------------------------
# TEST 5: OBSERVATION ACCOUNTING
# ------------------------------------------

def test_observation_counts():
    result = analyze_market(make_request())

    assert (
        result.quality.source_observation_count
        == 15
    )
    assert (
        result.quality.valid_common_price_dates
        == 5
    )
    assert (
        result.quality.valid_common_return_periods
        == 4
    )

    for item in result.asset_statistics:
        assert item.observation_count == 4


# ------------------------------------------
# TEST 6: ANNUALIZED RETURN
# ------------------------------------------

def test_annualized_arithmetic_return():
    result = analyze_market(make_request())

    vti_returns = [
        PRICES["VTI"][i] / PRICES["VTI"][i - 1]
        - 1
        for i in range(1, 5)
    ]

    expected = mean(vti_returns) * 12

    actual = statistic_for(
        result, "VTI"
    ).annualized_arithmetic_return

    assert actual == pytest.approx(expected)


# ------------------------------------------
# TEST 7: ANNUALIZED VOLATILITY
# ------------------------------------------

def test_annualized_volatility():
    result = analyze_market(make_request())

    vti_returns = [
        PRICES["VTI"][i] / PRICES["VTI"][i - 1]
        - 1
        for i in range(1, 5)
    ]

    expected = stdev(vti_returns) * sqrt(12)

    actual = statistic_for(
        result, "VTI"
    ).annualized_volatility

    assert actual == pytest.approx(expected)


# ------------------------------------------
# TEST 8: CUMULATIVE RETURN
# ------------------------------------------

def test_cumulative_return():
    result = analyze_market(make_request())

    actual = statistic_for(
        result, "VTI"
    ).cumulative_return

    assert actual == pytest.approx(0.20)


# ------------------------------------------
# TEST 9: MATRIX DIMENSIONS
# ------------------------------------------

def test_matrix_dimensions():
    result = analyze_market(make_request())

    for matrix in [
        result.covariance_matrix,
        result.correlation_matrix,
    ]:
        assert matrix is not None
        assert matrix.tickers == [
            "VTI", "BND", "SGOV"
        ]
        assert len(matrix.values) == 3

        for row in matrix.values:
            assert len(row) == 3


# ------------------------------------------
# TEST 10: CORRELATION PROPERTIES
# ------------------------------------------

def test_correlation_properties():
    result = analyze_market(make_request())

    matrix = result.correlation_matrix.values

    for i in range(3):
        assert matrix[i][i] == pytest.approx(1.0)

        for j in range(3):
            assert matrix[i][j] == pytest.approx(
                matrix[j][i]
            )
            assert -1 <= matrix[i][j] <= 1


# ------------------------------------------
# TEST 11: ANNUALIZED COVARIANCE
# ------------------------------------------

def test_annualized_covariance():
    result = analyze_market(make_request())

    vti = calculate_returns(
        PRICES["VTI"],
        ReturnMethod.SIMPLE,
    )
    bnd = calculate_returns(
        PRICES["BND"],
        ReturnMethod.SIMPLE,
    )

    expected = covariance(vti, bnd) * 12

    actual = (
        result.covariance_matrix.values[0][1]
    )

    assert actual == pytest.approx(expected)


# ------------------------------------------
# TEST 12: STRICT MISSING DATA
# ------------------------------------------

def test_strict_missing_data_rejected():
    request = make_request(
        config={
            "frequency": "monthly",
            "annualization_periods": 12,
            "minimum_common_observations": 3,
            "missing_data_policy": "strict",
        }
    )

    request.observations = [
        observation
        for observation in request.observations
        if not (
            observation.ticker == "BND"
            and observation.observation_date
            == DATES[0]
        )
    ]

    result = analyze_market(request)

    assert result.status == "rejected"
    assert result.downstream_ready is False
    assert result.quality.status == QualityStatus.FAIL


# ------------------------------------------
# TEST 13: INTERIOR ALIGNMENT GAP
# ------------------------------------------

def test_interior_alignment_gap_rejected():
    request = make_request()

    request.observations = [
        observation
        for observation in request.observations
        if not (
            observation.ticker == "BND"
            and observation.observation_date
            == DATES[2]
        )
    ]

    result = analyze_market(request)

    assert result.status == "rejected"
    assert result.downstream_ready is False

    codes = [
        issue.code
        for issue in result.quality.issues
    ]

    assert "INTERIOR_ALIGNMENT_GAP" in codes


# ------------------------------------------
# TEST 14: INSUFFICIENT HISTORY
# ------------------------------------------

def test_insufficient_history_rejected():
    short_prices = {
        ticker: values[:3]
        for ticker, values in PRICES.items()
    }

    request = make_request(
        prices=short_prices,
        dates=DATES[:3],
    )

    result = analyze_market(request)

    assert result.status == "rejected"
    assert result.downstream_ready is False

    codes = [
        issue.code
        for issue in result.quality.issues
    ]

    assert "INSUFFICIENT_HISTORY" in codes


# ------------------------------------------
# TEST 15: ZERO VARIANCE
# ------------------------------------------

def test_zero_variance_rejected():
    prices = dict(PRICES)

    # Constant prices imply zero periodic returns.
    prices["SGOV"] = [100] * 5

    result = analyze_market(
        make_request(prices=prices)
    )

    assert result.status == "rejected"
    assert result.downstream_ready is False

    codes = [
        issue.code
        for issue in result.quality.issues
    ]

    assert "ZERO_VARIANCE" in codes


# ------------------------------------------
# TEST 16: DUPLICATE OBSERVATION
# ------------------------------------------

def test_duplicate_observation_rejected():
    request = make_request()

    raw = request.model_dump()

    raw["observations"].append(
        raw["observations"][0].copy()
    )

    with pytest.raises(ValidationError):
        MarketAnalysisRequest.model_validate(raw)


# ------------------------------------------
# TEST 17: UNDEFINED ASSET
# ------------------------------------------

def test_undefined_asset_rejected():
    request = make_request()

    raw = request.model_dump()

    raw["observations"][0]["ticker"] = "XYZ"

    with pytest.raises(ValidationError):
        MarketAnalysisRequest.model_validate(raw)


# ------------------------------------------
# TEST 18: INVALID PRICE
# ------------------------------------------

def test_invalid_price_rejected():
    request = make_request()

    raw = request.model_dump()

    raw["observations"][0]["price"] = -10

    with pytest.raises(ValidationError):
        MarketAnalysisRequest.model_validate(raw)


# ------------------------------------------
# TEST 19: INVALID ANNUALIZATION
# ------------------------------------------

def test_invalid_annualization_rejected():
    request = make_request()

    raw = request.model_dump()

    raw["config"]["annualization_periods"] = 252

    with pytest.raises(ValidationError):
        MarketAnalysisRequest.model_validate(raw)


# ------------------------------------------
# TEST 20: SYNTHETIC PROVENANCE
# ------------------------------------------

def test_synthetic_provenance():
    result = analyze_market(make_request())

    assert result.provenance.is_synthetic is True
    assert (
        result.provenance.source_type.value
        == "synthetic"
    )
    assert result.provenance.dataset_id == "TEST-001"


# ------------------------------------------
# TEST 21: LOG-RETURN ENGINE PATH
# ------------------------------------------

def test_log_return_engine():
    request = make_request(
        config={
            "frequency": "monthly",
            "annualization_periods": 12,
            "minimum_common_observations": 3,
            "return_method": "log",
        }
    )

    result = analyze_market(request)

    assert result.status == "completed"

    vti_log_returns = calculate_returns(
        PRICES["VTI"],
        ReturnMethod.LOG,
    )

    expected = (
        exp(mean(vti_log_returns)) - 1
    ) * 12

    actual = statistic_for(
        result, "VTI"
    ).annualized_arithmetic_return

    assert actual == pytest.approx(expected)


# ------------------------------------------
# TEST 22: REJECTION HAS NO MATRICES
# ------------------------------------------

def test_rejected_results_have_no_matrices():
    prices = dict(PRICES)
    prices["SGOV"] = [100] * 5

    result = analyze_market(
        make_request(prices=prices)
    )

    assert result.correlation_matrix is None
    assert result.covariance_matrix is None
    assert result.asset_statistics == []


# ------------------------------------------
# TEST 23: END-OF-SERIES MISSING DATE
# ------------------------------------------

def test_boundary_missing_date_warning():
    request = make_request()

    request.observations = [
        observation
        for observation in request.observations
        if not (
            observation.ticker == "BND"
            and observation.observation_date
            == DATES[-1]
        )
    ]

    result = analyze_market(request)

    assert result.status == (
        "completed_with_warnings"
    )
    assert result.downstream_ready is True
    assert result.quality.status == (
        QualityStatus.WARNING
    )
    assert (
        result.quality.valid_common_return_periods
        == 3
    )
