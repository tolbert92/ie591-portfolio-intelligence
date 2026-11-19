
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.2.1 — Mathematical Validation Tests

Tests the core calculation engine using independently
calculated reference values and invalid-input cases.

No external market data, API keys, or numerical
optimization solver is required.
"""

import math
from copy import deepcopy

import pytest

from backend.optimization_models import OptimizationRequest
from backend.optimization_engine import (
    OptimizationEngineError,
    validate_covariance_matrix,
    validate_weights,
    calculate_expected_return,
    calculate_portfolio_variance,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_portfolio_metrics,
    equal_weight_portfolio,
)


# --------------------------------------------------
# Test fixtures
# --------------------------------------------------

@pytest.fixture
def request_payload():
    return {
        "optimization_id": "IE591-GATE421-TEST",
        "assets": [
            {
                "ticker": "VTI",
                "asset_name": "Synthetic Equity",
                "asset_class": "equity",
                "expected_annual_return": 0.12,
                "minimum_weight": 0.0,
                "maximum_weight": 1.0,
            },
            {
                "ticker": "BND",
                "asset_name": "Synthetic Bonds",
                "asset_class": "fixed_income",
                "expected_annual_return": 0.06,
                "minimum_weight": 0.0,
                "maximum_weight": 1.0,
            },
        ],
        "covariance_matrix": [
            [0.04, 0.006],
            [0.006, 0.01],
        ],
        "constraints": {
            "fully_invested": True,
            "allow_short_selling": False,
            "allow_leverage": False,
        },
        "config": {
            "objective": "minimum_variance",
            "risk_free_annual_rate": 0.02,
        },
        "provenance": {
            "source_type": "synthetic",
            "source_name": "IE 591 Mathematical Test Data",
            "dataset_id": "GATE421-001",
            "upstream_analysis_id": "IE591-GATE34-THREE",
            "is_synthetic": True,
        },
    }


@pytest.fixture
def optimization_request(request_payload):
    return OptimizationRequest.model_validate(
        request_payload
    )


@pytest.fixture
def weights():
    return [0.6, 0.4]


@pytest.fixture
def covariance():
    return [
        [0.04, 0.006],
        [0.006, 0.01],
    ]


# --------------------------------------------------
# Expected return
# --------------------------------------------------

def test_expected_return_reference_calculation(weights):
    # 0.6(0.12) + 0.4(0.06) = 0.096
    result = calculate_expected_return(
        weights, [0.12, 0.06]
    )

    assert result == pytest.approx(0.096)


def test_expected_return_equal_weights():
    result = calculate_expected_return(
        [0.5, 0.5], [0.12, 0.06]
    )

    assert result == pytest.approx(0.09)


def test_expected_return_negative_asset_return():
    result = calculate_expected_return(
        [0.5, 0.5], [0.10, -0.04]
    )

    assert result == pytest.approx(0.03)


def test_expected_return_mismatched_lengths():
    with pytest.raises(OptimizationEngineError):
        calculate_expected_return(
            [0.5, 0.5], [0.12]
        )


def test_expected_return_nonfinite_input():
    with pytest.raises(OptimizationEngineError):
        calculate_expected_return(
            [0.5, 0.5],
            [0.12, float("nan")],
        )


# --------------------------------------------------
# Portfolio variance and volatility
# --------------------------------------------------

def test_variance_reference_calculation(
    weights, covariance
):
    # w'Σw =
    # 0.6²(0.04) + 2(0.6)(0.4)(0.006)
    # + 0.4²(0.01) = 0.01888

    result = calculate_portfolio_variance(
        weights, covariance
    )

    assert result == pytest.approx(0.01888)


def test_volatility_reference_calculation(
    weights, covariance
):
    expected = math.sqrt(0.01888)

    result = calculate_portfolio_volatility(
        weights, covariance
    )

    assert result == pytest.approx(expected)


def test_variance_single_asset():
    result = calculate_portfolio_variance(
        [1.0], [[0.04]]
    )

    assert result == pytest.approx(0.04)


def test_volatility_single_asset():
    result = calculate_portfolio_volatility(
        [1.0], [[0.04]]
    )

    assert result == pytest.approx(0.20)


def test_zero_covariance_matrix():
    result = calculate_portfolio_variance(
        [0.5, 0.5],
        [[0.0, 0.0], [0.0, 0.0]],
    )

    assert result == pytest.approx(0.0)


def test_singular_covariance_matrix():
    # Perfectly correlated assets produce
    # a positive-semidefinite singular matrix.

    matrix = [
        [0.04, 0.02],
        [0.02, 0.01],
    ]

    result = calculate_portfolio_variance(
        [0.5, 0.5], matrix
    )

    assert result == pytest.approx(0.0225)


# --------------------------------------------------
# Covariance validation
# --------------------------------------------------

def test_valid_covariance_matrix(covariance):
    result = validate_covariance_matrix(
        covariance, 2
    )

    assert result == pytest.approx(
        covariance[0], rel=1e-10
    ) if False else result == covariance


def test_covariance_wrong_dimensions():
    with pytest.raises(OptimizationEngineError):
        validate_covariance_matrix(
            [[0.04]], 2
        )


def test_covariance_non_square():
    with pytest.raises(OptimizationEngineError):
        validate_covariance_matrix(
            [[0.04, 0.01], [0.01]], 2
        )


def test_covariance_asymmetric():
    with pytest.raises(OptimizationEngineError):
        validate_covariance_matrix(
            [[0.04, 0.02], [0.01, 0.03]], 2
        )


def test_covariance_negative_diagonal():
    with pytest.raises(OptimizationEngineError):
        validate_covariance_matrix(
            [[-0.04, 0.0], [0.0, 0.01]], 2
        )


def test_covariance_not_positive_semidefinite():
    # Eigenvalues: 0.07 and -0.03.
    # This matrix must be rejected.

    matrix = [
        [0.02, 0.05],
        [0.05, 0.02],
    ]

    with pytest.raises(
        OptimizationEngineError,
        match="positive semidefinite",
    ):
        validate_covariance_matrix(matrix, 2)


def test_covariance_nonfinite_value():
    with pytest.raises(OptimizationEngineError):
        validate_covariance_matrix(
            [
                [0.04, float("inf")],
                [float("inf"), 0.01],
            ],
            2,
        )


def test_covariance_three_asset_diagonal():
    matrix = [
        [0.04, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0025],
    ]

    result = validate_covariance_matrix(
        matrix, 3
    )

    assert len(result) == 3
    assert result[2][2] == pytest.approx(0.0025)


# --------------------------------------------------
# Portfolio weight validation
# --------------------------------------------------

def test_valid_portfolio_weights(
    optimization_request, weights
):
    result = validate_weights(
        weights, optimization_request
    )

    assert result == pytest.approx([0.6, 0.4])


def test_weights_do_not_sum_to_one(
    optimization_request
):
    with pytest.raises(
        OptimizationEngineError,
        match="sum to 1.0",
    ):
        validate_weights(
            [0.5, 0.4], optimization_request
        )


def test_negative_weight_rejected(
    optimization_request
):
    with pytest.raises(OptimizationEngineError):
        validate_weights(
            [-0.1, 1.1], optimization_request
        )


def test_weight_count_mismatch(
    optimization_request
):
    with pytest.raises(OptimizationEngineError):
        validate_weights(
            [1.0], optimization_request
        )


def test_nonfinite_weight_rejected(
    optimization_request
):
    with pytest.raises(OptimizationEngineError):
        validate_weights(
            [float("nan"), 0.5],
            optimization_request,
        )


def test_asset_upper_bound_enforced(request_payload):
    payload = deepcopy(request_payload)
    payload["assets"][0]["maximum_weight"] = 0.5

    request = OptimizationRequest.model_validate(
        payload
    )

    with pytest.raises(
        OptimizationEngineError,
        match="allocation bounds",
    ):
        validate_weights([0.6, 0.4], request)


def test_asset_lower_bound_enforced(request_payload):
    payload = deepcopy(request_payload)
    payload["assets"][0]["minimum_weight"] = 0.7

    request = OptimizationRequest.model_validate(
        payload
    )

    with pytest.raises(
        OptimizationEngineError,
        match="allocation bounds",
    ):
        validate_weights([0.6, 0.4], request)


# --------------------------------------------------
# Sharpe ratio
# --------------------------------------------------

def test_sharpe_ratio_reference():
    result = calculate_sharpe_ratio(
        expected_annual_return=0.10,
        annualized_volatility=0.20,
        risk_free_annual_rate=0.02,
    )

    assert result == pytest.approx(0.4)


def test_sharpe_ratio_zero_volatility():
    result = calculate_sharpe_ratio(
        0.10, 0.0, 0.02
    )

    assert result is None


def test_sharpe_ratio_negative_volatility():
    with pytest.raises(OptimizationEngineError):
        calculate_sharpe_ratio(
            0.10, -0.20, 0.02
        )


def test_sharpe_ratio_negative_result():
    result = calculate_sharpe_ratio(
        0.01, 0.10, 0.03
    )

    assert result == pytest.approx(-0.2)


# --------------------------------------------------
# Standardized portfolio metrics
# --------------------------------------------------

def test_complete_portfolio_metrics(
    optimization_request, weights
):
    result = calculate_portfolio_metrics(
        optimization_request, weights
    )

    assert result.expected_annual_return == pytest.approx(
        0.096
    )

    assert result.annualized_variance == pytest.approx(
        0.01888
    )

    assert result.annualized_volatility == pytest.approx(
        math.sqrt(0.01888)
    )

    expected_sharpe = (
        0.096 - 0.02
    ) / math.sqrt(0.01888)

    assert result.sharpe_ratio == pytest.approx(
        expected_sharpe
    )


def test_maximum_volatility_constraint(request_payload):
    payload = deepcopy(request_payload)

    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.10

    request = OptimizationRequest.model_validate(
        payload
    )

    with pytest.raises(
        OptimizationEngineError,
        match="maximum annual volatility",
    ):
        calculate_portfolio_metrics(
            request, [0.6, 0.4]
        )


def test_minimum_return_constraint(request_payload):
    payload = deepcopy(request_payload)

    payload["constraints"][
        "minimum_expected_annual_return"
    ] = 0.11

    request = OptimizationRequest.model_validate(
        payload
    )

    with pytest.raises(
        OptimizationEngineError,
        match="minimum expected annual return",
    ):
        calculate_portfolio_metrics(
            request, [0.6, 0.4]
        )


def test_feasible_portfolio_constraints(request_payload):
    payload = deepcopy(request_payload)

    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.15

    payload["constraints"][
        "minimum_expected_annual_return"
    ] = 0.09

    request = OptimizationRequest.model_validate(
        payload
    )

    result = calculate_portfolio_metrics(
        request, [0.6, 0.4]
    )

    assert result.expected_annual_return >= 0.09
    assert result.annualized_volatility <= 0.15


# --------------------------------------------------
# Equal-weight baseline
# --------------------------------------------------

def test_equal_weight_baseline(optimization_request):
    result = equal_weight_portfolio(
        optimization_request
    )

    assert result == pytest.approx([0.5, 0.5])


def test_equal_weight_baseline_metrics(
    optimization_request
):
    weights = equal_weight_portfolio(
        optimization_request
    )

    metrics = calculate_portfolio_metrics(
        optimization_request, weights
    )

    assert metrics.expected_annual_return == pytest.approx(
        0.09
    )


def test_equal_weight_infeasible_bounds(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"][0]["minimum_weight"] = 0.6
    payload["assets"][1]["maximum_weight"] = 0.4

    request = OptimizationRequest.model_validate(
        payload
    )

    with pytest.raises(OptimizationEngineError):
        equal_weight_portfolio(request)


# --------------------------------------------------
# Final engine foundation regression
# --------------------------------------------------

def test_engine_does_not_mutate_covariance(covariance):
    original = deepcopy(covariance)

    calculate_portfolio_variance(
        [0.6, 0.4], covariance
    )

    assert covariance == original


def test_repeated_calculation_is_deterministic(
    optimization_request
):
    first = calculate_portfolio_metrics(
        optimization_request, [0.6, 0.4]
    )

    second = calculate_portfolio_metrics(
        optimization_request, [0.6, 0.4]
    )

    assert first.model_dump() == second.model_dump()
