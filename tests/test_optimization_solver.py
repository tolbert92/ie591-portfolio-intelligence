
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.2.2 — Constrained Solver Validation

Tests:
- Minimum-variance optimization
- Maximum-Sharpe optimization
- Independently calculated reference solutions
- Asset allocation bounds
- Minimum return constraints
- Maximum volatility constraints
- Feasible and infeasible portfolios
- Covariance validation
- Deterministic execution
- Solver diagnostics

Uses synthetic educational data only.
No external APIs or OpenAI credentials required.
"""

import math
from copy import deepcopy

import pytest

from backend.optimization_models import (
    OptimizationObjective,
    OptimizationRequest,
)

from backend.optimization_solver import (
    OptimizationSolverError,
    solve_portfolio,
)


# --------------------------------------------------
# Synthetic test data
# --------------------------------------------------

@pytest.fixture
def request_payload():
    """
    Two uncorrelated synthetic assets.

    Expected returns:
        Asset A: 12%
        Asset B: 6%

    Annual variances:
        Asset A: 0.04
        Asset B: 0.01

    Risk-free rate: 2%

    Analytical minimum-variance weights:
        A = 0.20
        B = 0.80

    Analytical maximum-Sharpe weights:
        A = 5/13
        B = 8/13
    """
    return {
        "optimization_id": "IE591-GATE422-TEST",
        "assets": [
            {
                "ticker": "ASSET_A",
                "asset_name": "Synthetic Equity",
                "asset_class": "equity",
                "expected_annual_return": 0.12,
                "minimum_weight": 0.0,
                "maximum_weight": 1.0,
            },
            {
                "ticker": "ASSET_B",
                "asset_name": "Synthetic Bonds",
                "asset_class": "fixed_income",
                "expected_annual_return": 0.06,
                "minimum_weight": 0.0,
                "maximum_weight": 1.0,
            },
        ],
        "covariance_matrix": [
            [0.04, 0.0],
            [0.0, 0.01],
        ],
        "constraints": {
            "fully_invested": True,
            "allow_short_selling": False,
            "allow_leverage": False,
        },
        "config": {
            "objective": "minimum_variance",
            "risk_free_annual_rate": 0.02,
            "solver_tolerance": 1e-10,
            "maximum_iterations": 2000,
        },
        "provenance": {
            "source_type": "synthetic",
            "source_name": "IE 591 Solver Validation",
            "dataset_id": "GATE422-001",
            "upstream_analysis_id": "IE591-GATE34-THREE",
            "is_synthetic": True,
        },
    }


@pytest.fixture
def request(request_payload):
    return OptimizationRequest.model_validate(
        request_payload
    )


def build_request(payload):
    return OptimizationRequest.model_validate(
        payload
    )


# --------------------------------------------------
# Minimum-variance optimization
# --------------------------------------------------

def test_minimum_variance_solver_succeeds(request):
    result = solve_portfolio(request)

    assert result.success is True
    assert result.objective == (
        OptimizationObjective.MINIMUM_VARIANCE
    )


def test_minimum_variance_analytical_weights(request):
    """
    For two uncorrelated assets:

    w_A = variance_B / (variance_A + variance_B)

    w_A = 0.01 / 0.05 = 0.20
    w_B = 0.80
    """
    result = solve_portfolio(request)

    assert result.weights[0] == pytest.approx(
        0.20, abs=1e-4
    )
    assert result.weights[1] == pytest.approx(
        0.80, abs=1e-4
    )


def test_minimum_variance_reference_metrics(request):
    result = solve_portfolio(request)

    # Expected return:
    # 0.2(0.12) + 0.8(0.06) = 0.072

    assert (
        result.metrics.expected_annual_return
        == pytest.approx(0.072, abs=1e-5)
    )

    # Variance:
    # 0.2^2(0.04) + 0.8^2(0.01) = 0.008

    assert (
        result.metrics.annualized_variance
        == pytest.approx(0.008, abs=1e-5)
    )

    assert (
        result.metrics.annualized_volatility
        == pytest.approx(math.sqrt(0.008), abs=1e-5)
    )


def test_minimum_variance_beats_equal_weight(request):
    result = solve_portfolio(request)

    equal_weight_variance = (
        0.5**2 * 0.04
        + 0.5**2 * 0.01
    )

    assert (
        result.metrics.annualized_variance
        < equal_weight_variance
    )


# --------------------------------------------------
# Maximum-Sharpe optimization
# --------------------------------------------------

def test_maximum_sharpe_solver_succeeds(request):
    result = solve_portfolio(
        request,
        objective="maximum_sharpe",
    )

    assert result.success is True
    assert result.objective == (
        OptimizationObjective.MAXIMUM_SHARPE
    )


def test_maximum_sharpe_analytical_weights(request):
    """
    For uncorrelated assets, the tangency portfolio
    weights are proportional to:

        (expected_return - risk_free_rate)
        / asset_variance

    Asset A:
        (0.12 - 0.02) / 0.04 = 2.5

    Asset B:
        (0.06 - 0.02) / 0.01 = 4.0

    Normalized:
        A = 2.5 / 6.5 = 5/13
        B = 4.0 / 6.5 = 8/13
    """
    result = solve_portfolio(
        request,
        objective="maximum_sharpe",
    )

    assert result.weights[0] == pytest.approx(
        5 / 13, abs=1e-4
    )
    assert result.weights[1] == pytest.approx(
        8 / 13, abs=1e-4
    )


def test_maximum_sharpe_beats_equal_weight(request):
    result = solve_portfolio(
        request,
        objective="maximum_sharpe",
    )

    equal_return = 0.09

    equal_volatility = math.sqrt(
        0.5**2 * 0.04
        + 0.5**2 * 0.01
    )

    equal_sharpe = (
        equal_return - 0.02
    ) / equal_volatility

    assert result.metrics.sharpe_ratio is not None

    assert (
        result.metrics.sharpe_ratio
        > equal_sharpe
    )


def test_configured_maximum_sharpe(request_payload):
    payload = deepcopy(request_payload)

    payload["config"]["objective"] = "maximum_sharpe"

    result = solve_portfolio(
        build_request(payload)
    )

    assert result.objective == (
        OptimizationObjective.MAXIMUM_SHARPE
    )


# --------------------------------------------------
# Allocation constraints
# --------------------------------------------------

def test_weights_sum_to_one(request):
    result = solve_portfolio(request)

    assert sum(result.weights) == pytest.approx(
        1.0, abs=1e-8
    )


def test_no_negative_weights(request):
    result = solve_portfolio(request)

    assert all(
        weight >= 0
        for weight in result.weights
    )


def test_maximum_asset_weight_enforced(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"][1]["maximum_weight"] = 0.6

    result = solve_portfolio(
        build_request(payload)
    )

    assert result.weights[1] <= 0.600001
    assert result.weights[0] >= 0.399999


def test_minimum_asset_weight_enforced(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"][0]["minimum_weight"] = 0.4

    result = solve_portfolio(
        build_request(payload)
    )

    assert result.weights[0] >= 0.399999


def test_fixed_asset_allocation(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"][0]["minimum_weight"] = 0.4
    payload["assets"][0]["maximum_weight"] = 0.4

    payload["assets"][1]["minimum_weight"] = 0.6
    payload["assets"][1]["maximum_weight"] = 0.6

    result = solve_portfolio(
        build_request(payload)
    )

    assert result.weights == pytest.approx(
        (0.4, 0.6), abs=1e-6
    )


# --------------------------------------------------
# Portfolio-level constraints
# --------------------------------------------------

def test_minimum_expected_return_constraint(
    request_payload
):
    payload = deepcopy(request_payload)

    payload["constraints"][
        "minimum_expected_annual_return"
    ] = 0.10

    result = solve_portfolio(
        build_request(payload)
    )

    assert (
        result.metrics.expected_annual_return
        >= 0.10 - 1e-6
    )

    # 0.06 + 0.06*w_A >= 0.10
    # Therefore w_A >= 2/3.

    assert result.weights[0] >= (
        2 / 3 - 1e-5
    )


def test_maximum_volatility_constraint(
    request_payload
):
    payload = deepcopy(request_payload)

    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.10

    result = solve_portfolio(
        build_request(payload)
    )

    assert (
        result.metrics.annualized_volatility
        <= 0.10 + 1e-6
    )


def test_feasible_combined_constraints(
    request_payload
):
    payload = deepcopy(request_payload)

    payload["constraints"][
        "minimum_expected_annual_return"
    ] = 0.08

    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.12

    result = solve_portfolio(
        build_request(payload)
    )

    assert (
        result.metrics.expected_annual_return
        >= 0.08 - 1e-6
    )

    assert (
        result.metrics.annualized_volatility
        <= 0.12 + 1e-6
    )


# --------------------------------------------------
# Infeasible optimization
# --------------------------------------------------

def test_impossible_return_constraint_rejected(
    request_payload
):
    payload = deepcopy(request_payload)

    # No long-only allocation can exceed
    # the maximum individual expected return
    # of 12% in this example.

    payload["constraints"][
        "minimum_expected_annual_return"
    ] = 0.20

    with pytest.raises(OptimizationSolverError):
        solve_portfolio(
            build_request(payload)
        )


def test_impossible_volatility_constraint_rejected(
    request_payload
):
    payload = deepcopy(request_payload)

    # Analytical minimum volatility is
    # sqrt(0.008), approximately 8.94%.

    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.02

    with pytest.raises(OptimizationSolverError):
        solve_portfolio(
            build_request(payload)
        )


def test_invalid_covariance_rejected(
    request_payload
):
    payload = deepcopy(request_payload)

    # Symmetric with positive diagonal,
    # but not positive semidefinite.

    payload["covariance_matrix"] = [
        [0.01, 0.05],
        [0.05, 0.01],
    ]

    with pytest.raises(
        OptimizationSolverError,
        match="positive semidefinite",
    ):
        solve_portfolio(
            build_request(payload)
        )


def test_unsupported_objective_rejected(request):
    with pytest.raises(
        OptimizationSolverError,
        match="Unsupported optimization objective",
    ):
        solve_portfolio(
            request,
            objective="maximize_dividends",
        )


# --------------------------------------------------
# Solver diagnostics
# --------------------------------------------------

def test_solver_reports_iterations(request):
    result = solve_portfolio(request)

    assert result.iterations >= 0
    assert result.solver_message


def test_solver_reports_constraint_violation(request):
    result = solve_portfolio(request)

    assert (
        result.maximum_constraint_violation
        <= 1e-6
    )


def test_solver_uses_multiple_starting_points(request):
    result = solve_portfolio(request)

    assert result.starting_points_attempted >= 2


def test_solver_objective_value_is_finite(request):
    result = solve_portfolio(request)

    assert math.isfinite(
        result.objective_value
    )


def test_solver_deterministic(request):
    first = solve_portfolio(request)
    second = solve_portfolio(request)

    assert first.weights == pytest.approx(
        second.weights,
        abs=1e-8,
    )

    assert (
        first.metrics.annualized_variance
        == pytest.approx(
            second.metrics.annualized_variance,
            abs=1e-10,
        )
    )


def test_solver_does_not_mutate_request(request):
    original = request.model_dump(mode="json")

    solve_portfolio(request)

    assert request.model_dump(mode="json") == original


# --------------------------------------------------
# Three-asset portfolio
# --------------------------------------------------

def test_three_asset_minimum_variance(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"].append({
        "ticker": "ASSET_C",
        "asset_name": "Synthetic Cash Equivalent",
        "asset_class": "cash_equivalent",
        "expected_annual_return": 0.03,
        "minimum_weight": 0.0,
        "maximum_weight": 1.0,
    })

    payload["covariance_matrix"] = [
        [0.04, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0025],
    ]

    result = solve_portfolio(
        build_request(payload)
    )

    # For diagonal covariance, minimum-variance
    # weights are proportional to inverse variance.
    #
    # 1/0.04 = 25
    # 1/0.01 = 100
    # 1/0.0025 = 400
    #
    # Total = 525

    expected_weights = (
        25 / 525,
        100 / 525,
        400 / 525,
    )

    assert result.weights == pytest.approx(
        expected_weights,
        abs=1e-4,
    )


def test_three_asset_maximum_sharpe(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"].append({
        "ticker": "ASSET_C",
        "asset_name": "Synthetic Cash Equivalent",
        "asset_class": "cash_equivalent",
        "expected_annual_return": 0.03,
        "minimum_weight": 0.0,
        "maximum_weight": 1.0,
    })

    payload["covariance_matrix"] = [
        [0.04, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0025],
    ]

    result = solve_portfolio(
        build_request(payload),
        objective="maximum_sharpe",
    )

    # Excess return / variance:
    # A = 0.10 / 0.04 = 2.5
    # B = 0.04 / 0.01 = 4.0
    # C = 0.01 / 0.0025 = 4.0

    expected_weights = (
        2.5 / 10.5,
        4.0 / 10.5,
        4.0 / 10.5,
    )

    assert result.weights == pytest.approx(
        expected_weights,
        abs=1e-4,
    )
