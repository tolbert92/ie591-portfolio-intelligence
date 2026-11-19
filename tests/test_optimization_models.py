
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.1 — Data Contract Validation Tests

Tests the optimization input and output schemas.
No optimization engine is required for these tests.
"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from backend.optimization_models import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationObjective,
    OptimizationStatus,
)


def valid_request():
    """Create a valid two-asset optimization request."""
    return {
        "optimization_id": "IE591-GATE41-TEST",
        "assets": [
            {
                "ticker": "VTI",
                "asset_name": "Synthetic Equity",
                "asset_class": "equity",
                "expected_annual_return": 0.1596,
                "minimum_weight": 0.0,
                "maximum_weight": 1.0,
            },
            {
                "ticker": "BND",
                "asset_name": "Synthetic Bonds",
                "asset_class": "fixed_income",
                "expected_annual_return": 0.0998,
                "minimum_weight": 0.0,
                "maximum_weight": 1.0,
            },
        ],
        "covariance_matrix": [
            [0.00508643, 0.00319212],
            [0.00319212, 0.00200330],
        ],
        "constraints": {
            "fully_invested": True,
            "allow_short_selling": False,
            "allow_leverage": False,
        },
        "config": {
            "objective": "minimum_variance",
            "risk_free_annual_rate": 0.02,
            "efficient_frontier_points": 25,
        },
        "provenance": {
            "source_type": "synthetic",
            "source_name": "IE 591 Validation Dataset",
            "dataset_id": "GATE41-001",
            "upstream_analysis_id": "IE591-GATE34-THREE",
            "is_synthetic": True,
        },
    }


def valid_response():
    """Create a valid completed optimization response."""
    request = OptimizationRequest.model_validate(
        valid_request()
    )

    return {
        "optimization_id": request.optimization_id,
        "status": "completed",
        "objective": "minimum_variance",
        "provenance": request.provenance.model_dump(),
        "allocations": [
            {"ticker": "VTI", "weight": 0.4},
            {"ticker": "BND", "weight": 0.6},
        ],
        "portfolio_metrics": {
            "expected_annual_return": 0.12372,
            "annualized_volatility": 0.05,
            "annualized_variance": 0.0025,
            "sharpe_ratio": 2.0744,
        },
        "efficient_frontier": [],
        "quality": {
            "status": "pass",
            "constraints_satisfied": True,
            "weights_sum_to_one": True,
            "covariance_matrix_valid": True,
            "issues": [],
        },
        "downstream_ready": True,
    }


def rejects_request(payload):
    with pytest.raises(ValidationError):
        OptimizationRequest.model_validate(payload)


def rejects_response(payload):
    with pytest.raises(ValidationError):
        OptimizationResponse.model_validate(payload)


# ---------------------------------------------
# Valid request tests
# ---------------------------------------------

def test_valid_optimization_request():
    request = OptimizationRequest.model_validate(
        valid_request()
    )

    assert request.optimization_id == "IE591-GATE41-TEST"
    assert len(request.assets) == 2
    assert request.config.objective == (
        OptimizationObjective.MINIMUM_VARIANCE
    )


def test_default_configuration():
    payload = valid_request()
    payload.pop("config")
    payload.pop("constraints")

    request = OptimizationRequest.model_validate(payload)

    assert request.config.efficient_frontier_points == 25
    assert request.constraints.fully_invested is True
    assert request.constraints.allow_short_selling is False
    assert request.constraints.allow_leverage is False


def test_maximum_sharpe_objective():
    payload = valid_request()
    payload["config"]["objective"] = "maximum_sharpe"

    request = OptimizationRequest.model_validate(payload)

    assert request.config.objective == (
        OptimizationObjective.MAXIMUM_SHARPE
    )


def test_three_asset_request():
    payload = valid_request()

    payload["assets"].append({
        "ticker": "SGOV",
        "asset_name": "Synthetic Treasury",
        "asset_class": "cash_equivalent",
        "expected_annual_return": 0.0259,
    })

    payload["covariance_matrix"] = [
        [0.00508643, 0.00319212, -0.00018774],
        [0.00319212, 0.00200330, -0.00011784],
        [-0.00018774, -0.00011784, 0.000007],
    ]

    request = OptimizationRequest.model_validate(payload)

    assert len(request.assets) == 3
    assert len(request.covariance_matrix) == 3


# ---------------------------------------------
# Asset validation tests
# ---------------------------------------------

def test_duplicate_tickers_rejected():
    payload = valid_request()
    payload["assets"][1]["ticker"] = "VTI"

    rejects_request(payload)


def test_single_asset_rejected():
    payload = valid_request()
    payload["assets"] = payload["assets"][:1]
    payload["covariance_matrix"] = [[0.005]]

    rejects_request(payload)


def test_negative_minimum_weight_rejected():
    payload = valid_request()
    payload["assets"][0]["minimum_weight"] = -0.1

    rejects_request(payload)


def test_weight_above_one_rejected():
    payload = valid_request()
    payload["assets"][0]["maximum_weight"] = 1.2

    rejects_request(payload)


def test_minimum_exceeds_maximum_rejected():
    payload = valid_request()
    payload["assets"][0]["minimum_weight"] = 0.8
    payload["assets"][0]["maximum_weight"] = 0.2

    rejects_request(payload)


def test_infeasible_minimum_weights_rejected():
    payload = valid_request()
    payload["assets"][0]["minimum_weight"] = 0.7
    payload["assets"][1]["minimum_weight"] = 0.7

    rejects_request(payload)


def test_infeasible_maximum_weights_rejected():
    payload = valid_request()
    payload["assets"][0]["maximum_weight"] = 0.3
    payload["assets"][1]["maximum_weight"] = 0.3

    rejects_request(payload)


def test_valid_bounded_weights():
    payload = valid_request()
    payload["assets"][0]["minimum_weight"] = 0.2
    payload["assets"][0]["maximum_weight"] = 0.6
    payload["assets"][1]["minimum_weight"] = 0.4
    payload["assets"][1]["maximum_weight"] = 0.8

    request = OptimizationRequest.model_validate(payload)

    assert request.assets[0].minimum_weight == 0.2
    assert request.assets[1].maximum_weight == 0.8


# ---------------------------------------------
# Covariance validation tests
# ---------------------------------------------

def test_wrong_matrix_row_count_rejected():
    payload = valid_request()
    payload["covariance_matrix"] = [[0.005, 0.003]]

    rejects_request(payload)


def test_non_square_matrix_rejected():
    payload = valid_request()
    payload["covariance_matrix"] = [
        [0.005, 0.003],
        [0.003],
    ]

    rejects_request(payload)


def test_asymmetric_matrix_rejected():
    payload = valid_request()
    payload["covariance_matrix"] = [
        [0.005, 0.003],
        [0.001, 0.002],
    ]

    rejects_request(payload)


def test_negative_variance_rejected():
    payload = valid_request()
    payload["covariance_matrix"][0][0] = -0.005

    rejects_request(payload)


def test_nonfinite_covariance_rejected():
    payload = valid_request()
    payload["covariance_matrix"][0][0] = float("nan")

    rejects_request(payload)


def test_infinite_expected_return_rejected():
    payload = valid_request()
    payload["assets"][0]["expected_annual_return"] = float(
        "inf"
    )

    rejects_request(payload)


# ---------------------------------------------
# Configuration and constraints
# ---------------------------------------------

def test_short_selling_rejected():
    payload = valid_request()
    payload["constraints"]["allow_short_selling"] = True

    rejects_request(payload)


def test_leverage_rejected():
    payload = valid_request()
    payload["constraints"]["allow_leverage"] = True

    rejects_request(payload)


def test_partially_invested_rejected():
    payload = valid_request()
    payload["constraints"]["fully_invested"] = False

    rejects_request(payload)


def test_invalid_objective_rejected():
    payload = valid_request()
    payload["config"]["objective"] = "maximum_return"

    rejects_request(payload)


def test_invalid_frontier_point_count_rejected():
    payload = valid_request()
    payload["config"]["efficient_frontier_points"] = 1

    rejects_request(payload)


def test_invalid_solver_tolerance_rejected():
    payload = valid_request()
    payload["config"]["solver_tolerance"] = 0

    rejects_request(payload)


def test_invalid_iteration_limit_rejected():
    payload = valid_request()
    payload["config"]["maximum_iterations"] = 10

    rejects_request(payload)


def test_optional_risk_constraints():
    payload = valid_request()
    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.15
    payload["constraints"][
        "minimum_expected_annual_return"
    ] = 0.08

    request = OptimizationRequest.model_validate(payload)

    assert (
        request.constraints.maximum_annual_volatility
        == 0.15
    )


# ---------------------------------------------
# Provenance and strict schema validation
# ---------------------------------------------

def test_synthetic_provenance():
    request = OptimizationRequest.model_validate(
        valid_request()
    )

    assert request.provenance.is_synthetic is True
    assert request.provenance.source_type == "synthetic"
    assert (
        request.provenance.upstream_analysis_id
        == "IE591-GATE34-THREE"
    )


def test_inconsistent_synthetic_provenance_rejected():
    payload = valid_request()
    payload["provenance"]["is_synthetic"] = False

    rejects_request(payload)


def test_unexpected_field_rejected():
    payload = valid_request()
    payload["unknown_field"] = "unexpected"

    rejects_request(payload)


def test_missing_provenance_rejected():
    payload = valid_request()
    payload.pop("provenance")

    rejects_request(payload)


# ---------------------------------------------
# Response validation tests
# ---------------------------------------------

def test_valid_completed_response():
    response = OptimizationResponse.model_validate(
        valid_response()
    )

    assert response.status == OptimizationStatus.COMPLETED
    assert response.downstream_ready is True
    assert len(response.allocations) == 2
    assert response.portfolio_metrics is not None


def test_response_weights_sum_to_one():
    response = OptimizationResponse.model_validate(
        valid_response()
    )

    total = sum(
        allocation.weight
        for allocation in response.allocations
    )

    assert total == pytest.approx(1.0)


def test_downstream_invalid_weights_rejected():
    payload = valid_response()
    payload["allocations"][0]["weight"] = 0.3

    rejects_response(payload)


def test_downstream_missing_metrics_rejected():
    payload = valid_response()
    payload["portfolio_metrics"] = None

    rejects_response(payload)


def test_downstream_failed_quality_rejected():
    payload = valid_response()
    payload["quality"]["status"] = "fail"

    rejects_response(payload)


def test_downstream_unsatisfied_constraints_rejected():
    payload = valid_response()
    payload["quality"]["constraints_satisfied"] = False

    rejects_response(payload)


def test_downstream_invalid_covariance_rejected():
    payload = valid_response()
    payload["quality"]["covariance_matrix_valid"] = False

    rejects_response(payload)


def test_rejected_response_cannot_be_downstream_ready():
    payload = valid_response()
    payload["status"] = "rejected"

    rejects_response(payload)


def test_rejected_response_cannot_contain_allocations():
    payload = valid_response()
    payload["status"] = "rejected"
    payload["downstream_ready"] = False

    rejects_response(payload)


def test_valid_rejected_response():
    payload = deepcopy(valid_response())

    payload["status"] = "rejected"
    payload["allocations"] = []
    payload["portfolio_metrics"] = None
    payload["efficient_frontier"] = []
    payload["downstream_ready"] = False

    payload["quality"]["status"] = "fail"
    payload["quality"]["constraints_satisfied"] = False
    payload["quality"]["weights_sum_to_one"] = False

    response = OptimizationResponse.model_validate(payload)

    assert response.status == OptimizationStatus.REJECTED
    assert response.downstream_ready is False


def test_valid_efficient_frontier_point():
    payload = valid_response()

    payload["efficient_frontier"] = [{
        "target_annual_return": 0.12,
        "expected_annual_return": 0.12,
        "annualized_volatility": 0.05,
        "allocations": [
            {"ticker": "VTI", "weight": 0.5},
            {"ticker": "BND", "weight": 0.5},
        ],
    }]

    response = OptimizationResponse.model_validate(payload)

    assert len(response.efficient_frontier) == 1


def test_response_calculation_version():
    response = OptimizationResponse.model_validate(
        valid_response()
    )

    assert response.calculation_version == "1.0.0"


def test_response_generated_timestamp():
    response = OptimizationResponse.model_validate(
        valid_response()
    )

    assert response.generated_at is not None
    assert response.generated_at.tzinfo is not None
