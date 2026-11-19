"""IE 591 — Skill #3, Gate 4.2.3: efficient frontier tests.

Synthetic educational portfolios; no API credentials or network access.
"""

import math
from copy import deepcopy

import pytest

from backend.optimization_frontier import (
    EfficientFrontierError,
    generate_efficient_frontier,
)
from backend.optimization_models import OptimizationRequest


@pytest.fixture
def request_payload():
    return {
        "optimization_id": "IE591-GATE423-TEST",
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
        "covariance_matrix": [[0.04, 0.0], [0.0, 0.01]],
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
            "source_name": "IE 591 Frontier Validation",
            "dataset_id": "GATE423-001",
            "upstream_analysis_id": "IE591-GATE34-THREE",
            "is_synthetic": True,
        },
    }


@pytest.fixture
def optimization_request(request_payload):
    return OptimizationRequest.model_validate(request_payload)


def make_request(payload):
    return OptimizationRequest.model_validate(payload)


def weights(point):
    return {allocation.ticker: allocation.weight for allocation in point.allocations}


# Mathematical reference values for two independent assets:
# minimum variance = (20% A, 80% B), return = 7.2%, volatility = sqrt(.008)
# maximum return = (100% A, 0% B), return = 12%, volatility = 20%.


def test_generates_requested_number_of_points(optimization_request):
    assert len(generate_efficient_frontier(optimization_request, 7)) == 7


def test_two_point_frontier(optimization_request):
    assert len(generate_efficient_frontier(optimization_request, 2)) == 2


def test_default_point_count(optimization_request):
    points = generate_efficient_frontier(optimization_request)
    assert len(points) == optimization_request.config.efficient_frontier_points


def test_minimum_variance_endpoint(optimization_request):
    first = generate_efficient_frontier(optimization_request, 5)[0]
    assert first.expected_annual_return == pytest.approx(0.072, abs=1e-5)
    assert first.annualized_volatility == pytest.approx(math.sqrt(0.008), abs=1e-5)
