"""IE 591 — Skill #3, Gate 4.3: Portfolio Optimization API tests.

Tests are deterministic and use synthetic inputs only.
Run: python -m pytest -v tests/test_optimization_api.py
"""

from copy import deepcopy
from datetime import datetime
import math

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.optimization_models import OptimizationResponse, EfficientFrontierPoint

client = TestClient(app)


@pytest.fixture
def payload():
    return {
        "optimization_id": "IE591-API-SYNTHETIC-001",
        "assets": [
            {"ticker": "ASSET_A", "asset_name": "Synthetic Equity", "asset_class": "equity", "expected_annual_return": 0.12, "minimum_weight": 0.0, "maximum_weight": 1.0},
            {"ticker": "ASSET_B", "asset_name": "Synthetic Bonds", "asset_class": "fixed_income", "expected_annual_return": 0.06, "minimum_weight": 0.0, "maximum_weight": 1.0},
        ],
        "covariance_matrix": [[0.04, 0.0], [0.0, 0.01]],
        "constraints": {"fully_invested": True, "allow_short_selling": False, "allow_leverage": False},
        "config": {"objective": "minimum_variance", "risk_free_annual_rate": 0.02, "efficient_frontier_points": 5, "solver_tolerance": 1e-9, "maximum_iterations": 2000},
        "provenance": {"source_type": "synthetic", "source_name": "IE 591 Synthetic Test Dataset", "dataset_id": "IE591-SYNTH-001", "upstream_analysis_id": "MARKET-SYNTH-001", "is_synthetic": True},
    }


def post_solve(payload):
    return client.post("/api/optimization/solve", json=payload)


def post_frontier(payload):
    return client.post("/api/optimization/frontier", json=payload)


def test_optimization_status_endpoint():
    response = client.get("/api/optimization/status")
    assert response.status_code == 200
    data = response.json()
    assert data["skill_id"] == 3
    assert data["skill_name"] == "Portfolio Optimization"
    assert data["engine"] == "available"
    assert data["efficient_frontier"] is True
    assert data["external_data_fetching"] is False
    assert data["endpoints"]["solve"] == "/api/optimization/solve"
    assert data["endpoints"]["frontier"] == "/api/optimization/frontier"
    assert {"minimum_variance", "maximum_sharpe"}.issubset(set(data["objectives"]))


def test_optimization_routes_registered():
    routes = {route.path for route in app.routes}
    assert {"/api/optimization/status", "/api/optimization/solve", "/api/optimization/frontier"}.issubset(routes)


def test_optimization_openapi_documented():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "post" in paths["/api/optimization/solve"]
    assert "post" in paths["/api/optimization/frontier"]
    assert "get" in paths["/api/optimization/status"]


def test_minimum_variance_response_contract(payload):
    response = post_solve(payload)
    assert response.status_code == 200, response.text
    data = response.json()
    OptimizationResponse.model_validate(data)
    assert data["optimization_id"] == payload["optimization_id"]
    assert data["status"] == "completed"
    assert data["objective"] == "minimum_variance"
    assert data["downstream_ready"] is True
    assert data["quality"]["status"] == "pass"
    assert data["quality"]["constraints_satisfied"] is True
    assert data["quality"]["weights_sum_to_one"] is True
    assert data["quality"]["covariance_matrix_valid"] is True
    assert data["calculation_version"]
    assert datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
    assert data["efficient_frontier"] == []


def test_minimum_variance_analytical_weights(payload):
    response = post_solve(payload)
    assert response.status_code == 200, response.text
    allocations = {item["ticker"]: item["weight"] for item in response.json()["allocations"]}
    assert allocations["ASSET_A"] == pytest.approx(0.2, abs=1e-4)
    assert allocations["ASSET_B"] == pytest.approx(0.8, abs=1e-4)
    assert math.fsum(allocations.values()) == pytest.approx(1.0, abs=1e-8)
