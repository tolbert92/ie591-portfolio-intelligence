
"""
IE 591 — AI Portfolio Optimization
Stage 1: Backend Foundation Tests

Run from the repository root:
    python -m pytest -v
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.main import app
from backend.models import (
    AnalysisRequest,
    FinancialGoal,
    InvestorProfile,
    Portfolio,
)

client = TestClient(app)


# ---------------------------------------
# API endpoint tests
# ---------------------------------------

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == (
        "IE 591 Portfolio Intelligence"
    )
    assert data["skills_total"] == 10
    assert data["skills_integrated"] == 0


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_skills_endpoint():
    response = client.get("/api/skills")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 10
    assert data["integrated"] == 0
    assert len(data["skills"]) == 10

    assert all(
        skill["status"] == "pending_integration"
        for skill in data["skills"]
    )


def test_application_status():
    response = client.get("/api/status")

    assert response.status_code == 200

    data = response.json()

    assert data["stage"] == 1
    assert data["skills_integrated"] == 0
    assert data["professor_validation"] == "pending"


def test_unknown_endpoint_returns_404():
    response = client.get("/api/not-a-real-endpoint")

    assert response.status_code == 404


# ---------------------------------------
# Investor validation tests
# ---------------------------------------

def valid_investor():
    return {
        "investor_id": "SYNTHETIC-001",
        "investor_name": "Synthetic Investor A",
        "initial_investment": 250000,
        "annual_contribution": 12000,
        "investment_horizon_years": 25,
        "risk_tolerance": "moderate",
        "currency": "USD",
        "goals": [],
    }


def test_valid_investor_profile():
    investor = InvestorProfile(**valid_investor())

    assert investor.investor_id == "SYNTHETIC-001"
    assert investor.initial_investment == 250000
    assert investor.risk_tolerance.value == "moderate"


def test_negative_investment_rejected():
    data = valid_investor()
    data["initial_investment"] = -100

    with pytest.raises(ValidationError):
        InvestorProfile(**data)


def test_invalid_risk_tolerance_rejected():
    data = valid_investor()
    data["risk_tolerance"] = "extreme"

    with pytest.raises(ValidationError):
        InvestorProfile(**data)


def test_invalid_horizon_rejected():
    data = valid_investor()
    data["investment_horizon_years"] = 0

    with pytest.raises(ValidationError):
        InvestorProfile(**data)


def test_duplicate_goal_ids_rejected():
    data = valid_investor()

    goal = {
        "goal_id": "GOAL-001",
        "goal_type": "retirement",
        "target_amount": 1500000,
        "target_year": 2051,
        "priority": "mandatory",
        "currency": "USD",
    }

    data["goals"] = [goal, goal.copy()]

    with pytest.raises(ValidationError):
        InvestorProfile(**data)


# ---------------------------------------
# Portfolio validation tests
# ---------------------------------------

def test_valid_portfolio():
    portfolio = Portfolio(
        portfolio_id="PORT-001",
        portfolio_name="Balanced Portfolio",
        allocations=[
            {"asset_id": "STOCKS", "weight": 0.6},
            {"asset_id": "BONDS", "weight": 0.3},
            {"asset_id": "CASH", "weight": 0.1},
        ],
    )

    assert len(portfolio.allocations) == 3


def test_portfolio_weights_must_total_one():
    with pytest.raises(ValidationError):
        Portfolio(
            portfolio_id="PORT-002",
            portfolio_name="Invalid Portfolio",
            allocations=[
                {"asset_id": "STOCKS", "weight": 0.6},
                {"asset_id": "BONDS", "weight": 0.2},
            ],
        )


def test_duplicate_asset_ids_rejected():
    with pytest.raises(ValidationError):
        Portfolio(
            portfolio_id="PORT-003",
            portfolio_name="Duplicate Assets",
            allocations=[
                {"asset_id": "STOCKS", "weight": 0.5},
                {"asset_id": "STOCKS", "weight": 0.5},
            ],
        )


# ---------------------------------------
# Shared contract tests
# ---------------------------------------

def test_valid_analysis_request():
    request = AnalysisRequest(
        request_id="REQ-001",
        investor=valid_investor(),
        portfolios=[],
    )

    assert request.schema_version == "1.0.0"
    assert request.request_id == "REQ-001"


def test_unexpected_fields_rejected():
    data = valid_investor()
    data["unsupported_field"] = "unexpected"

    with pytest.raises(ValidationError):
        InvestorProfile(**data)


def test_nonfinite_investment_rejected():
    data = valid_investor()
    data["initial_investment"] = float("inf")

    with pytest.raises(ValidationError):
        InvestorProfile(**data)
