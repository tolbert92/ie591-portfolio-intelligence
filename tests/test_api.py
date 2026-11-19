
"""
IE 591 — Portfolio Intelligence
Automated Backend Test Suite

Stage 3: Skill #2 — Market Intelligence Integration
Version: 0.3.1

Tests:
- FastAPI application endpoints
- Application and skill status
- Investor profile validation
- Financial goal validation
- Portfolio allocation validation
- Shared analysis contracts
- Investor profile creation and retrieval
- Market Intelligence API integration
- API error handling

Run from the repository root:
    python -m pytest -v
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.main import app
from backend.models import (
    AnalysisRequest,
    InvestorProfile,
    Portfolio,
)


client = TestClient(app)


# ==================================================
# TEST DATA HELPERS
# ==================================================

def valid_investor():
    """Return a valid synthetic investor profile."""
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


def unique_investor():
    """Return a synthetic investor with a unique ID."""
    data = valid_investor()
    data["investor_id"] = f"TEST-{uuid4().hex}"
    return data


# ==================================================
# SECTION 1 — APPLICATION API TESTS
# ==================================================

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

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == (
        "portfolio-intelligence-api"
    )
    assert data["version"] == "0.4.0"


def test_skills_endpoint():
    response = client.get("/api/skills")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 10
    assert data["integrated"] == 0
    assert len(data["skills"]) == 10

    assert data["skills"][0] == {
        "id": 1,
        "name": "Investor Profile",
        "status": "integration_in_progress",
    }

    assert data["skills"][1] == {
        "id": 2,
        "name": "Market Intelligence",
        "status": "integration_in_progress",
    }

    assert data["skills"][2] == {
        "id": 3,
        "name": "Portfolio Optimization",
        "status": "integration_in_progress",
    }

    assert all(
        skill["status"] == "pending_integration"
        for skill in data["skills"][3:]
    )


def test_application_status():
    response = client.get("/api/status")

    assert response.status_code == 200

    data = response.json()

    assert data["stage"] == 4
    assert data["stage_name"] == (
        "Skill #2 Integration"
    )
    assert data["frontend"] == (
        "investor_profile_connected"
    )
    assert data["backend"] == (
        "market_intelligence_api"
    )
    assert data["analytics"] == (
        "market_engine_connected"
    )
    assert data["skills_integrated"] == 0
    assert data["professor_validation"] == "pending"


def test_unknown_endpoint_returns_404():
    response = client.get(
        "/api/not-a-real-endpoint"
    )

    assert response.status_code == 404


# ==================================================
# SECTION 2 — INVESTOR MODEL VALIDATION
# ==================================================

def test_valid_investor_profile():
    investor = InvestorProfile(
        **valid_investor()
    )

    assert investor.investor_id == (
        "SYNTHETIC-001"
    )
    assert investor.initial_investment == 250000
    assert investor.annual_contribution == 12000
    assert investor.risk_tolerance.value == (
        "moderate"
    )


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

    data["goals"] = [
        goal,
        goal.copy(),
    ]

    with pytest.raises(ValidationError):
        InvestorProfile(**data)


# ==================================================
# SECTION 3 — PORTFOLIO VALIDATION
# ==================================================

def test_valid_portfolio():
    portfolio = Portfolio(
        portfolio_id="PORT-001",
        portfolio_name="Balanced Portfolio",
        allocations=[
            {
                "asset_id": "STOCKS",
                "weight": 0.6,
            },
            {
                "asset_id": "BONDS",
                "weight": 0.3,
            },
            {
                "asset_id": "CASH",
                "weight": 0.1,
            },
        ],
    )

    assert len(portfolio.allocations) == 3

    total_weight = sum(
        asset.weight
        for asset in portfolio.allocations
    )

    assert total_weight == pytest.approx(1.0)


def test_portfolio_weights_must_total_one():
    with pytest.raises(ValidationError):
        Portfolio(
            portfolio_id="PORT-002",
            portfolio_name="Invalid Portfolio",
            allocations=[
                {
                    "asset_id": "STOCKS",
                    "weight": 0.6,
                },
                {
                    "asset_id": "BONDS",
                    "weight": 0.2,
                },
            ],
        )


def test_duplicate_asset_ids_rejected():
    with pytest.raises(ValidationError):
        Portfolio(
            portfolio_id="PORT-003",
            portfolio_name="Duplicate Assets",
            allocations=[
                {
                    "asset_id": "STOCKS",
                    "weight": 0.5,
                },
                {
                    "asset_id": "STOCKS",
                    "weight": 0.5,
                },
            ],
        )


# ==================================================
# SECTION 4 — SHARED DATA CONTRACTS
# ==================================================

def test_valid_analysis_request():
    request = AnalysisRequest(
        request_id="REQ-001",
        investor=valid_investor(),
        portfolios=[],
    )

    assert request.schema_version == "1.0.0"
    assert request.request_id == "REQ-001"
    assert request.investor.investor_id == (
        "SYNTHETIC-001"
    )


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


# ==================================================
# SECTION 5 — SKILL #1 API INTEGRATION
# ==================================================

def test_create_investor_profile():
    profile = unique_investor()

    response = client.post(
        "/api/investor-profile",
        json=profile,
    )

    assert response.status_code == 201

    data = response.json()

    assert data["status"] == "created"
    assert data["investor_id"] == (
        profile["investor_id"]
    )
    assert data["storage"] == (
        "temporary_memory"
    )
    assert data["profile"]["initial_investment"] == (
        250000
    )


def test_retrieve_investor_profile():
    profile = unique_investor()

    created = client.post(
        "/api/investor-profile",
        json=profile,
    )

    assert created.status_code == 201

    response = client.get(
        "/api/investor-profile/"
        f"{profile['investor_id']}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "found"
    assert data["profile"]["investor_id"] == (
        profile["investor_id"]
    )
    assert data["profile"]["risk_tolerance"] == (
        "moderate"
    )


def test_duplicate_investor_profile_rejected():
    profile = unique_investor()

    first = client.post(
        "/api/investor-profile",
        json=profile,
    )

    second = client.post(
        "/api/investor-profile",
        json=profile,
    )

    assert first.status_code == 201
    assert second.status_code == 409

    assert second.json()["detail"] == (
        "An investor profile with this ID "
        "already exists."
    )


def test_invalid_investor_profile_rejected():
    profile = unique_investor()
    profile["initial_investment"] = -500

    response = client.post(
        "/api/investor-profile",
        json=profile,
    )

    assert response.status_code == 422


def test_missing_investor_profile_returns_404():
    missing_id = f"MISSING-{uuid4().hex}"

    response = client.get(
        "/api/investor-profile/"
        f"{missing_id}"
    )

    assert response.status_code == 404

    assert response.json()["detail"] == (
        "Investor profile not found."
    )


# ==================================================
# SECTION 6 — SKILL #2 API INTEGRATION
# ==================================================

def test_market_status_endpoint():
    response = client.get("/api/market/status")

    assert response.status_code == 200

    data = response.json()

    assert data["skill_id"] == 2
    assert data["skill_name"] == (
        "Market Intelligence"
    )
    assert data["engine"] == "available"
    assert data["status"] == (
        "integration_in_progress"
    )
    assert data["api_endpoint"] == (
        "/api/market/analyze"
    )
    assert data["method"] == "POST"
    assert data["external_data_fetching"] is False


def test_market_analysis_invalid_request():
    response = client.post(
        "/api/market/analyze",
        json={},
    )

    assert response.status_code == 422


def test_market_analysis_endpoint_registered():
    routes = {
        route.path
        for route in app.routes
    }

    assert "/api/market/analyze" in routes
    assert "/api/market/status" in routes
