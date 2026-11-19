
"""
IE 591 — Portfolio Intelligence
Automated Backend Test Suite

Stage 4: Skill #3 — Portfolio Optimization API Integration
Version: 0.4.0

Tests:
- FastAPI application endpoints
- Application and skill status
- Investor profile validation
- Financial goal validation
- Portfolio allocation validation
- Shared analysis contracts
- Investor profile creation and retrieval
- Market Intelligence API integration
- Portfolio Optimization API status integration
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
