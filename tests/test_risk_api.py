"""IE 591 Skill #4 — Stage 4.6 FastAPI risk-assessment regression tests.

Run from the repository root: python -m pytest -v tests/test_risk_api.py
These tests exercise public HTTP contracts, not private engine implementation.
They use synthetic data, make no network calls, and never mutate Skills #1–#3.
"""
from __future__ import annotations

from copy import deepcopy
from math import exp, pi, sqrt
from statistics import NormalDist

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.risk_models import RiskAssessmentRequest, RiskAssessmentResult, RiskExportPackage


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as http:
        yield http


@pytest.fixture
def payload() -> dict:
    """Reproducible two-asset minimum-variance Skill #3 example."""
    return {
        "assessment_id": "IE591-RISK-TEST",
        "portfolio": {
            "portfolio_id": "IE591-OPT-DEMO",
            "assets": [
                {
                    "ticker": "ASSET_A", "asset_name": "Synthetic Equity",
                    "asset_class": "equity", "weight": 0.2,
                    "expected_annual_return": 0.12,
                },
                {
                    "ticker": "ASSET_B", "asset_name": "Synthetic Bonds",
                    "asset_class": "fixed_income", "weight": 0.8,
                    "expected_annual_return": 0.06,
                },
            ],
            "covariance_matrix": [[0.04, 0.0], [0.0, 0.01]],
            "provenance": {
                "source_type": "synthetic", "source_name": "IE 591 Test Fixture",
                "dataset_id": "RISK-API-TEST-001",
                "upstream_analysis_id": "IE591-OPT-DEMO",
                "is_synthetic": True,
            },
        },
        "config": {
            "method": "auto", "horizon": "1d",
            "confidence_level": 0.95, "trading_days_per_year": 252,
        },
        "stress_scenarios": [],
    }


def assess(client: TestClient, payload: dict):
    return client.post("/api/risk/assess", json=payload)


def scenarios(client: TestClient, payload: dict):
    return client.post("/api/risk/scenarios", json=payload)


def test_status_discovery(client):
    response = client.get("/api/risk/status")
    assert response.status_code == 200
    data = response.json()
    assert data["skill_id"] == 4
    assert data["engine"] == "available"
    assert data["contract_version"] == "1.0.0"
    assert set(data["methods"]) == {"auto", "parametric", "historical"}
    assert set(data["horizons"]) == {"1d", "10d", "30d", "1y"}
    assert data["historical_minimum_windows"] >= 20
    assert data["predefined_scenarios"] is True
    assert data["custom_scenarios"] is True
    assert data["external_data_fetching"] is False
    assert data["endpoints"] == {
        "assess": "/api/risk/assess", "scenarios": "/api/risk/scenarios"
    }


def test_openapi_registers_risk_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/api/risk/status"]
    assert "post" in paths["/api/risk/assess"]
    assert "post" in paths["/api/risk/scenarios"]


def test_assessment_response_contract_and_identity(client, payload):
    response = assess(client, payload)
    assert response.status_code == 200, response.text
    result = RiskAssessmentResult.model_validate(response.json())
    assert result.assessment_id == payload["assessment_id"]
    assert result.portfolio_id == payload["portfolio"]["portfolio_id"]
    assert result.status.value == "completed"
    assert result.downstream_ready is True
    assert result.provenance.dataset_id == "RISK-API-TEST-001"
    assert result.provenance.upstream_analysis_id == "IE591-OPT-DEMO"
    assert result.provenance.is_synthetic is True
    assert result.generated_at.tzinfo is not None


def test_expected_volatility_matches_portfolio_math(client, payload):
    result = assess(client, payload).json()
    expected = sqrt(0.2**2 * 0.04 + 0.8**2 * 0.01)
    assert result["annualized_volatility"] == pytest.approx(expected, abs=1e-10)


def test_auto_method_falls_back_without_history(client, payload):
    result = assess(client, payload).json()
    assert result["method_requested"] == "auto"
    assert result["method_used"] == "parametric"
    assert "Historical simulation unavailable" in result["method_fallback_reason"]
    assert result["validation"]["status"] == "warning"
    assert "PARAMETRIC_FALLBACK" in {
        issue["code"] for issue in result["validation"]["issues"]
    }
    assert result["validation"]["disabled_calculations"] == []


def test_parametric_var_and_cvar_against_independent_normal_formula(client, payload):
    payload["config"]["method"] = "parametric"
    result = assess(client, payload).json()
    annual_mean = 0.2 * 0.12 + 0.8 * 0.06
    sigma = sqrt(0.008 / 252)
    mu = annual_mean / 252
    z = NormalDist().inv_cdf(0.95)
    expected_var = -mu + z * sigma
    expected_cvar = -mu + sigma * exp(-z * z / 2) / (sqrt(2 * pi) * 0.05)
    assert result["value_at_risk"]["value"] == pytest.approx(expected_var, abs=1e-10)
    assert result["conditional_value_at_risk"]["value"] == pytest.approx(expected_cvar, abs=1e-10)
    assert result["value_at_risk"]["unit"] == "decimal"
    assert result["value_at_risk"]["method"] == "parametric"
    assert result["method_fallback_reason"] is None
    assert result["validation"]["status"] == "pass"


@pytest.mark.parametrize("horizon", ["1d", "10d", "30d", "1y"])
def test_all_supported_horizons(client, payload, horizon):
    payload["config"].update(method="parametric", horizon=horizon)
    response = assess(client, payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["horizon"] == horizon
    assert result["value_at_risk"]["horizon"] == horizon
    assert result["conditional_value_at_risk"]["horizon"] == horizon


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.975, 0.99])
def test_all_supported_confidence_levels(client, payload, confidence):
    payload["config"].update(method="parametric", confidence_level=confidence)
    response = assess(client, payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["confidence_level"] == pytest.approx(confidence)
    assert result["conditional_value_at_risk"]["value"] >= result["value_at_risk"]["value"]


def test_explicit_historical_insufficient_data_never_silently_falls_back(client, payload):
    payload["config"]["method"] = "historical"
    result = assess(client, payload).json()
    assert result["status"] == "partial"
    assert result["method_used"] is None
    assert result["value_at_risk"] is None
    assert result["conditional_value_at_risk"] is None
    assert result["downstream_ready"] is False
    assert "INSUFFICIENT_HISTORICAL_WINDOWS" in {
        issue["code"] for issue in result["validation"]["issues"]
    }
    assert {"value_at_risk", "conditional_value_at_risk"}.issubset(
        result["validation"]["disabled_calculations"]
    )


def test_historical_method_with_sufficient_observations(client, payload):
    payload["config"]["method"] = "historical"
    payload["portfolio"]["historical_returns"] = {
        "frequency": "daily",
        "observations": [[-0.01 if i % 5 == 0 else 0.005, 0.002] for i in range(60)],
    }
    response = assess(client, payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["method_used"] == "historical"
    assert result["value_at_risk"]["method"] == "historical"
    assert result["conditional_value_at_risk"]["value"] >= result["value_at_risk"]["value"]
    assert result["downstream_ready"] is True


def test_auto_uses_historical_when_sufficient(client, payload):
    payload["portfolio"]["historical_returns"] = {
        "frequency": "daily", "observations": [[0.002 * ((i % 7) - 3), 0.001] for i in range(60)]
    }
    result = assess(client, payload).json()
    assert result["method_used"] == "historical"
    assert result["method_fallback_reason"] is None


def test_risk_attribution_reconciles(client, payload):
    result = assess(client, payload).json()
    rows = {r["ticker"]: r for r in result["asset_risk_contributions"]}
    assert set(rows) == {"ASSET_A", "ASSET_B"}
    assert sum(r["percent_total_risk"] for r in rows.values()) == pytest.approx(1)
    assert sum(r["component_volatility"] for r in rows.values()) == pytest.approx(
        result["annualized_volatility"]
    )
    assert rows["ASSET_A"]["percent_total_risk"] == pytest.approx(0.2)
    assert rows["ASSET_B"]["percent_total_risk"] == pytest.approx(0.8)


def test_predefined_stress_scenarios_reconcile(client, payload):
    response = scenarios(client, payload)
    assert response.status_code == 200, response.text
    rows = response.json()
    assert len(rows) == 3
    by_id = {row["scenario_id"]: row for row in rows}
    assert by_id["combined_market_shock"]["portfolio_shock_return"] == pytest.approx(-0.156)
    assert by_id["bond_selloff"]["portfolio_shock_return"] == pytest.approx(-0.09)
    assert by_id["equity_drawdown"]["portfolio_shock_return"] == pytest.approx(-0.024)
    for row in rows:
        assert row["hypothetical"] is True
        assert sum(row["contributions"].values()) == pytest.approx(row["portfolio_shock_return"])
    assert [r["portfolio_shock_return"] for r in rows] == sorted(
        r["portfolio_shock_return"] for r in rows
    )


def test_assessment_and_scenario_endpoint_agree(client, payload):
    assessment = assess(client, payload)
    scenario_response = scenarios(client, payload)
    assert assessment.status_code == scenario_response.status_code == 200
    assert assessment.json()["stress_results"] == scenario_response.json()


def test_custom_scenario_weighted_shock(client, payload):
    payload["stress_scenarios"] = [{
        "scenario_id": "custom_rate_shock", "name": "Custom rate shock",
        "scenario_type": "custom",
        "shocks": [{"ticker": "ASSET_B", "shock_return": -0.25}],
    }]
    result = assess(client, payload)
    assert result.status_code == 200, result.text
    custom = next(r for r in result.json()["stress_results"]
                  if r["scenario_id"] == "custom_rate_shock")
    assert custom["portfolio_shock_return"] == pytest.approx(-0.20)
    assert custom["contributions"]["ASSET_A"] == pytest.approx(0)
    assert custom["contributions"]["ASSET_B"] == pytest.approx(-0.20)


def test_custom_scenario_cannot_reuse_predefined_id(client, payload):
    payload["stress_scenarios"] = [{
        "scenario_id": "bond_selloff", "name": "Collision",
        "scenario_type": "custom",
        "shocks": [{"ticker": "ASSET_A", "shock_return": -0.10}],
    }]
    for endpoint in (assess, scenarios):
        response = endpoint(client, payload)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] in {
            "RISK_ASSESSMENT_FAILED", "RISK_SCENARIO_INVALID"
        }


def test_classification_is_dimension_specific(client, payload):
    result = assess(client, payload).json()
    classification = result["classification"]
    assert classification["dimensions"] == {
        "volatility": "low", "downside": "low",
        "concentration": "very_high", "stress": "high",
    }
    assert classification["overall"] == "very_high"
    assert classification["explanations"]
    assert classification["thresholds"]["concentration_high"] == pytest.approx(0.5)


def test_missing_investor_profile_never_implies_suitability(client, payload):
    result = assess(client, payload).json()
    alignment = result["investor_alignment"]
    assert alignment["status"] == "insufficient_information"
    assert alignment["suitability_determination"] is False
    assert set(alignment["missing_fields"]) == {
        "risk_tolerance", "investment_horizon_years",
        "financial_objectives", "constraints",
    }


def test_investor_mismatch_is_flagged_not_suitability_advice(client, payload):
    payload["investor_profile"] = {
        "profile_id": "INV-001", "risk_tolerance": "low",
        "investment_horizon_years": 1,
        "financial_objectives": ["capital preservation"],
        "constraints": ["no leverage"],
    }
    result = assess(client, payload).json()
    alignment = result["investor_alignment"]
    assert alignment["status"] == "potential_mismatch"
    assert alignment["suitability_determination"] is False


def test_export_package_roundtrip_validates(client, payload):
    response = assess(client, payload)
    assert response.status_code == 200, response.text
    package = RiskExportPackage(
        synthetic=True,
        input=RiskAssessmentRequest.model_validate(payload),
        assessment=RiskAssessmentResult.model_validate(response.json()),
    )
    restored = RiskExportPackage.model_validate_json(package.model_dump_json())
    assert restored.project == "IE 591 Portfolio Intelligence"
    assert restored.skill == "Risk Assessment"
    assert restored.input.assessment_id == restored.assessment.assessment_id
    assert restored.assessment.downstream_ready is True


@pytest.mark.parametrize("change", [
    lambda p: p["portfolio"]["assets"][0].update(weight=0.4),
    lambda p: p["portfolio"]["assets"][1].update(ticker="ASSET_A"),
    lambda p: p["portfolio"].update(covariance_matrix=[[0.04, 0], [0, -0.01]]),
    lambda p: p["portfolio"].update(covariance_matrix=[[0.04, 0.01], [0, 0.01]]),
    lambda p: p["portfolio"].update(covariance_matrix=[[0.04, 0.03], [0.03, 0.01]]),
    lambda p: p["config"].update(confidence_level=0.80),
    lambda p: p["config"].update(horizon="5d"),
    lambda p: p["config"].update(method="unrecognized"),
    lambda p: p.update(unexpected_field=True),
])
def test_invalid_requests_rejected_with_422(client, payload, change):
    change(payload)
    response = assess(client, payload)
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_unknown_custom_scenario_ticker_rejected(client, payload):
    payload["stress_scenarios"] = [{
        "scenario_id": "unknown_ticker", "name": "Unknown ticker",
        "scenario_type": "custom",
        "shocks": [{"ticker": "DOES_NOT_EXIST", "shock_return": -0.10}],
    }]
    assert assess(client, payload).status_code == 422
    assert scenarios(client, payload).status_code == 422


def test_duplicate_custom_scenario_ids_rejected(client, payload):
    entry = {
        "scenario_id": "custom_duplicate", "name": "Duplicate",
        "scenario_type": "custom",
        "shocks": [{"ticker": "ASSET_A", "shock_return": -0.10}],
    }
    payload["stress_scenarios"] = [deepcopy(entry), deepcopy(entry)]
    assert assess(client, payload).status_code == 422


def test_unmapped_asset_class_records_warning(client, payload):
    payload["portfolio"]["assets"][0]["asset_class"] = "alternative_unknown"
    response = assess(client, payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert "UNMAPPED_PREDEFINED_SCENARIO_CLASSES" in {
        issue["code"] for issue in result["validation"]["issues"]
    }
    assert result["validation"]["status"] == "warning"


def test_requests_are_deterministic_except_generation_timestamp(client, payload):
    payload["config"]["method"] = "parametric"
    first = assess(client, payload)
    second = assess(client, payload)
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    a.pop("generated_at")
    b.pop("generated_at")
    assert a == b


def test_existing_skill_endpoints_remain_registered(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/api/health", "/api/skills", "/api/investor-profile",
        "/api/market/status", "/api/market/analyze",
        "/api/optimization/status", "/api/optimization/solve",
        "/api/optimization/frontier",
    ):
        assert path in paths, f"Legacy Skill #1–#3 endpoint missing: {path}"
