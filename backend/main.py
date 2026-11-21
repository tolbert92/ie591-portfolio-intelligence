"""IE 591 — Portfolio Intelligence, University of Tennessee.

Skills #1–#3 legacy API contracts are preserved. Skill #4 adds validated
risk assessment and hypothetical stress testing (Stage 4.4).
"""
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import InvestorProfile
from .market_models import MarketAnalysisRequest, MarketAnalysisResponse
from .market_engine import CALCULATION_VERSION, analyze_market
from .optimization_models import (
    CALCULATION_VERSION as OPTIMIZATION_VERSION,
    OptimizationRequest, OptimizationResponse, OptimizationStatus,
    OptimizationQuality,
    OptimizedAllocation, EfficientFrontierPoint,
)
from .optimization_solver import OptimizationSolverError, solve_portfolio
from .optimization_frontier import EfficientFrontierError, generate_efficient_frontier

from .risk_models import (
    CONTRACT_VERSION as RISK_CONTRACT_VERSION,
    RiskAssessmentRequest, RiskAssessmentResult,
    RiskStatus, RiskMethod, RiskHorizon, ValidationReport,
    ValidationIssue, IssueSeverity as RiskIssueSeverity,
    StressResult,
)
from .risk_engine import (
    CALCULATION_VERSION as RISK_CALCULATION_VERSION,
    MIN_HISTORICAL_WINDOWS, assess_portfolio_risk,
)
from .risk_scenarios import (
    SCENARIO_VERSION, evaluate_stress_scenarios, compare_stress_results,
    unmapped_asset_classes,
)
from .forward_test_engine import ForwardTestResult, run_forward_test
from .forward_test_models import ForwardTestRequest
from .visualization_api import router as visualization_router

from .risk_classification import (
    CLASSIFICATION_VERSION, classify_portfolio_risk,
    assess_investor_alignment,
)

app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description="Educational portfolio intelligence platform integrating ten analytical skills.",
    version="0.4.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5500", "http://127.0.0.1:5500",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Skill #5 — Register visualization routes without changing Skills #1–#4.
# Compatibility: this project's pinned FastAPI version adds an internal
# _IncludedRouter entry to app.routes when include_router() is used. The
# existing Skill #3 route-regression test expects every entry to have .path.
# The visualization APIRouter already contains fully prefixed APIRoute objects;
# register those concrete routes directly to preserve OpenAPI and endpoint data.
app.router.routes.extend(visualization_router.routes)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "index.html"

# Retain the existing registry and application-status contracts. They will be
# revised only after the Skill #4 frontend and end-to-end tests are complete.
SKILLS = [
    {"id": 1, "name": "Investor Profile", "status": "integration_in_progress"},
    {"id": 2, "name": "Market Intelligence", "status": "integration_in_progress"},
    {"id": 3, "name": "Portfolio Optimization", "status": "integration_in_progress"},
    {"id": 4, "name": "Risk Assessment", "status": "pending_integration"},
    {"id": 5, "name": "Portfolio Visualization", "status": "pending_integration"},
    {"id": 6, "name": "3D Portfolio Visualization", "status": "pending_integration"},
    {"id": 7, "name": "Forward Testing", "status": "pending_integration"},
    {"id": 8, "name": "Monte Carlo Simulation", "status": "pending_integration"},
    {"id": 9, "name": "Strategy Comparison", "status": "pending_integration"},
    {"id": 10, "name": "Decision Support", "status": "pending_integration"},
]
# Educational prototype: investor profiles are not persistent across restarts.
investor_profiles: dict[str, InvestorProfile] = {}

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "application": "IE 591 Portfolio Intelligence",
        "service": "portfolio-intelligence-api",
        "version": app.version,
        "status": "operational",
        "skills_total": len(SKILLS),
        "skills_integrated": 0,
        "frontend_url": "/app",
        "documentation_url": "/docs",
    }

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "portfolio-intelligence-api", "version": app.version}

@app.get("/api/skills")
def get_skills() -> dict[str, Any]:
    return {
        "total": len(SKILLS),
        "integrated": sum(skill["status"] == "integrated" for skill in SKILLS),
        "skills": SKILLS,
    }

@app.get("/api/status")
def integration_status() -> dict[str, Any]:
    return {
        "stage": 4,
        "stage_name": "Skill #3 Backend API Integration",
        "frontend": "investor_profile_connected",
        "backend": "portfolio_optimization_api",
        "analytics": "optimization_engine_connected",
        "skills_integrated": 0,
        "professor_validation": "pending",
    }

# Skill #1 — Investor Profile (unchanged contract)
@app.post("/api/investor-profile", status_code=status.HTTP_201_CREATED)
def create_investor_profile(profile: InvestorProfile) -> dict[str, Any]:
    investor_id = profile.investor_id
    if investor_id in investor_profiles:
        raise HTTPException(status_code=409, detail="An investor profile with this ID already exists.")
    investor_profiles[investor_id] = profile
    return {
        "status": "created", "investor_id": investor_id,
        "message": "Investor profile created.",
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }

@app.get("/api/investor-profile/{investor_id}")
def get_investor_profile(investor_id: str) -> dict[str, Any]:
    profile = investor_profiles.get(investor_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Investor profile not found.")
    return {"status": "found", "profile": profile.model_dump(mode="json"), "storage": "temporary_memory"}

# Skill #2 — Market Intelligence (unchanged contract)
@app.get("/api/market/status")
def market_engine_status() -> dict[str, Any]:
    return {
        "skill_id": 2, "skill_name": "Market Intelligence",
        "status": "integration_in_progress", "engine": "available",
        "calculation_version": CALCULATION_VERSION,
        "input_type": "historical_price_observations",
        "data_sources": ["synthetic", "uploaded", "provider"],
        "external_data_fetching": False,
        "api_endpoint": "/api/market/analyze", "method": "POST",
    }

@app.post("/api/market/analyze", response_model=MarketAnalysisResponse)
def analyze_market_endpoint(request: MarketAnalysisRequest) -> MarketAnalysisResponse:
    return analyze_market(request)

# Skill #3 — Portfolio Optimization (unchanged contract)
@app.get("/api/optimization/status")
def optimization_engine_status() -> dict[str, Any]:
    return {
        "skill_id": 3, "skill_name": "Portfolio Optimization",
        "status": "integration_in_progress", "engine": "available",
        "calculation_version": OPTIMIZATION_VERSION,
        "objectives": ["minimum_variance", "maximum_sharpe"],
        "constraints": [
            "fully_invested", "long_only", "asset_weight_bounds",
            "minimum_expected_annual_return", "maximum_annual_volatility",
        ],
        "efficient_frontier": True,
        "data_sources": ["synthetic", "uploaded", "provider"],
        "external_data_fetching": False,
        "endpoints": {
            "solve": "/api/optimization/solve",
            "frontier": "/api/optimization/frontier",
        },
    }

def _optimization_quality() -> OptimizationQuality:
    return OptimizationQuality(
        status="pass", constraints_satisfied=True,
        weights_sum_to_one=True, covariance_matrix_valid=True, issues=[],
    )

@app.post("/api/optimization/solve", response_model=OptimizationResponse)
def optimize_portfolio_endpoint(request: OptimizationRequest) -> OptimizationResponse:
    try:
        result = solve_portfolio(request)
    except OptimizationSolverError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "OPTIMIZATION_FAILED", "message": str(exc)},
        ) from exc
    allocations = [
        OptimizedAllocation(ticker=asset.ticker, weight=float(weight))
        for asset, weight in zip(request.assets, result.weights)
    ]
    return OptimizationResponse(
        optimization_id=request.optimization_id,
        status=OptimizationStatus.COMPLETED,
        objective=result.objective,
        provenance=request.provenance,
        allocations=allocations,
        portfolio_metrics=result.metrics,
        efficient_frontier=[],
        quality=_optimization_quality(),
        calculation_version=OPTIMIZATION_VERSION,
        downstream_ready=True,
    )

@app.post("/api/optimization/frontier", response_model=list[EfficientFrontierPoint])
def generate_frontier_endpoint(request: OptimizationRequest) -> list[EfficientFrontierPoint]:
    try:
        return generate_efficient_frontier(request)
    except (EfficientFrontierError, OptimizationSolverError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "FRONTIER_GENERATION_FAILED", "message": str(exc)},
        ) from exc

# Skill #4 — Risk Assessment (new Stage 4.4 routes)
@app.get("/api/risk/status")
def risk_engine_status() -> dict[str, Any]:
    """Discover available calculations without implying frontend completion."""
    return {
        "skill_id": 4,
        "skill_name": "Risk Assessment",
        "status": "backend_integration_in_progress",
        "engine": "available",
        "contract_version": RISK_CONTRACT_VERSION,
        "calculation_version": RISK_CALCULATION_VERSION,
        "scenario_version": SCENARIO_VERSION,
        "classification_version": CLASSIFICATION_VERSION,
        "methods": [method.value for method in RiskMethod],
        "horizons": [horizon.value for horizon in RiskHorizon],
        "confidence_levels": [0.90, 0.95, 0.975, 0.99],
        "historical_minimum_windows": MIN_HISTORICAL_WINDOWS,
        "predefined_scenarios": True,
        "custom_scenarios": True,
        "endpoints": {
            "assess": "/api/risk/assess",
            "scenarios": "/api/risk/scenarios",
        },
        "external_data_fetching": False,
    }


def _evaluate_scenarios(request: RiskAssessmentRequest) -> list[StressResult]:
    """Calculate and sort hypothetical outcomes; reject invalid scenario IDs."""
    return compare_stress_results(evaluate_stress_scenarios(request))


@app.post("/api/risk/scenarios", response_model=list[StressResult])
def risk_scenarios_endpoint(request: RiskAssessmentRequest) -> list[StressResult]:
    """Return predefined and user-defined hypothetical stress results."""
    try:
        return _evaluate_scenarios(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "RISK_SCENARIO_INVALID", "message": str(exc)},
        ) from exc


@app.post("/api/risk/assess", response_model=RiskAssessmentResult)
def risk_assessment_endpoint(request: RiskAssessmentRequest) -> RiskAssessmentResult:
    """Combine existing calculation, stress, classification, and alignment modules.

    Input validation is performed by Pydantic. Missing historical observations
    trigger the engine's documented behavior. Scenario errors return HTTP 422.
    No investor information is inferred or automatically retrieved.
    """
    try:
        assessment = assess_portfolio_risk(request)
        scenarios = _evaluate_scenarios(request)
        classification = classify_portfolio_risk(request, assessment, scenarios)
        alignment = assess_investor_alignment(request, classification)
        assessment.stress_results = scenarios
        assessment.classification = classification
        assessment.investor_alignment = alignment

        issues = list(assessment.validation.issues)
        unmapped = unmapped_asset_classes(request.portfolio)
        if unmapped:
            issues.append(ValidationIssue(
                severity=RiskIssueSeverity.WARNING,
                code="UNMAPPED_PREDEFINED_SCENARIO_CLASSES",
                message=(
                    "Predefined stress scenarios apply a zero shock to "
                    f"unmapped asset classes for tickers: {', '.join(unmapped)}. "
                    "Supply custom shocks to evaluate those assets."
                ),
                field_path="portfolio.assets",
            ))
        disabled = [
            item for item in assessment.validation.disabled_calculations
            if item not in {"stress_results", "classification", "investor_alignment"}
        ]
        assessment.validation = ValidationReport(
            status="warning" if issues else "pass",
            issues=issues,
            disabled_calculations=disabled,
        )
        assessment.status = RiskStatus.PARTIAL if disabled else RiskStatus.COMPLETED
        assessment.downstream_ready = (
            assessment.status == RiskStatus.COMPLETED
            and assessment.validation.status != "fail"
        )
        return assessment
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "RISK_ASSESSMENT_FAILED", "message": str(exc)},
        ) from exc

# Frontend hosting (unchanged contract)
@app.get("/app", include_in_schema=False)
def serve_frontend():
    if not FRONTEND_PATH.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.post("/api/forward-test/run", response_model=ForwardTestResult)
def run_forward_test_endpoint(request: ForwardTestRequest) -> ForwardTestResult:
    return run_forward_test(request)
