
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #1 — Investor Profile
Skill #2 — Market Intelligence
Skill #3 — Portfolio Optimization

Gate 4.3 — Backend API Integration

Educational portfolio analytics application.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import InvestorProfile

from .market_models import (
    MarketAnalysisRequest,
    MarketAnalysisResponse,
)

from .market_engine import (
    CALCULATION_VERSION,
    analyze_market,
)

from .optimization_models import (
    CALCULATION_VERSION as OPTIMIZATION_VERSION,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationStatus,
    OptimizationQuality,
    OptimizationIssue,
    IssueSeverity,
    OptimizedAllocation,
    EfficientFrontierPoint,
)

from .optimization_solver import (
    OptimizationSolverError,
    solve_portfolio,
)

from .optimization_frontier import (
    EfficientFrontierError,
    generate_efficient_frontier,
)


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description=(
        "Educational portfolio intelligence "
        "platform integrating ten analytical skills."
    ),
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "index.html"


# ==========================================
# SKILL REGISTRY
# ==========================================

SKILLS = [
    {
        "id": 1,
        "name": "Investor Profile",
        "status": "integration_in_progress",
    },
    {
        "id": 2,
        "name": "Market Intelligence",
        "status": "integration_in_progress",
    },
    {
        "id": 3,
        "name": "Portfolio Optimization",
        "status": "integration_in_progress",
    },
    {
        "id": 4,
        "name": "Risk Assessment",
        "status": "pending_integration",
    },
    {
        "id": 5,
        "name": "Portfolio Visualization",
        "status": "pending_integration",
    },
    {
        "id": 6,
        "name": "3D Portfolio Visualization",
        "status": "pending_integration",
    },
    {
        "id": 7,
        "name": "Forward Testing",
        "status": "pending_integration",
    },
    {
        "id": 8,
        "name": "Monte Carlo Simulation",
        "status": "pending_integration",
    },
    {
        "id": 9,
        "name": "Strategy Comparison",
        "status": "pending_integration",
    },
    {
        "id": 10,
        "name": "Decision Support",
        "status": "pending_integration",
    },
]


# ==========================================
# TEMPORARY INVESTOR PROFILE STORAGE
# ==========================================

# Educational prototype only.
# Profiles are not persistent across restarts.

investor_profiles: dict[str, InvestorProfile] = {}


# ==========================================
# ROOT
# ==========================================

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


# ==========================================
# HEALTH
# ==========================================

@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "portfolio-intelligence-api",
        "version": app.version,
    }


# ==========================================
# SKILL REGISTRY
# ==========================================

@app.get("/api/skills")
def get_skills() -> dict[str, Any]:
    return {
        "total": len(SKILLS),
        "integrated": sum(
            skill["status"] == "integrated"
            for skill in SKILLS
        ),
        "skills": SKILLS,
    }


# ==========================================
# APPLICATION STATUS
# ==========================================

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


# ==========================================
# SKILL #1 — INVESTOR PROFILE
# ==========================================

@app.post(
    "/api/investor-profile",
    status_code=status.HTTP_201_CREATED,
)
def create_investor_profile(
    profile: InvestorProfile,
) -> dict[str, Any]:

    investor_id = profile.investor_id

    if investor_id in investor_profiles:
        raise HTTPException(
            status_code=409,
            detail=(
                "An investor profile with this ID "
                "already exists."
            ),
        )

    investor_profiles[investor_id] = profile

    return {
        "status": "created",
        "investor_id": investor_id,
        "message": "Investor profile created.",
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }


@app.get("/api/investor-profile/{investor_id}")
def get_investor_profile(
    investor_id: str,
) -> dict[str, Any]:

    profile = investor_profiles.get(investor_id)

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Investor profile not found.",
        )

    return {
        "status": "found",
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }


# ==========================================
# SKILL #2 — MARKET INTELLIGENCE
# ==========================================

@app.get("/api/market/status")
def market_engine_status() -> dict[str, Any]:
    return {
        "skill_id": 2,
        "skill_name": "Market Intelligence",
        "status": "integration_in_progress",
        "engine": "available",
        "calculation_version": CALCULATION_VERSION,
        "input_type": "historical_price_observations",
        "data_sources": [
            "synthetic",
            "uploaded",
            "provider",
        ],
        "external_data_fetching": False,
        "api_endpoint": "/api/market/analyze",
        "method": "POST",
    }


@app.post(
    "/api/market/analyze",
    response_model=MarketAnalysisResponse,
)
def analyze_market_endpoint(
    request: MarketAnalysisRequest,
) -> MarketAnalysisResponse:

    return analyze_market(request)


# ==========================================
# SKILL #3 — PORTFOLIO OPTIMIZATION
# ==========================================

@app.get("/api/optimization/status")
def optimization_engine_status() -> dict[str, Any]:
    """
    Describe available optimization capabilities.
    """
    return {
        "skill_id": 3,
        "skill_name": "Portfolio Optimization",
        "status": "integration_in_progress",
        "engine": "available",
        "calculation_version": OPTIMIZATION_VERSION,
        "objectives": [
            "minimum_variance",
            "maximum_sharpe",
        ],
        "constraints": [
            "fully_invested",
            "long_only",
            "asset_weight_bounds",
            "minimum_expected_annual_return",
            "maximum_annual_volatility",
        ],
        "efficient_frontier": True,
        "data_sources": [
            "synthetic",
            "uploaded",
            "provider",
        ],
        "external_data_fetching": False,
        "endpoints": {
            "solve": "/api/optimization/solve",
            "frontier": "/api/optimization/frontier",
        },
    }


def _optimization_quality() -> OptimizationQuality:
    """
    Construct a passing quality report.

    Called only after the numerical solver has
    successfully validated its final solution.
    """
    return OptimizationQuality(
        status="pass",
        constraints_satisfied=True,
        weights_sum_to_one=True,
        covariance_matrix_valid=True,
        issues=[],
    )


@app.post(
    "/api/optimization/solve",
    response_model=OptimizationResponse,
)
def optimize_portfolio_endpoint(
    request: OptimizationRequest,
) -> OptimizationResponse:
    """
    Execute constrained portfolio optimization.

    Pydantic validates the request structure.

    The solver validates covariance properties,
    feasibility, allocation bounds, and numerical
    convergence.

    Infeasible optimization requests return
    HTTP 422 with a descriptive error.
    """
    try:
        result = solve_portfolio(request)

    except OptimizationSolverError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "OPTIMIZATION_FAILED",
                "message": str(exc),
            },
        ) from exc

    allocations = [
        OptimizedAllocation(
            ticker=asset.ticker,
            weight=float(weight),
        )
        for asset, weight in zip(
            request.assets,
            result.weights,
        )
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


@app.post(
    "/api/optimization/frontier",
    response_model=list[EfficientFrontierPoint],
)
def generate_frontier_endpoint(
    request: OptimizationRequest,
) -> list[EfficientFrontierPoint]:
    """
    Generate the efficient frontier using the
    configured efficient_frontier_points value.

    Each point includes:
    - Target annual return
    - Calculated annual return
    - Annualized volatility
    - Asset allocations

    Existing allocation and risk constraints
    are preserved by the frontier engine.

    Unreachable targets return HTTP 422.
    """
    try:
        return generate_efficient_frontier(request)

    except (
        EfficientFrontierError,
        OptimizationSolverError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "FRONTIER_GENERATION_FAILED",
                "message": str(exc),
            },
        ) from exc


# ==========================================
# FRONTEND HOSTING
# ==========================================

@app.get(
    "/app",
    include_in_schema=False,
)
def serve_frontend():

    if not FRONTEND_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found.",
        )

    return FileResponse(
        FRONTEND_PATH,
        media_type="text/html",
    )
