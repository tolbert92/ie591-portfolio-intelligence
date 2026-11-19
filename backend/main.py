
"""
IE 591 — Portfolio Intelligence
University of Tennessee

FastAPI application
Stage 3 — Skill #2 Market Intelligence Integration

Skill #1: Investor Profile
Skill #2: Market Intelligence

Educational application using synthetic datasets.
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
from .market_engine import analyze_market


# ------------------------------------------
# APPLICATION CONFIGURATION
# ------------------------------------------

app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description=(
        "Educational portfolio intelligence platform "
        "integrating ten analytical skills."
    ),
    version="0.3.0",
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


# ------------------------------------------
# APPLICATION PATHS
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "index.html"


# ------------------------------------------
# SKILL REGISTRY
# ------------------------------------------

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
        "status": "pending_integration",
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


# ------------------------------------------
# TEMPORARY STORAGE
# ------------------------------------------

# Investor profiles are stored in memory only.
# Data will be lost when the server restarts.
#
# This remains unchanged from Skill #1.

investor_profiles: dict[str, InvestorProfile] = {}


# ------------------------------------------
# ROOT ENDPOINT
# ------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "application": "IE 591 Portfolio Intelligence",
        "service": "portfolio-intelligence-api",
        "version": app.version,
        "status": "operational",
        "skills_total": 10,
        "skills_integrated": 0,
        "frontend_url": "/app",
        "documentation_url": "/docs",
    }


# ------------------------------------------
# HEALTH ENDPOINT
# ------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "portfolio-intelligence-api",
        "version": app.version,
    }


# ------------------------------------------
# SKILL REGISTRY ENDPOINT
# ------------------------------------------

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


# ------------------------------------------
# INTEGRATION STATUS
# ------------------------------------------

@app.get("/api/status")
def integration_status() -> dict[str, Any]:
    return {
        "stage": 3,
        "stage_name": "Skill #2 Integration",
        "frontend": "investor_profile_connected",
        "backend": "market_intelligence_api",
        "analytics": "market_engine_connected",
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
        "message": "Investor profile created.",
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }


@app.get(
    "/api/investor-profile/{investor_id}",
)
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
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }


# ==========================================
# SKILL #2 — MARKET INTELLIGENCE
# ==========================================

@app.get("/api/market/status")
def market_engine_status() -> dict[str, Any]:
    """
    Report Skill #2 engine availability.

    Engine availability does not mean that
    browser acceptance testing is complete.
    """

    return {
        "skill_id": 2,
        "skill_name": "Market Intelligence",
        "status": "integration_in_progress",
        "engine": "available",
        "calculation_version": "1.0.0",
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
    """
    Analyze supplied historical market prices.

    Processing:
      1. Validate request with Pydantic.
      2. Align historical observations.
      3. Calculate returns and volatility.
      4. Calculate covariance and correlation.
      5. Return quality findings and statistics.

    Invalid request structure returns HTTP 422.

    Analytically unsuitable but structurally
    valid datasets return a structured result
    with status='rejected'.

    No market prices are fetched externally.
    """

    return analyze_market(request)


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
