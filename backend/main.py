
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Stage 3 — Market Intelligence Integration

Skill #1: Investor Profile
Skill #2: Market Intelligence

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


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description=(
        "Educational portfolio intelligence "
        "platform integrating ten analytical skills."
    ),
    version="0.3.1",
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
        "status": "created",
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
