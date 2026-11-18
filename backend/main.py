
"""
IE 591 — AI Portfolio Optimization
Unified Application Backend

Stage 2: Skill #1 — Investor Profile Integration
Version: 0.2.0

Development API with temporary in-memory profile storage.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import InvestorProfile


app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description=(
        "Unified backend for the ten-skill AI Portfolio "
        "Optimization application."
    ),
    version="0.2.0",
)

# Development-only CORS settings.
# Restrict origins before public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

SKILLS = [
    (1, "Investor Profile"),
    (2, "Market Intelligence"),
    (3, "Portfolio Optimization"),
    (4, "Risk Assessment"),
    (5, "Portfolio Visualization"),
    (6, "3D Portfolio Visualization"),
    (7, "Forward Testing"),
    (8, "Monte Carlo Simulation"),
    (9, "Strategy Comparison"),
    (10, "Decision Support"),
]

# Temporary development storage.
# Data is lost when the Python process restarts.
# Do not use for sensitive or production investor records.
investor_profiles: dict[str, InvestorProfile] = {}


@app.get("/")
def root():
    return {
        "application": "IE 591 Portfolio Intelligence",
        "version": "0.2.0",
        "status": "development",
        "message": "Unified portfolio API foundation",
        "skills_total": 10,
        "skills_integrated": 0,
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "portfolio-intelligence-api",
        "version": "0.2.0",
    }


@app.get("/api/skills")
def list_skills():
    return {
        "total": len(SKILLS),
        "integrated": 0,
        "skills": [
            {
                "id": number,
                "name": name,
                "status": "integration_in_progress"
                if number == 1
                else "pending_integration",
            }
            for number, name in SKILLS
        ],
    }


@app.get("/api/status")
def application_status():
    return {
        "stage": 2,
        "stage_name": "Skill #1 Integration",
        "frontend": "prototype_approved",
        "backend": "investor_profile_api",
        "analytics": "not_connected",
        "skills_integrated": 0,
        "professor_validation": "pending",
    }


@app.post("/api/investor-profile", status_code=201)
def create_investor_profile(profile: InvestorProfile):
    """
    Validate and temporarily store an investor profile.

    Pydantic automatically rejects invalid request data.
    """

    if profile.investor_id in investor_profiles:
        raise HTTPException(
            status_code=409,
            detail="An investor profile with this ID already exists.",
        )

    investor_profiles[profile.investor_id] = profile

    return {
        "status": "created",
        "message": "Investor profile validated and stored.",
        "investor_id": profile.investor_id,
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }


@app.get("/api/investor-profile/{investor_id}")
def get_investor_profile(investor_id: str):
    """
    Retrieve a previously submitted investor profile.
    """

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
