
"""
IE 591 — AI Portfolio Optimization
Unified Application Backend

Stage 2: Skill #1 — Investor Profile Integration
Version: 0.2.0

Features:
- FastAPI application
- Health and status endpoints
- Ten-skill registry
- Investor profile validation
- Investor profile creation and retrieval
- Temporary in-memory profile storage
- HTML frontend hosting at /app

Educational development application.
Use synthetic investor data only.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.models import InvestorProfile


# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description=(
        "Unified backend for the ten-skill AI Portfolio "
        "Optimization application."
    ),
    version="0.2.0",
)


# ==================================================
# CORS CONFIGURATION
# ==================================================

# Development origins only.
# The deployed /app frontend uses the same origin
# as the API once its URL is updated.

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


# ==================================================
# SKILL REGISTRY
# ==================================================

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


# ==================================================
# TEMPORARY INVESTOR PROFILE STORAGE
# ==================================================

# Development-only storage.
#
# Profiles are stored in the current Python process.
# Data disappears when the process restarts.
# Multiple server workers do not share this dictionary.
#
# Do not store actual personal or financial information.

investor_profiles: dict[str, InvestorProfile] = {}


# ==================================================
# APPLICATION ROOT
# ==================================================

@app.get("/")
def root():
    """
    Return application metadata.

    Preserved for compatibility with Stage 1 and
    Stage 2 automated tests.
    """

    return {
        "application": "IE 591 Portfolio Intelligence",
        "version": "0.2.0",
        "status": "development",
        "message": "Unified portfolio API foundation",
        "skills_total": 10,
        "skills_integrated": 0,
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health_check():
    """
    Confirm that the FastAPI backend is responding.
    """

    return {
        "status": "healthy",
        "service": "portfolio-intelligence-api",
        "version": "0.2.0",
    }


# ==================================================
# SKILL REGISTRY ENDPOINT
# ==================================================

@app.get("/api/skills")
def list_skills():
    """
    Return the status of all ten planned skills.

    Skill #1 is currently being integrated.
    Skills #2 through #10 remain pending.
    """

    return {
        "total": len(SKILLS),
        "integrated": 0,
        "skills": [
            {
                "id": number,
                "name": name,
                "status": (
                    "integration_in_progress"
                    if number == 1
                    else "pending_integration"
                ),
            }
            for number, name in SKILLS
        ],
    }


# ==================================================
# APPLICATION STATUS
# ==================================================

@app.get("/api/status")
def application_status():
    """
    Report the current development stage.
    """

    return {
        "stage": 2,
        "stage_name": "Skill #1 Integration",
        "frontend": "prototype_approved",
        "backend": "investor_profile_api",
        "analytics": "not_connected",
        "skills_integrated": 0,
        "professor_validation": "pending",
    }


# ==================================================
# SKILL #1 — CREATE INVESTOR PROFILE
# ==================================================

@app.post(
    "/api/investor-profile",
    status_code=201,
)
def create_investor_profile(profile: InvestorProfile):
    """
    Validate and temporarily store an investor profile.

    Pydantic validates incoming data automatically.

    Returns:
        201: Profile created
        409: Duplicate investor ID
        422: Invalid input data
    """

    if profile.investor_id in investor_profiles:
        raise HTTPException(
            status_code=409,
            detail=(
                "An investor profile with this ID "
                "already exists."
            ),
        )

    investor_profiles[profile.investor_id] = profile

    return {
        "status": "created",
        "message": (
            "Investor profile validated and stored."
        ),
        "investor_id": profile.investor_id,
        "profile": profile.model_dump(mode="json"),
        "storage": "temporary_memory",
    }


# ==================================================
# SKILL #1 — RETRIEVE INVESTOR PROFILE
# ==================================================

@app.get("/api/investor-profile/{investor_id}")
def get_investor_profile(investor_id: str):
    """
    Retrieve a previously submitted investor profile.

    Returns:
        200: Profile found
        404: Profile not found
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


# ==================================================
# STAGE 2 — HTML FRONTEND HOSTING
# ==================================================

# Resolve the repository root regardless of the
# directory from which the server is launched.

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_FILE = BASE_DIR / "index.html"


@app.get("/app", include_in_schema=False)
def serve_frontend():
    """
    Serve the Portfolio Intelligence HTML interface.

    Example:
        https://your-render-service.onrender.com/app

    The / endpoint remains JSON for test compatibility.
    """

    if not FRONTEND_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend application not found.",
        )

    return FileResponse(
        FRONTEND_FILE,
        media_type="text/html",
    )
