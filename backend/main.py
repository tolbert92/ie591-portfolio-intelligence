
"""
IE 591 — AI Portfolio Optimization
Unified Application Backend

Stage 1: FastAPI Foundation
Version: 0.1.0

This module provides the initial REST API for the
ten-skill integrated portfolio intelligence application.

Financial analysis engines are not connected yet.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="IE 591 Portfolio Intelligence API",
    description=(
        "Unified backend for the ten-skill AI Portfolio "
        "Optimization application."
    ),
    version="0.1.0",
)

# Development-only CORS configuration.
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
    allow_methods=["GET"],
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


@app.get("/")
def root():
    """Return basic application information."""
    return {
        "application": "IE 591 Portfolio Intelligence",
        "version": "0.1.0",
        "status": "development",
        "message": "Unified portfolio API foundation",
        "skills_total": 10,
        "skills_integrated": 0,
    }


@app.get("/api/health")
def health_check():
    """Confirm the API is responding."""
    return {
        "status": "healthy",
        "service": "portfolio-intelligence-api",
        "version": "0.1.0",
    }


@app.get("/api/skills")
def list_skills():
    """List the ten planned analytical modules."""
    return {
        "total": len(SKILLS),
        "integrated": 0,
        "skills": [
            {
                "id": number,
                "name": name,
                "status": "pending_integration",
            }
            for number, name in SKILLS
        ],
    }


@app.get("/api/status")
def application_status():
    """Return application implementation status."""
    return {
        "stage": 1,
        "stage_name": "Python Backend Foundation",
        "frontend": "prototype_approved",
        "backend": "foundation",
        "analytics": "not_connected",
        "skills_integrated": 0,
        "professor_validation": "pending",
    }
