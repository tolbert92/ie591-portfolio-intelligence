"""IE 591 Skill #5 — stateless visualization API router.

Include ``router`` in the existing FastAPI application during Stage 5.3.
No source portfolio is mutated; no investment recommendations are generated.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .optimization_models import OptimizationRequest, OptimizationResponse
from .risk_models import RiskExportPackage
from .visualization_adapters import from_optimization, from_risk
from .visualization_comparison import (
    VisualizationComparisonError,
    compare_visualization_portfolios,
)
from .visualization_engine import (
    VisualizationEngineError,
    build_visualization_charts,
)
from .visualization_models import VisualizationDataset
from .visualization_validation import validate_visualization_dataset

router = APIRouter(prefix="/api/visualization", tags=["Skill 5 — Portfolio Visualization"])


class OptimizationHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: OptimizationRequest
    result: OptimizationResponse


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: VisualizationDataset
    portfolio_ids: list[str] = Field(min_length=2)


def _invalid(exc: Exception) -> HTTPException:
    """Return a safe client error without exposing stack traces."""
    return HTTPException(status_code=422, detail=str(exc))


def _validated_charts(dataset: VisualizationDataset) -> dict[str, Any]:
    report = validate_visualization_dataset(dataset)
    if report.has_errors:
        raise HTTPException(status_code=422, detail=report.to_dict())
    try:
        return build_visualization_charts(dataset, report=report)
    except (VisualizationEngineError, ValidationError, ValueError) as exc:
        raise _invalid(exc) from exc


@router.get("/status")
def visualization_status() -> dict[str, Any]:
    """Report router availability, not upstream provider availability."""
    return {"skill": 5, "name": "Portfolio Visualization", "api_status": "available"}


@router.post("/validate")
def validate_dataset(dataset: VisualizationDataset) -> dict[str, Any]:
    """Return tiered diagnostics and per-chart availability."""
    return validate_visualization_dataset(dataset).to_dict()


@router.post("/charts")
def visualization_charts(dataset: VisualizationDataset) -> dict[str, Any]:
    """Generate renderer-neutral chart data from a validated dataset."""
    return _validated_charts(dataset)


@router.post("/from-optimization")
def charts_from_optimization(payload: OptimizationHandoff) -> dict[str, Any]:
    """Validate matching Skill #3 inputs/results and render charts."""
    try:
        dataset = from_optimization(payload.request, payload.result)
    except (ValidationError, ValueError, TypeError) as exc:
        raise _invalid(exc) from exc
    return _validated_charts(dataset)


@router.post("/from-risk")
def charts_from_risk(export: RiskExportPackage) -> dict[str, Any]:
    """Render available Skill #4 risk results, including partial outputs."""
    try:
        dataset = from_risk(export)
    except (ValidationError, ValueError, TypeError) as exc:
        raise _invalid(exc) from exc
    return _validated_charts(dataset)


@router.post("/compare")
def compare_portfolios(payload: ComparisonRequest) -> dict[str, Any]:
    """Compare two or more portfolios without recalculating source results."""
    report = validate_visualization_dataset(payload.dataset)
    if report.has_errors:
        raise HTTPException(status_code=422, detail=report.to_dict())
    try:
        return compare_visualization_portfolios(
            payload.dataset,
            payload.portfolio_ids,
            report=report,
        )
    except (VisualizationComparisonError, ValidationError, ValueError) as exc:
        raise _invalid(exc) from exc


__all__ = ["router", "OptimizationHandoff", "ComparisonRequest"]
