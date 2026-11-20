
"""Map IE 591 optimization and risk results into visualization datasets."""

from __future__ import annotations

from .optimization_models import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizedAllocation,
)
from .risk_models import (
    RiskExportPackage,
    RiskMeasure as SourceRiskMeasure,
)
from .visualization_models import (
    AssetAllocation,
    AssetRiskContribution,
    EfficientFrontier,
    EfficientFrontierPoint,
    PortfolioAllocation,
    PortfolioMetrics,
    PortfolioRisk,
    RiskMeasure,
    StressContribution,
    StressScenarioResult,
    VisualizationDataset,
    VisualizationDiagnostic,
    VisualizationProvenance,
)


def from_optimization(
    request: OptimizationRequest,
    result: OptimizationResponse,
) -> VisualizationDataset:
    """Map a downstream-ready optimization and its matching request."""

    if request.optimization_id != result.optimization_id:
        raise ValueError(
            "Optimization request and result IDs must match"
        )

    if request.provenance != result.provenance:
        raise ValueError(
            "Optimization request and result provenance must match"
        )

    if not result.downstream_ready or result.portfolio_metrics is None:
        raise ValueError(
            "Visualization requires a downstream-ready optimization"
        )

    portfolio_id = result.optimization_id

    provenance = VisualizationProvenance(
        **result.provenance.model_dump(),
        source_skill="Portfolio Optimization",
        analysis_id=portfolio_id,
        calculation_version=result.calculation_version,
        generated_at=result.generated_at,
    )

    assets = {
        asset.ticker: asset
        for asset in request.assets
    }

    def allocations(
        rows: list[OptimizedAllocation],
    ) -> tuple[AssetAllocation, ...]:
        unknown = {
            row.ticker for row in rows
        }.difference(assets)

        if unknown:
            raise ValueError(
                f"Unknown optimization tickers: {sorted(unknown)}"
            )

        return tuple(
            AssetAllocation(
                ticker=row.ticker,
                weight=row.weight,
                asset_name=assets[row.ticker].asset_name,
                asset_class=assets[row.ticker].asset_class,
            )
            for row in rows
        )

    portfolio = PortfolioAllocation(
        portfolio_id=portfolio_id,
        allocations=allocations(result.allocations),
        portfolio_metrics=PortfolioMetrics.model_validate(
            result.portfolio_metrics.model_dump()
        ),
        provenance=provenance,
    )

    points = tuple(
        EfficientFrontierPoint(
            target_annual_return=point.target_annual_return,
            expected_annual_return=point.expected_annual_return,
            annualized_volatility=point.annualized_volatility,
            allocations=allocations(point.allocations),
        )
        for point in result.efficient_frontier
    )

    frontiers = (
        (
            EfficientFrontier(
                portfolio_id=portfolio_id,
                points=points,
                provenance=provenance,
            ),
        )
        if points
        else ()
    )

    return VisualizationDataset(
        portfolios=(portfolio,),
        efficient_frontiers=frontiers,
        diagnostics=tuple(
            VisualizationDiagnostic(
                severity=issue.severity.value,
                code=issue.code,
                message=issue.message,
            )
            for issue in result.quality.issues
        ),
    )


def from_risk(
    export: RiskExportPackage,
) -> VisualizationDataset:
    """Map a risk export, including partial or blocked results and diagnostics."""

    request = export.input
    result = export.assessment

    if (
        request.portfolio.provenance.is_synthetic
        != result.provenance.is_synthetic
    ):
        raise ValueError(
            "Risk input and result synthetic flags must match"
        )

    portfolio_id = request.portfolio.portfolio_id

    provenance_fields = {
        "source_skill": "Risk Assessment",
        "analysis_id": result.assessment_id,
        "calculation_version": result.calculation_version,
        "generated_at": result.generated_at,
    }

    portfolio = PortfolioAllocation(
        portfolio_id=portfolio_id,
        allocations=tuple(
            AssetAllocation.model_validate(
                asset.model_dump(
                    exclude={"expected_annual_return"}
                )
            )
            for asset in request.portfolio.assets
        ),
        provenance=VisualizationProvenance(
            **request.portfolio.provenance.model_dump(),
            **provenance_fields,
        ),
    )

    scenarios = {
        scenario.scenario_id: scenario
        for scenario in request.stress_scenarios
    }

    stress_results = []

    for row in result.stress_results:
        scenario = scenarios.get(row.scenario_id)

        if scenario is None:
            raise ValueError(
                f"Unknown risk stress scenario ID: {row.scenario_id!r}"
            )

        stress_results.append(
            StressScenarioResult(
                scenario_id=row.scenario_id,
                name=scenario.name,
                scenario_type=scenario.scenario_type,
                portfolio_shock_return=row.portfolio_shock_return,
                hypothetical=row.hypothetical,
                contributions=tuple(
                    StressContribution(
                        ticker=ticker,
                        contribution=value,
                    )
                    for ticker, value in row.contributions.items()
                ),
            )
        )

    def measure(
        row: SourceRiskMeasure | None,
    ) -> RiskMeasure | None:
        if row is None:
            return None

        return RiskMeasure(
            value=row.value,
            unit=row.unit,
            method=row.method.value,
            horizon=row.horizon.value,
            confidence_level=row.confidence_level,
        )

    risk = PortfolioRisk(
        portfolio_id=portfolio_id,
        status=result.status.value,
        annualized_volatility=result.annualized_volatility,
        value_at_risk=measure(result.value_at_risk),
        conditional_value_at_risk=measure(
            result.conditional_value_at_risk
        ),
        asset_risk_contributions=tuple(
            AssetRiskContribution.model_validate(
                row.model_dump()
            )
            for row in result.asset_risk_contributions
        ),
        stress_results=tuple(stress_results),
        provenance=VisualizationProvenance(
            **result.provenance.model_dump(),
            **provenance_fields,
        ),
        diagnostics=tuple(
            VisualizationDiagnostic(
                severity=issue.severity.value,
                code=issue.code,
                message=issue.message,
                field_path=issue.field_path,
            )
            for issue in result.validation.issues
        ),
    )

    return VisualizationDataset(
        portfolios=(portfolio,),
        risk_results=(risk,),
    )


__all__ = [
    "from_optimization",
    "from_risk",
]
