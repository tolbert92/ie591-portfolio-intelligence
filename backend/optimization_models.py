
"""
IE 591 — Portfolio Intelligence
University of Tennessee

Skill #3 — Portfolio Optimization
Gate 4.1 — Data Contracts

Defines validated input and output schemas for
deterministic, long-only portfolio optimization.

No optimization calculations are performed here.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


CALCULATION_VERSION = "1.0.0"

WEIGHT_TOLERANCE = 1e-8
MATRIX_TOLERANCE = 1e-8


# -------------------------------------------------
# Enumerations
# -------------------------------------------------

class OptimizationObjective(str, Enum):
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"


class OptimizationStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    REJECTED = "rejected"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# -------------------------------------------------
# Shared Types
# -------------------------------------------------

FiniteNumber = Annotated[
    float,
    Field(allow_inf_nan=False),
]

NonnegativeFinite = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]

Weight = Annotated[
    float,
    Field(ge=0, le=1, allow_inf_nan=False),
]


class ContractModel(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


# -------------------------------------------------
# Input Contracts
# -------------------------------------------------

class OptimizationAsset(ContractModel):
    """
    Asset-level optimization inputs.

    Expected return and covariance must use
    the same annualization convention.
    """

    ticker: str = Field(
        min_length=1,
        max_length=30,
    )

    asset_name: str = Field(
        min_length=1,
        max_length=150,
    )

    asset_class: str = Field(
        min_length=1,
        max_length=80,
    )

    expected_annual_return: FiniteNumber

    minimum_weight: Weight = 0.0

    maximum_weight: Weight = 1.0

    @model_validator(mode="after")
    def validate_weight_bounds(self):
        if self.minimum_weight > self.maximum_weight:
            raise ValueError(
                "minimum_weight cannot exceed maximum_weight."
            )

        return self


class OptimizationConstraints(ContractModel):
    """
    Portfolio-level constraints.

    This version permits long-only portfolios
    whose weights sum to 100%.
    """

    fully_invested: Literal[True] = True

    allow_short_selling: Literal[False] = False

    allow_leverage: Literal[False] = False

    maximum_annual_volatility: (
        NonnegativeFinite | None
    ) = None

    minimum_expected_annual_return: (
        FiniteNumber | None
    ) = None


class OptimizationConfig(ContractModel):
    """Calculation and solver configuration."""

    objective: OptimizationObjective = (
        OptimizationObjective.MINIMUM_VARIANCE
    )

    risk_free_annual_rate: FiniteNumber = 0.0

    efficient_frontier_points: int = Field(
        default=25,
        ge=2,
        le=200,
    )

    solver_tolerance: float = Field(
        default=1e-8,
        gt=0,
        le=0.01,
        allow_inf_nan=False,
    )

    maximum_iterations: int = Field(
        default=1000,
        ge=100,
        le=100000,
    )


class OptimizationProvenance(ContractModel):
    """
    Records the origin of optimization inputs.

    The upstream analysis ID identifies the
    Market Intelligence calculation that
    supplied the statistical inputs.
    """

    source_type: Literal[
        "synthetic",
        "uploaded",
        "provider",
    ]

    source_name: str = Field(
        min_length=1,
        max_length=150,
    )

    dataset_id: str = Field(
        min_length=1,
        max_length=150,
    )

    upstream_analysis_id: str = Field(
        min_length=1,
        max_length=150,
    )

    is_synthetic: bool

    @model_validator(mode="after")
    def validate_synthetic_provenance(self):
        if (
            self.source_type == "synthetic"
            and not self.is_synthetic
        ):
            raise ValueError(
                "Synthetic source_type requires "
                "is_synthetic=true."
            )

        return self


class OptimizationRequest(ContractModel):
    """
    Complete request for portfolio optimization.

    covariance_matrix contains annualized
    covariance values in the same ticker order
    as the assets list.

    Market Intelligence data must be mapped
    into this contract before optimization.
    """

    optimization_id: str = Field(
        min_length=1,
        max_length=150,
    )

    assets: list[OptimizationAsset] = Field(
        min_length=2,
        max_length=100,
    )

    covariance_matrix: list[list[FiniteNumber]]

    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints
    )

    config: OptimizationConfig = Field(
        default_factory=OptimizationConfig
    )

    provenance: OptimizationProvenance

    @model_validator(mode="after")
    def validate_request(self):
        tickers = [
            asset.ticker for asset in self.assets
        ]

        if len(set(tickers)) != len(tickers):
            raise ValueError(
                "Asset tickers must be unique."
            )

        count = len(self.assets)
        matrix = self.covariance_matrix

        if len(matrix) != count:
            raise ValueError(
                "Covariance matrix row count must "
                "match the number of assets."
            )

        for row in matrix:
            if len(row) != count:
                raise ValueError(
                    "Covariance matrix must be square "
                    "and match the asset count."
                )

        for i in range(count):
            if matrix[i][i] < 0:
                raise ValueError(
                    "Covariance diagonal values "
                    "cannot be negative."
                )

            for j in range(i + 1, count):
                if not math.isclose(
                    matrix[i][j],
                    matrix[j][i],
                    rel_tol=MATRIX_TOLERANCE,
                    abs_tol=MATRIX_TOLERANCE,
                ):
                    raise ValueError(
                        "Covariance matrix must be symmetric."
                    )

        minimum_total = sum(
            asset.minimum_weight
            for asset in self.assets
        )

        maximum_total = sum(
            asset.maximum_weight
            for asset in self.assets
        )

        if minimum_total > 1 + WEIGHT_TOLERANCE:
            raise ValueError(
                "Sum of minimum asset weights "
                "cannot exceed 100%."
            )

        if maximum_total < 1 - WEIGHT_TOLERANCE:
            raise ValueError(
                "Sum of maximum asset weights "
                "must permit a fully invested portfolio."
            )

        return self


# -------------------------------------------------
# Output Contracts
# -------------------------------------------------

class OptimizationIssue(ContractModel):
    """Validation or solver diagnostic."""

    code: str = Field(min_length=1)

    severity: IssueSeverity

    message: str = Field(min_length=1)

    ticker: str | None = None


class OptimizedAllocation(ContractModel):
    """Final weight assigned to an asset."""

    ticker: str = Field(min_length=1)

    weight: Weight


class PortfolioMetrics(ContractModel):
    """Annualized portfolio performance metrics."""

    expected_annual_return: FiniteNumber

    annualized_volatility: NonnegativeFinite

    annualized_variance: NonnegativeFinite

    sharpe_ratio: FiniteNumber | None = None


class EfficientFrontierPoint(ContractModel):
    """One calculated point on the efficient frontier."""

    target_annual_return: FiniteNumber

    expected_annual_return: FiniteNumber

    annualized_volatility: NonnegativeFinite

    allocations: list[OptimizedAllocation] = Field(
        min_length=2
    )


class OptimizationQuality(ContractModel):
    """Optimization quality and feasibility report."""

    status: Literal[
        "pass",
        "warning",
        "fail",
    ]

    constraints_satisfied: bool

    weights_sum_to_one: bool

    covariance_matrix_valid: bool

    issues: list[OptimizationIssue] = Field(
        default_factory=list
    )


class OptimizationResponse(ContractModel):
    """
    Standardized response from Skill #3.

    Rejected analyses may return no allocation,
    metrics, or efficient-frontier points.
    """

    optimization_id: str

    status: OptimizationStatus

    objective: OptimizationObjective

    provenance: OptimizationProvenance

    allocations: list[OptimizedAllocation] = Field(
        default_factory=list
    )

    portfolio_metrics: PortfolioMetrics | None = None

    efficient_frontier: list[
        EfficientFrontierPoint
    ] = Field(default_factory=list)

    quality: OptimizationQuality

    calculation_version: str = CALCULATION_VERSION

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    downstream_ready: bool = False

    @model_validator(mode="after")
    def validate_response(self):
        if self.status == OptimizationStatus.REJECTED:
            if self.downstream_ready:
                raise ValueError(
                    "Rejected optimization cannot "
                    "be downstream ready."
                )

            if self.allocations or self.portfolio_metrics:
                raise ValueError(
                    "Rejected optimization cannot "
                    "contain a final portfolio."
                )

            if self.efficient_frontier:
                raise ValueError(
                    "Rejected optimization cannot "
                    "contain frontier results."
                )

        if self.downstream_ready:
            if self.status == OptimizationStatus.REJECTED:
                raise ValueError(
                    "Rejected results are not downstream ready."
                )

            if self.quality.status == "fail":
                raise ValueError(
                    "Failed quality cannot be downstream ready."
                )

            if not self.quality.constraints_satisfied:
                raise ValueError(
                    "Constraints must be satisfied "
                    "before downstream transfer."
                )

            if not self.quality.weights_sum_to_one:
                raise ValueError(
                    "Weights must sum to one "
                    "before downstream transfer."
                )

            if not self.quality.covariance_matrix_valid:
                raise ValueError(
                    "Covariance must be valid "
                    "before downstream transfer."
                )

            if not self.allocations:
                raise ValueError(
                    "Downstream-ready results "
                    "require allocations."
                )

            if self.portfolio_metrics is None:
                raise ValueError(
                    "Downstream-ready results "
                    "require portfolio metrics."
                )

            total = sum(
                allocation.weight
                for allocation in self.allocations
            )

            if not math.isclose(
                total,
                1.0,
                rel_tol=WEIGHT_TOLERANCE,
                abs_tol=WEIGHT_TOLERANCE,
            ):
                raise ValueError(
                    "Downstream-ready allocation "
                    "weights must sum to one."
                )

        return self
