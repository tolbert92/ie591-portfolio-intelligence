
"""
IE 591 — AI Portfolio Optimization
Shared Data Models

Stage 1: Python Backend Foundation
Version: 0.1.0

Defines validated data structures for investor profiles,
portfolio allocations, financial goals, and analysis requests.

These models do not perform investment calculations.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unexpected input fields."""

    model_config = ConfigDict(extra="forbid")


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class GoalPriority(str, Enum):
    MANDATORY = "mandatory"
    HIGH = "high"
    DISCRETIONARY = "discretionary"
    ASPIRATIONAL = "aspirational"


class GoalType(str, Enum):
    RETIREMENT = "retirement"
    WEALTH_ACCUMULATION = "wealth_accumulation"
    EDUCATION = "education"
    CAPITAL_PRESERVATION = "capital_preservation"
    OTHER = "other"


NonNegativeMoney = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]

PositiveMoney = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False),
]


class FinancialGoal(StrictModel):
    """An investor's financial objective."""

    goal_id: str = Field(min_length=1, max_length=100)
    goal_type: GoalType
    target_amount: PositiveMoney
    target_year: int = Field(ge=2020, le=2200)
    priority: GoalPriority = GoalPriority.HIGH
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")


class InvestorProfile(StrictModel):
    """Validated investor information."""

    investor_id: str = Field(min_length=1, max_length=100)
    investor_name: str = Field(min_length=1, max_length=200)

    initial_investment: NonNegativeMoney
    annual_contribution: NonNegativeMoney = 0

    investment_horizon_years: int = Field(ge=1, le=70)
    risk_tolerance: RiskTolerance

    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")

    goals: list[FinancialGoal] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_goal_identifiers(self):
        """Prevent duplicate financial goal identifiers."""
        goal_ids = [goal.goal_id for goal in self.goals]

        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError(
                "Financial goal identifiers must be unique."
            )

        return self


class AssetAllocation(StrictModel):
    """One asset's allocation in a portfolio."""

    asset_id: str = Field(min_length=1, max_length=100)

    weight: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )


class Portfolio(StrictModel):
    """Portfolio structure and asset allocations."""

    portfolio_id: str = Field(min_length=1, max_length=100)

    portfolio_name: str = Field(min_length=1, max_length=200)

    allocations: list[AssetAllocation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_allocations(self):
        """Validate allocation uniqueness and total weight."""

        asset_ids = [
            allocation.asset_id
            for allocation in self.allocations
        ]

        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError(
                "Portfolio asset identifiers must be unique."
            )

        total_weight = sum(
            allocation.weight
            for allocation in self.allocations
        )

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "Portfolio allocation weights must total 1.0."
            )

        return self


class AnalysisRequest(StrictModel):
    """Shared request contract for future analytical skills."""

    schema_version: str = Field(default="1.0.0")

    request_id: str = Field(min_length=1, max_length=100)

    investor: InvestorProfile

    portfolios: list[Portfolio] = Field(default_factory=list)


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisResponse(StrictModel):
    """Standard status response for analytical operations."""

    request_id: str

    status: AnalysisStatus

    message: str

    evidence_ids: list[str] = Field(default_factory=list)
