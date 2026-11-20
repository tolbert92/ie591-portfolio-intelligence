"""IE 591 Skill #4 — Stage 4.3.2: explainable risk classification.

Pure, deterministic analytics using the approved Stage 4.1 Pydantic contracts.
All returns and volatility thresholds are decimal fractions. Thresholds are
illustrative and configurable, NOT regulatory standards or investment advice.

The four risk dimensions are annualized volatility, horizon-specific CVaR,
asset concentration, and worst hypothetical stress loss. Missing dimensions
are INDETERMINATE, never LOW. Overall classification uses the highest known
risk level and is INDETERMINATE if no dimensions can be assessed. Missing
inputs are explicitly explained and never silently invented.
"""
from __future__ import annotations

from math import isfinite
from typing import Iterable

try:
    from .risk_models import (
        AlignmentStatus, InvestorAlignment, InvestorRiskProfile,
        RiskAssessmentRequest, RiskAssessmentResult, RiskClassification,
        RiskLevel, RiskPortfolio, RiskThresholds, StressResult,
    )
except ImportError:  # Supports execution from the backend directory.
    from risk_models import (
        AlignmentStatus, InvestorAlignment, InvestorRiskProfile,
        RiskAssessmentRequest, RiskAssessmentResult, RiskClassification,
        RiskLevel, RiskPortfolio, RiskThresholds, StressResult,
    )

CLASSIFICATION_VERSION = "1.0.0"

# Ordered from least to greatest risk. INDETERMINATE is not an ordinal level.
_RISK_ORDER = (RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH)
_RISK_RANK = {level: index for index, level in enumerate(_RISK_ORDER)}

# The concentration threshold is configurable. The following multiples and
# stress/CVaR cutoffs are transparent, illustrative Stage 4.3.2 conventions.
# Stress and CVaR are loss magnitudes expressed as fractions of portfolio value.
# Stress results are hypothetical and are not assigned probabilities.
_DOWNSIDE_CUTOFFS = (0.02, 0.05, 0.10)
_STRESS_CUTOFFS = (0.05, 0.15, 0.30)


def _band(value: float, cutoffs: tuple[float, float, float]) -> RiskLevel:
    """Use strict greater-than boundaries: exactly at a cutoff stays lower."""
    if not isfinite(value) or value < 0:
        raise ValueError("Risk metric must be finite and nonnegative")
    if value <= cutoffs[0]:
        return RiskLevel.LOW
    if value <= cutoffs[1]:
        return RiskLevel.MODERATE
    if value <= cutoffs[2]:
        return RiskLevel.HIGH
    return RiskLevel.VERY_HIGH


def asset_concentration(portfolio: RiskPortfolio) -> dict[str, float]:
    """Return validated asset weights, preserving portfolio asset order."""
    return {asset.ticker: asset.weight for asset in portfolio.assets}


def asset_class_concentration(portfolio: RiskPortfolio) -> dict[str, float]:
    """Aggregate asset weights by normalized, nonempty asset-class labels."""
    totals: dict[str, float] = {}
    for asset in portfolio.assets:
        label = asset.asset_class.strip().lower().replace("-", "_").replace(" ", "_")
        label = label or "unclassified"
        totals[label] = totals.get(label, 0.0) + asset.weight
    return totals


def concentration_indicators(portfolio: RiskPortfolio) -> dict[str, object]:
    """Compute descriptive concentration metrics; no thresholds applied here.

    Herfindahl-Hirschman indices are the sums of squared portfolio shares.
    A single-asset portfolio has an asset HHI of 1.0. These metrics describe
    allocation concentration, not covariance-based risk contribution.
    """
    assets = asset_concentration(portfolio)
    classes = asset_class_concentration(portfolio)
    return {
        "asset_weights": assets,
        "asset_class_weights": classes,
        "largest_asset_weight": max(assets.values()),
        "largest_asset_class_weight": max(classes.values()),
        "asset_hhi": sum(weight * weight for weight in assets.values()),
        "asset_class_hhi": sum(weight * weight for weight in classes.values()),
        "effective_asset_count": 1.0 / sum(weight * weight for weight in assets.values()),
    }


def _concentration_band(largest_weight: float, threshold: float) -> RiskLevel:
    """Use the user-configurable HIGH cutoff and explicit nearby bands.

    LOW <= 0.5 * threshold; MODERATE <= threshold; HIGH <= min(1, 1.5 *
