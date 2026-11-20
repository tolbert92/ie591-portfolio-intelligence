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
    threshold); VERY_HIGH above that. At threshold=1, VERY_HIGH is impossible.
    This is an illustrative rubric, not a claim about portfolio suitability.
    """
    if largest_weight <= threshold * 0.5:
        return RiskLevel.LOW
    if largest_weight <= threshold:
        return RiskLevel.MODERATE
    if largest_weight <= min(1.0, threshold * 1.5):
        return RiskLevel.HIGH
    return RiskLevel.VERY_HIGH


def classify_portfolio_risk(
    request: RiskAssessmentRequest,
    assessment: RiskAssessmentResult,
    stress_results: Iterable[StressResult] | None = None,
) -> RiskClassification:
    """Classify four dimensions without inferring unavailable observations.

    Annual volatility is always evaluated when present. Downside classification
    uses the calculated CVaR for the SELECTED horizon and confidence level;
    the fixed cutoffs are illustrative and therefore should be compared only
    within the same horizon/confidence configuration. Stress classification
    uses the worst supplied hypothetical scenario, if any.

    The overall level is the maximum of available dimensions, a conservative
    analytical flag rather than a probability or suitability determination.
    """
    if assessment.portfolio_id != request.portfolio.portfolio_id:
        raise ValueError("Assessment portfolio_id does not match the request")
    if assessment.assessment_id != request.assessment_id:
        raise ValueError("Assessment assessment_id does not match the request")
    if assessment.horizon != request.config.horizon or abs(
        assessment.confidence_level - request.config.confidence_level
    ) > 1e-12:
        raise ValueError("Assessment horizon/confidence does not match the request")

    thresholds: RiskThresholds = request.config.thresholds
    dimensions: dict[str, RiskLevel] = {}
    explanations: list[str] = []

    if assessment.annualized_volatility is None:
        dimensions["volatility"] = RiskLevel.INDETERMINATE
        explanations.append("Annualized volatility is unavailable; volatility classification is indeterminate.")
    else:
        cutoffs = (
            thresholds.volatility_moderate,
            thresholds.volatility_high,
            thresholds.volatility_very_high,
        )
        level = _band(assessment.annualized_volatility, cutoffs)
        dimensions["volatility"] = level
        explanations.append(
            f"Annualized volatility {assessment.annualized_volatility:.2%} is classified "
            f"{level.value} against configured cutoffs "
            f"({cutoffs[0]:.0%}, {cutoffs[1]:.0%}, {cutoffs[2]:.0%})."
        )

    cvar = assessment.conditional_value_at_risk
    if cvar is None:
        dimensions["downside"] = RiskLevel.INDETERMINATE
        explanations.append("CVaR is unavailable; downside classification is indeterminate.")
    elif cvar.horizon != request.config.horizon or cvar.confidence_level is None or abs(
        cvar.confidence_level - request.config.confidence_level
    ) > 1e-12:
        raise ValueError("CVaR horizon/confidence does not match the request")
    else:
        # A negative CVaR represents a gain threshold. No modeled loss => 0.
        loss = max(0.0, cvar.value)
        level = _band(loss, _DOWNSIDE_CUTOFFS)
        dimensions["downside"] = level
        explanations.append(
            f"Modeled CVaR loss {loss:.2%} ({request.config.horizon.value}, "
            f"{request.config.confidence_level:.1%} confidence) is classified "
            f"{level.value} using illustrative downside cutoffs "
            "(2%, 5%, 10%); classifications across different horizons "
            "are not directly comparable."
        )

    indicators = concentration_indicators(request.portfolio)
    largest = float(indicators["largest_asset_weight"])
    level = _concentration_band(largest, thresholds.concentration_high)
    dimensions["concentration"] = level
    explanations.append(
        f"Largest asset allocation {largest:.2%}; asset HHI "
        f"{float(indicators['asset_hhi']):.3f}; concentration level {level.value}. "
        f"Configured concentration threshold {thresholds.concentration_high:.1%}; "
        "asset-class totals are available separately."
    )

    supplied = list(stress_results) if stress_results is not None else list(assessment.stress_results)
    if not supplied:
        dimensions["stress"] = RiskLevel.INDETERMINATE
        explanations.append("No stress scenarios were evaluated; stress classification is indeterminate.")
    else:
        scenario_ids = [item.scenario_id for item in supplied]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("Duplicate stress scenario IDs in classification inputs")
        worst = min(item.portfolio_shock_return for item in supplied)
        if not isfinite(worst):
            raise ValueError("Stress scenario returns must be finite")
        loss = max(0.0, -worst)
        level = _band(loss, _STRESS_CUTOFFS)
        dimensions["stress"] = level
        explanations.append(
            f"Worst hypothetical stress loss {loss:.2%} across {len(supplied)} "
            f"scenario(s) is classified {level.value} using illustrative "
            "cutoffs (5%, 15%, 30%); scenario likelihood is unknown."
        )

    known = [level for level in dimensions.values() if level in _RISK_RANK]
    overall = max(known, key=_RISK_RANK.__getitem__) if known else RiskLevel.INDETERMINATE
    missing = [name for name, level in dimensions.items() if level == RiskLevel.INDETERMINATE]
    explanations.append(
        f"Overall {overall.value}: highest available dimension, not an average "
        "or investment suitability determination."
    )
    if missing:
        explanations.append("Unavailable dimensions: " + ", ".join(missing) + ".")
    return RiskClassification(
        overall=overall,
        dimensions=dimensions,
        explanations=explanations,
        thresholds=thresholds.model_copy(deep=True),
    )


def _normalize_tolerance(value: str | None) -> RiskLevel | None:
    """Recognize only explicit, unambiguous risk-tolerance labels."""
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "low": RiskLevel.LOW,
        "conservative": RiskLevel.LOW,
        "moderate": RiskLevel.MODERATE,
        "medium": RiskLevel.MODERATE,
        "balanced": RiskLevel.MODERATE,
        "high": RiskLevel.HIGH,
        "aggressive": RiskLevel.HIGH,
        "very_high": RiskLevel.VERY_HIGH,
        "very_aggressive": RiskLevel.VERY_HIGH,
    }
    return aliases.get(normalized)


def assess_investor_alignment(
    request: RiskAssessmentRequest,
    classification: RiskClassification,
    profile: InvestorRiskProfile | None = None,
) -> InvestorAlignment:
    """Evaluate explicitly stated tolerance and horizon; no suitability advice.

    Missing or unrecognized tolerance cannot be treated as aligned. A horizon
    shorter than three years is flagged for human review if the portfolio is
    HIGH/VERY_HIGH; this illustrative heuristic is NOT a suitability rule.
    Objectives and constraints are retained as context but are not parsed as
    financial instructions. They require human review and cannot silently
    become numerical limits or an 'aligned' verdict.
    """
    investor = profile if profile is not None else request.investor_profile
    explanations: list[str] = []
    missing: list[str] = []
    if investor is None:
        return InvestorAlignment(
            status=AlignmentStatus.INSUFFICIENT_INFORMATION,
            explanations=["No investor profile supplied; portfolio suitability cannot be inferred."],
            missing_fields=["risk_tolerance", "investment_horizon_years", "financial_objectives", "constraints"],
            suitability_determination=False,
        )

    tolerance = _normalize_tolerance(investor.risk_tolerance)
    if tolerance is None:
        missing.append("risk_tolerance")
        explanations.append("Risk tolerance is missing or unrecognized; no tolerance match is inferred.")
    if investor.investment_horizon_years is None:
        missing.append("investment_horizon_years")
        explanations.append("Investment horizon is missing; horizon-related alignment is not assessed.")
    if not investor.financial_objectives:
        missing.append("financial_objectives")
    else:
        explanations.append("Financial objectives were supplied but require human interpretation; not automatically scored.")
    if not investor.constraints:
        missing.append("constraints")
    else:
        explanations.append("Investor constraints were supplied but require human interpretation; not automatically scored.")

    overall = classification.overall
    mismatch = False
    if overall == RiskLevel.INDETERMINATE:
        explanations.append("Portfolio overall risk is indeterminate; alignment cannot be confirmed.")
    elif tolerance is not None:
        if _RISK_RANK[overall] > _RISK_RANK[tolerance]:
            mismatch = True
            explanations.append(
                f"Potential mismatch: portfolio risk {overall.value} exceeds "
                f"stated tolerance {tolerance.value} under the illustrative rubric."
            )
        else:
            explanations.append(
                f"Portfolio risk {overall.value} does not exceed stated tolerance "
                f"{tolerance.value} under the illustrative rubric."
            )
    if investor.investment_horizon_years is not None and overall in (
        RiskLevel.HIGH, RiskLevel.VERY_HIGH
    ) and investor.investment_horizon_years < 3.0:
        mismatch = True
        explanations.append(
            "Potential mismatch: high assessed portfolio risk with an investment "
            "horizon under three years (illustrative review flag, not a suitability rule)."
        )

    if mismatch:
        status = AlignmentStatus.POTENTIAL_MISMATCH
    elif (overall == RiskLevel.INDETERMINATE or missing
          or investor.financial_objectives or investor.constraints):
        status = AlignmentStatus.INSUFFICIENT_INFORMATION
        if investor.financial_objectives or investor.constraints:
            explanations.append("Supplied objectives or constraints have not been evaluated; full alignment cannot be confirmed.")
    else:
        status = AlignmentStatus.ALIGNED
        explanations.append(
            "No mismatch was identified under the limited, illustrative checks; "
            "this is not a formal suitability determination."
        )
    return InvestorAlignment(
        status=status,
        explanations=explanations,
        missing_fields=missing,
        suitability_determination=False,
    )


__all__ = [
    "CLASSIFICATION_VERSION",
    "asset_concentration",
    "asset_class_concentration",
    "concentration_indicators",
    "classify_portfolio_risk",
    "assess_investor_alignment",
]
