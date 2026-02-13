"""Carbon emission calculator for LLM inference.

Core functions:
    find_carbon_factor(provider, model) -- look up emission factor
    calculate_carbon(input_tokens, output_tokens, ...) -- compute emissions
    calculate_equivalents(co2_grams) -- human-friendly comparisons
"""

from __future__ import annotations

from dataclasses import dataclass

from qortex_observe.carbon.config import DEFAULT_CARBON_FACTORS, FALLBACK_CARBON_FACTOR
from qortex_observe.carbon.types import (
    CarbonCalculation,
    CarbonEquivalents,
    CarbonFactor,
    CarbonFactorSource,
    ConfidenceLevel,
    GhgCalculationMethod,
    GhgDataQualityScore,
    UncertaintyBounds,
)


def find_carbon_factor(
    provider: str | None = None,
    model: str | None = None,
) -> CarbonFactor:
    """Look up emission factor for a provider/model pair.

    Matching strategy:
    1. Exact match on provider + model prefix (e.g. "claude-sonnet-4" matches "claude-sonnet")
    2. Most conservative (highest CO2) factor for the provider
    3. Fallback factor for unknown providers
    """
    if not provider or not model:
        return FALLBACK_CARBON_FACTOR

    provider_lower = provider.lower()
    model_lower = model.lower()

    # Try prefix match
    for factor in DEFAULT_CARBON_FACTORS:
        if factor.provider == provider_lower and model_lower.startswith(factor.model):
            return factor

    # Fall back to most conservative factor for same provider
    provider_factors = [f for f in DEFAULT_CARBON_FACTORS if f.provider == provider_lower]
    if provider_factors:
        return max(provider_factors, key=lambda f: f.output_co2_per_1m_tokens)

    return FALLBACK_CARBON_FACTOR


def calculate_carbon(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    provider: str | None = None,
    model: str | None = None,
    factor: CarbonFactor | None = None,
) -> CarbonCalculation:
    """Calculate carbon emissions for a single LLM inference.

    Either pass provider+model (auto-lookup) or an explicit factor.
    """
    if factor is None:
        factor = find_carbon_factor(provider, model)

    input_co2 = (input_tokens / 1_000_000) * factor.input_co2_per_1m_tokens
    output_co2 = (output_tokens / 1_000_000) * factor.output_co2_per_1m_tokens
    cache_co2 = (cache_read_tokens / 1_000_000) * factor.cache_read_co2_per_1m_tokens
    total_co2 = input_co2 + output_co2 + cache_co2

    total_tokens = input_tokens + output_tokens + cache_read_tokens
    water_ml = (total_tokens / 1_000_000) * factor.water_ml_per_1m_tokens

    return CarbonCalculation(
        input_co2_grams=input_co2,
        output_co2_grams=output_co2,
        cache_co2_grams=cache_co2,
        total_co2_grams=total_co2,
        water_ml=water_ml,
        factor=factor,
    )


def calculate_equivalents(co2_grams: float) -> CarbonEquivalents:
    """Convert CO2 grams to human-friendly equivalents.

    Reference values:
    - 1 km driving = 120g CO2
    - 1 phone charge = 10g CO2
    - 1 tree absorbs 48g CO2/day
    - 1 Google search = 0.2g CO2
    """
    return CarbonEquivalents(
        car_km=co2_grams / 120,
        phone_charges=round(co2_grams / 10),
        tree_days=co2_grams / 48,
        google_searches=round(co2_grams / 0.2),
    )


# ── Confidence and compliance mappers ────────────────────────────


def format_confidence(confidence: float) -> ConfidenceLevel:
    """Map numeric confidence to human-readable tier."""
    if confidence >= 0.7:
        return ConfidenceLevel.HIGH
    if confidence >= 0.5:
        return ConfidenceLevel.MEDIUM
    if confidence >= 0.3:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def confidence_to_uncertainty(confidence: float) -> UncertaintyBounds:
    """Map confidence to emission uncertainty bounds (multipliers)."""
    if confidence >= 0.7:
        return UncertaintyBounds(lower=0.85, upper=1.15)
    if confidence >= 0.5:
        return UncertaintyBounds(lower=0.70, upper=1.30)
    if confidence >= 0.3:
        return UncertaintyBounds(lower=0.50, upper=1.50)
    return UncertaintyBounds(lower=0.00, upper=2.00)


def source_to_calculation_method(source: CarbonFactorSource) -> GhgCalculationMethod:
    """Map factor source to GHG Protocol calculation method."""
    if source == CarbonFactorSource.MEASURED:
        return GhgCalculationMethod.SUPPLIER_SPECIFIC
    if source == CarbonFactorSource.RESEARCH:
        return GhgCalculationMethod.HYBRID
    return GhgCalculationMethod.AVERAGE_DATA


def confidence_to_data_quality(confidence: float) -> GhgDataQualityScore:
    """Map confidence to GHG Protocol 5-point data quality score (1=best)."""
    if confidence >= 0.8:
        return 1
    if confidence >= 0.6:
        return 2
    if confidence >= 0.4:
        return 3
    if confidence >= 0.2:
        return 4
    return 5
