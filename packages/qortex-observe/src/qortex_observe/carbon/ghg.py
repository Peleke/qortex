"""Regulatory carbon disclosure exports.

Formats:
    GHG Protocol -- Scope 3 Category 1 (Purchased Goods and Services)
    CDP          -- Climate Disclosure Project
    TCFD         -- Task Force on Climate-Related Financial Disclosures
    ISO 14064    -- ISO 14064-1:2018 compliant quantification
"""

from __future__ import annotations

from qortex_observe.carbon.calculator import (
    confidence_to_data_quality,
    confidence_to_uncertainty,
    format_confidence,
    source_to_calculation_method,
)
from qortex_observe.carbon.config import EMISSION_FACTOR_SOURCES, METHODOLOGY_DESCRIPTION
from qortex_observe.carbon.types import (
    CarbonFactorSource,
    CarbonSummary,
    CdpExport,
    GhgProtocolExport,
    Iso14064Summary,
    TcfdExport,
)


def export_ghg_protocol(
    summary: CarbonSummary,
    reporting_period: str,
    source: CarbonFactorSource = CarbonFactorSource.ESTIMATED,
) -> GhgProtocolExport:
    """Export to GHG Protocol Scope 3 Category 1 format.

    Args:
        summary: Aggregated carbon statistics.
        reporting_period: e.g. "2026-Q1" or "2026".
        source: Factor provenance for methodology determination.
    """
    method = source_to_calculation_method(source)
    quality = confidence_to_data_quality(summary.avg_confidence)
    bounds = confidence_to_uncertainty(summary.avg_confidence)
    uncertainty_pct = max(abs(1 - bounds.lower), abs(bounds.upper - 1)) * 100

    return GhgProtocolExport(
        reporting_period=reporting_period,
        emissions_tco2eq=summary.total_co2_grams / 1_000_000,
        calculation_method=method.value,
        data_quality=f"Score {quality}/5",
        uncertainty_percent=round(uncertainty_pct, 1),
        methodology=METHODOLOGY_DESCRIPTION,
        emission_factor_sources=list(EMISSION_FACTOR_SOURCES),
    )


def export_cdp(
    summary: CarbonSummary,
    reporting_year: int,
) -> CdpExport:
    """Export to CDP Climate Disclosure format.

    Args:
        summary: Aggregated carbon statistics.
        reporting_year: e.g. 2026.
    """
    quality = confidence_to_data_quality(summary.avg_confidence)

    return CdpExport(
        reporting_year=reporting_year,
        emissions_tco2eq=summary.total_co2_grams / 1_000_000,
        intensity_per_million_tokens=summary.intensity_per_million_tokens / 1_000_000,
        intensity_per_query=summary.intensity_per_query / 1_000_000,
        data_quality=f"Score {quality}/5",
    )


def export_tcfd(
    summary: CarbonSummary,
    reporting_period: str,
    quarterly_trend: list[dict] | None = None,
    base_year: int | None = None,
    base_year_emissions_grams: float | None = None,
) -> TcfdExport:
    """Export to TCFD-aligned climate disclosure.

    Args:
        summary: Aggregated carbon statistics.
        reporting_period: e.g. "2026-Q1".
        quarterly_trend: Optional list of quarterly data points.
        base_year: Reference year for comparison.
        base_year_emissions_grams: Emissions in base year (grams CO2).
    """
    base_change = None
    if base_year is not None and base_year_emissions_grams and base_year_emissions_grams > 0:
        base_change = (
            (summary.total_co2_grams - base_year_emissions_grams)
            / base_year_emissions_grams
            * 100
        )

    return TcfdExport(
        reporting_period=reporting_period,
        absolute_emissions_tco2eq=summary.total_co2_grams / 1_000_000,
        intensity_per_million_tokens=summary.intensity_per_million_tokens / 1_000_000,
        intensity_per_query=summary.intensity_per_query / 1_000_000,
        quarterly_trend=quarterly_trend or [],
        base_year=base_year,
        base_year_change_percent=round(base_change, 2) if base_change is not None else None,
    )


def export_iso14064(
    summary: CarbonSummary,
    reporting_period: str,
    base_year: int | None = None,
    base_year_emissions_grams: float | None = None,
) -> Iso14064Summary:
    """Export to ISO 14064-1:2018 format.

    Args:
        summary: Aggregated carbon statistics.
        reporting_period: e.g. "2026".
        base_year: Reference year for comparison.
        base_year_emissions_grams: Emissions in base year (grams CO2).
    """
    quality = confidence_to_data_quality(summary.avg_confidence)
    bounds = confidence_to_uncertainty(summary.avg_confidence)
    uncertainty_pct = max(abs(1 - bounds.lower), abs(bounds.upper - 1)) * 100

    tco2eq = summary.total_co2_grams / 1_000_000

    base_change = None
    if base_year is not None and base_year_emissions_grams and base_year_emissions_grams > 0:
        base_change = (
            (summary.total_co2_grams - base_year_emissions_grams)
            / base_year_emissions_grams
            * 100
        )

    return Iso14064Summary(
        reporting_period=reporting_period,
        emissions_tco2eq=tco2eq,
        uncertainty_lower_tco2eq=tco2eq * bounds.lower,
        uncertainty_upper_tco2eq=tco2eq * bounds.upper,
        uncertainty_percent=round(uncertainty_pct, 1),
        data_quality=f"Score {quality}/5",
        base_year=base_year,
        base_year_change_percent=round(base_change, 2) if base_change is not None else None,
    )
