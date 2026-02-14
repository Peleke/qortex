"""Carbon accounting types for LLM inference emissions.

Covers GHG Protocol Scope 3 Category 1, CDP, TCFD, and ISO 14064-1:2018
reporting formats. All emissions in grams CO2-equivalent (gCO2eq).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class CarbonFactorSource(str, Enum):
    """Provenance of a carbon emission factor."""

    MEASURED = "measured"
    RESEARCH = "research"
    ESTIMATED = "estimated"
    FALLBACK = "fallback"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence tier."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class GhgCalculationMethod(str, Enum):
    """GHG Protocol calculation methodology."""

    SUPPLIER_SPECIFIC = "supplier-specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average-data"


GhgDataQualityScore = Literal[1, 2, 3, 4, 5]


@dataclass(frozen=True)
class CarbonFactor:
    """Emission factor for a specific model.

    All CO2 values in grams per 1M tokens.
    Water in milliliters per 1M tokens.
    """

    provider: str
    model: str
    input_co2_per_1m_tokens: float
    output_co2_per_1m_tokens: float
    cache_read_co2_per_1m_tokens: float
    water_ml_per_1m_tokens: float
    confidence: float
    source: CarbonFactorSource
    last_updated: int = 0


@dataclass(frozen=True)
class CarbonCalculation:
    """Result of a single carbon calculation."""

    input_co2_grams: float
    output_co2_grams: float
    cache_co2_grams: float
    total_co2_grams: float
    water_ml: float
    factor: CarbonFactor


@dataclass(frozen=True)
class CarbonEquivalents:
    """Human-friendly equivalents for CO2 emissions."""

    car_km: float
    phone_charges: int
    tree_days: float
    google_searches: int


@dataclass(frozen=True)
class UncertaintyBounds:
    """Multiplier bounds for emission uncertainty."""

    lower: float
    upper: float


@dataclass(frozen=True)
class CarbonSummary:
    """Aggregated carbon statistics over a period."""

    trace_count: int
    total_co2_grams: float
    total_water_ml: float
    avg_co2_per_trace: float
    avg_confidence: float
    min_timestamp: int | None
    max_timestamp: int | None
    total_tokens: int
    intensity_per_million_tokens: float
    intensity_per_query: float
    uncertainty_lower: float
    uncertainty_upper: float


# ── Regulatory export types ──────────────────────────────────────


@dataclass(frozen=True)
class GhgProtocolExport:
    """GHG Protocol Scope 3 Category 1 export."""

    reporting_period: str
    scope: int = 3
    category: int = 1
    category_name: str = "Purchased Goods and Services"
    emissions_tco2eq: float = 0.0
    calculation_method: str = ""
    data_quality: str = ""
    uncertainty_percent: float = 0.0
    methodology: str = ""
    emission_factor_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CdpExport:
    """CDP Climate Disclosure export."""

    reporting_year: int
    scope: int = 3
    category: int = 1
    emissions_tco2eq: float = 0.0
    methodology: str = "hybrid"
    intensity_per_million_tokens: float = 0.0
    intensity_per_query: float = 0.0
    data_quality: str = ""


@dataclass(frozen=True)
class TcfdExport:
    """TCFD-aligned climate disclosure."""

    reporting_period: str
    absolute_emissions_tco2eq: float = 0.0
    intensity_per_million_tokens: float = 0.0
    intensity_per_query: float = 0.0
    quarterly_trend: list[dict] = field(default_factory=list)
    base_year: int | None = None
    base_year_change_percent: float | None = None


@dataclass(frozen=True)
class Iso14064Summary:
    """ISO 14064-1:2018 compliant summary."""

    reporting_period: str
    emissions_tco2eq: float = 0.0
    uncertainty_lower_tco2eq: float = 0.0
    uncertainty_upper_tco2eq: float = 0.0
    uncertainty_percent: float = 0.0
    data_quality: str = ""
    base_year: int | None = None
    base_year_change_percent: float | None = None
