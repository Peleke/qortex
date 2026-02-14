"""Carbon accounting for LLM inference emissions.

Public API:
    calculate_carbon(input_tokens, output_tokens, ...) -- compute emissions
    calculate_equivalents(co2_grams) -- human-friendly comparisons
    find_carbon_factor(provider, model) -- look up emission factor
    export_ghg_protocol(summary, period) -- GHG Protocol Scope 3 export
    export_cdp(summary, year) -- CDP Climate Disclosure export
    export_tcfd(summary, period) -- TCFD-aligned disclosure export
    export_iso14064(summary, period) -- ISO 14064-1:2018 export
"""

from qortex.observe.carbon.calculator import (
    calculate_carbon,
    calculate_equivalents,
    confidence_to_data_quality,
    confidence_to_uncertainty,
    find_carbon_factor,
    format_confidence,
    source_to_calculation_method,
)
from qortex.observe.carbon.config import (
    DEFAULT_CARBON_FACTORS,
    FALLBACK_CARBON_FACTOR,
)
from qortex.observe.carbon.ghg import (
    export_cdp,
    export_ghg_protocol,
    export_iso14064,
    export_tcfd,
)
from qortex.observe.carbon.types import (
    CarbonCalculation,
    CarbonEquivalents,
    CarbonFactor,
    CarbonFactorSource,
    CarbonSummary,
    CdpExport,
    ConfidenceLevel,
    GhgProtocolExport,
    Iso14064Summary,
    TcfdExport,
    UncertaintyBounds,
)

__all__ = [
    # Core functions
    "calculate_carbon",
    "calculate_equivalents",
    "find_carbon_factor",
    # Mappers
    "format_confidence",
    "confidence_to_uncertainty",
    "confidence_to_data_quality",
    "source_to_calculation_method",
    # Types
    "CarbonCalculation",
    "CarbonEquivalents",
    "CarbonFactor",
    "CarbonFactorSource",
    "CarbonSummary",
    "ConfidenceLevel",
    "UncertaintyBounds",
    # Config
    "DEFAULT_CARBON_FACTORS",
    "FALLBACK_CARBON_FACTOR",
    # Regulatory exports
    "GhgProtocolExport",
    "CdpExport",
    "TcfdExport",
    "Iso14064Summary",
    # Export functions
    "export_ghg_protocol",
    "export_cdp",
    "export_tcfd",
    "export_iso14064",
]
