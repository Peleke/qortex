"""Default carbon emission factors for LLM providers.

Factors are estimated from:
- ML CO2 Impact Calculator (Lacoste et al., 2019)
- Carbon Emissions and Large Neural Network Training (Patterson et al., 2022)
- Power Hungry Processing (Luccioni et al., 2024)
- Cloud Carbon Footprint methodology
- CodeCarbon hardware measurements

Output tokens use approximately 3:1 energy ratio vs input tokens.
"""

from __future__ import annotations

from qortex.observe.carbon.types import CarbonFactor, CarbonFactorSource

DEFAULT_CARBON_FACTORS: tuple[CarbonFactor, ...] = (
    CarbonFactor(
        provider="anthropic",
        model="claude-haiku",
        input_co2_per_1m_tokens=30,
        output_co2_per_1m_tokens=90,
        cache_read_co2_per_1m_tokens=3,
        water_ml_per_1m_tokens=600,
        confidence=0.35,
        source=CarbonFactorSource.ESTIMATED,
    ),
    CarbonFactor(
        provider="anthropic",
        model="claude-sonnet",
        input_co2_per_1m_tokens=150,
        output_co2_per_1m_tokens=450,
        cache_read_co2_per_1m_tokens=15,
        water_ml_per_1m_tokens=3000,
        confidence=0.3,
        source=CarbonFactorSource.ESTIMATED,
    ),
    CarbonFactor(
        provider="anthropic",
        model="claude-opus",
        input_co2_per_1m_tokens=400,
        output_co2_per_1m_tokens=1200,
        cache_read_co2_per_1m_tokens=40,
        water_ml_per_1m_tokens=8000,
        confidence=0.25,
        source=CarbonFactorSource.ESTIMATED,
    ),
    CarbonFactor(
        provider="openai",
        model="gpt-4o-mini",
        input_co2_per_1m_tokens=40,
        output_co2_per_1m_tokens=120,
        cache_read_co2_per_1m_tokens=4,
        water_ml_per_1m_tokens=800,
        confidence=0.35,
        source=CarbonFactorSource.ESTIMATED,
    ),
    CarbonFactor(
        provider="openai",
        model="gpt-4o",
        input_co2_per_1m_tokens=200,
        output_co2_per_1m_tokens=600,
        cache_read_co2_per_1m_tokens=20,
        water_ml_per_1m_tokens=4000,
        confidence=0.3,
        source=CarbonFactorSource.ESTIMATED,
    ),
    CarbonFactor(
        provider="openai",
        model="gpt-4",
        input_co2_per_1m_tokens=300,
        output_co2_per_1m_tokens=900,
        cache_read_co2_per_1m_tokens=30,
        water_ml_per_1m_tokens=6000,
        confidence=0.25,
        source=CarbonFactorSource.ESTIMATED,
    ),
)

FALLBACK_CARBON_FACTOR = CarbonFactor(
    provider="unknown",
    model="unknown",
    input_co2_per_1m_tokens=200,
    output_co2_per_1m_tokens=600,
    cache_read_co2_per_1m_tokens=20,
    water_ml_per_1m_tokens=4000,
    confidence=0.15,
    source=CarbonFactorSource.FALLBACK,
)

DEFAULT_GRID_CARBON_GRAMS_PER_KWH = 400

METHODOLOGY_DESCRIPTION = (
    "Per-token emission factors estimated from academic research "
    "(Lacoste et al. 2019, Patterson et al. 2022, Luccioni et al. 2024) "
    "with conservative fallbacks. Factors account for GPU power consumption "
    "during inference with approximately 3:1 output-to-input energy ratio. "
    "All calculations are Scope 3, Category 1 (Purchased Goods and Services)."
)

EMISSION_FACTOR_SOURCES = [
    "ML CO2 Impact Calculator (Lacoste et al., 2019)",
    "Carbon Emissions and Large Neural Network Training (Patterson et al., 2022)",
    "Power Hungry Processing (Luccioni et al., 2024)",
    "Cloud Carbon Footprint methodology",
    "CodeCarbon hardware measurements",
]
