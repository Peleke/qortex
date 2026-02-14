"""Tests for qortex.observe.carbon -- calculator, factors, events, GHG exports."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qortex.observe.carbon import (
    DEFAULT_CARBON_FACTORS,
    FALLBACK_CARBON_FACTOR,
    CarbonCalculation,
    CarbonFactor,
    CarbonFactorSource,
    CarbonSummary,
    calculate_carbon,
    calculate_equivalents,
    confidence_to_data_quality,
    confidence_to_uncertainty,
    find_carbon_factor,
    format_confidence,
)
from qortex.observe.carbon.ghg import (
    export_cdp,
    export_ghg_protocol,
    export_iso14064,
    export_tcfd,
)
from qortex.observe.carbon.types import ConfidenceLevel
from qortex.observe.events import CarbonTracked

# ── Calculator Tests ──────────────────────────────────────────────


class TestFindCarbonFactor:
    """find_carbon_factor() lookup logic."""

    def test_exact_provider_and_model_prefix(self):
        """claude-sonnet-4-xxx matches the 'claude-sonnet' factor."""
        factor = find_carbon_factor(provider="anthropic", model="claude-sonnet-4-20250514")
        assert factor.provider == "anthropic"
        assert factor.model == "claude-sonnet"

    def test_haiku_match(self):
        factor = find_carbon_factor(provider="anthropic", model="claude-haiku-3-5")
        assert factor.model == "claude-haiku"

    def test_opus_match(self):
        factor = find_carbon_factor(provider="anthropic", model="claude-opus-4")
        assert factor.model == "claude-opus"

    def test_openai_gpt4o(self):
        factor = find_carbon_factor(provider="openai", model="gpt-4o-2024-08-06")
        assert factor.provider == "openai"
        assert factor.model == "gpt-4o"

    def test_openai_gpt4o_mini(self):
        factor = find_carbon_factor(provider="openai", model="gpt-4o-mini")
        assert factor.provider == "openai"
        assert factor.model == "gpt-4o-mini"

    def test_unknown_anthropic_model_returns_most_conservative(self):
        """Unknown Anthropic model falls back to most expensive (opus)."""
        factor = find_carbon_factor(provider="anthropic", model="claude-999")
        assert factor.provider == "anthropic"
        # Most conservative = highest output_co2_per_1m_tokens
        assert factor.output_co2_per_1m_tokens == max(
            f.output_co2_per_1m_tokens for f in DEFAULT_CARBON_FACTORS if f.provider == "anthropic"
        )

    def test_unknown_provider_returns_fallback(self):
        factor = find_carbon_factor(provider="mistral", model="mixtral-8x7b")
        assert factor is FALLBACK_CARBON_FACTOR

    def test_none_provider_returns_fallback(self):
        factor = find_carbon_factor(provider=None, model=None)
        assert factor is FALLBACK_CARBON_FACTOR

    def test_empty_strings_return_fallback(self):
        factor = find_carbon_factor(provider="", model="")
        assert factor is FALLBACK_CARBON_FACTOR

    def test_case_insensitive(self):
        factor = find_carbon_factor(provider="Anthropic", model="Claude-Sonnet-4")
        assert factor.model == "claude-sonnet"


class TestCalculateCarbon:
    """calculate_carbon() emission calculations."""

    def test_basic_calculation(self):
        calc = calculate_carbon(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert isinstance(calc, CarbonCalculation)
        assert calc.input_co2_grams == pytest.approx(150.0)
        assert calc.output_co2_grams == pytest.approx(450.0)
        assert calc.total_co2_grams == pytest.approx(600.0)
        assert calc.water_ml > 0

    def test_cache_read_tokens(self):
        calc = calculate_carbon(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1_000_000,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert calc.cache_co2_grams == pytest.approx(15.0)
        assert calc.input_co2_grams == 0.0
        assert calc.output_co2_grams == 0.0

    def test_zero_tokens_zero_emissions(self):
        calc = calculate_carbon(
            input_tokens=0,
            output_tokens=0,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert calc.total_co2_grams == 0.0
        assert calc.water_ml == 0.0

    def test_explicit_factor_overrides_lookup(self):
        custom = CarbonFactor(
            provider="custom",
            model="test",
            input_co2_per_1m_tokens=100,
            output_co2_per_1m_tokens=300,
            cache_read_co2_per_1m_tokens=10,
            water_ml_per_1m_tokens=2000,
            confidence=0.9,
            source=CarbonFactorSource.MEASURED,
        )
        calc = calculate_carbon(
            input_tokens=1_000_000,
            output_tokens=500_000,
            factor=custom,
        )
        assert calc.input_co2_grams == pytest.approx(100.0)
        assert calc.output_co2_grams == pytest.approx(150.0)
        assert calc.factor is custom

    def test_water_calculation(self):
        calc = calculate_carbon(
            input_tokens=500_000,
            output_tokens=500_000,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        # Total 1M tokens, sonnet water_ml_per_1m = 3000
        assert calc.water_ml == pytest.approx(3000.0)

    def test_small_token_counts(self):
        """Typical enrichment: ~2K input, ~500 output."""
        calc = calculate_carbon(
            input_tokens=2000,
            output_tokens=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert calc.total_co2_grams > 0
        assert calc.total_co2_grams < 1.0  # Sub-gram for small calls

    def test_negative_tokens_clamped_to_zero(self):
        """Negative token counts are clamped to zero."""
        calc = calculate_carbon(
            input_tokens=-100,
            output_tokens=-50,
            cache_read_tokens=-25,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert calc.input_co2_grams == 0.0
        assert calc.output_co2_grams == 0.0
        assert calc.cache_co2_grams == 0.0
        assert calc.total_co2_grams == 0.0
        assert calc.water_ml == 0.0


class TestCalculateEquivalents:
    """calculate_equivalents() human-readable conversions."""

    def test_known_value(self):
        eq = calculate_equivalents(120.0)  # 120g = 1 km driving
        assert eq.car_km == pytest.approx(1.0)

    def test_phone_charges(self):
        eq = calculate_equivalents(50.0)  # 50g = 5 phone charges
        assert eq.phone_charges == 5

    def test_tree_days(self):
        eq = calculate_equivalents(48.0)  # 48g = 1 tree-day
        assert eq.tree_days == pytest.approx(1.0)

    def test_google_searches(self):
        eq = calculate_equivalents(1.0)  # 1g = 5 Google searches
        assert eq.google_searches == 5

    def test_zero_emissions(self):
        eq = calculate_equivalents(0.0)
        assert eq.car_km == 0.0
        assert eq.phone_charges == 0
        assert eq.tree_days == 0.0
        assert eq.google_searches == 0

    def test_negative_emissions_clamped(self):
        """Negative CO2 grams are clamped to zero."""
        eq = calculate_equivalents(-50.0)
        assert eq.car_km == 0.0
        assert eq.phone_charges == 0
        assert eq.tree_days == 0.0
        assert eq.google_searches == 0


# ── Confidence/Compliance Mapper Tests ────────────────────────────


class TestConfidenceMappers:
    def test_format_confidence_high(self):
        assert format_confidence(0.8) == ConfidenceLevel.HIGH

    def test_format_confidence_medium(self):
        assert format_confidence(0.5) == ConfidenceLevel.MEDIUM

    def test_format_confidence_low(self):
        assert format_confidence(0.3) == ConfidenceLevel.LOW

    def test_format_confidence_very_low(self):
        assert format_confidence(0.1) == ConfidenceLevel.VERY_LOW

    def test_uncertainty_bounds_high(self):
        bounds = confidence_to_uncertainty(0.8)
        assert bounds.lower == pytest.approx(0.85)
        assert bounds.upper == pytest.approx(1.15)

    def test_uncertainty_bounds_low(self):
        bounds = confidence_to_uncertainty(0.3)
        assert bounds.lower == pytest.approx(0.50)
        assert bounds.upper == pytest.approx(1.50)

    def test_confidence_above_one_clamped(self):
        """Confidence > 1.0 is clamped to 1.0 (HIGH tier)."""
        assert format_confidence(1.5) == ConfidenceLevel.HIGH

    def test_confidence_negative_clamped(self):
        """Negative confidence is clamped to 0.0 (VERY_LOW tier)."""
        assert format_confidence(-0.5) == ConfidenceLevel.VERY_LOW

    def test_data_quality_score_best(self):
        assert confidence_to_data_quality(0.9) == 1

    def test_data_quality_score_worst(self):
        assert confidence_to_data_quality(0.1) == 5


# ── GHG/Regulatory Export Tests ───────────────────────────────────


class TestGhgExports:
    """Regulatory export formatters."""

    @pytest.fixture()
    def sample_summary(self) -> CarbonSummary:
        return CarbonSummary(
            trace_count=100,
            total_co2_grams=5000.0,
            total_water_ml=100_000.0,
            avg_co2_per_trace=50.0,
            avg_confidence=0.3,
            min_timestamp=1700000000,
            max_timestamp=1700100000,
            total_tokens=10_000_000,
            intensity_per_million_tokens=500.0,
            intensity_per_query=50.0,
            uncertainty_lower=2500.0,
            uncertainty_upper=7500.0,
        )

    def test_ghg_protocol_export(self, sample_summary):
        export = export_ghg_protocol(sample_summary, "2026-Q1")
        assert export.scope == 3
        assert export.category == 1
        assert export.emissions_tco2eq == pytest.approx(5000.0 / 1_000_000)
        assert export.reporting_period == "2026-Q1"
        assert export.uncertainty_percent > 0

    def test_cdp_export(self, sample_summary):
        export = export_cdp(sample_summary, 2026)
        assert export.reporting_year == 2026
        assert export.emissions_tco2eq == pytest.approx(5000.0 / 1_000_000)
        assert "Score" in export.data_quality

    def test_tcfd_export(self, sample_summary):
        export = export_tcfd(sample_summary, "2026-Q1")
        assert export.reporting_period == "2026-Q1"
        assert export.absolute_emissions_tco2eq > 0

    def test_tcfd_with_base_year(self, sample_summary):
        export = export_tcfd(
            sample_summary,
            "2026-Q1",
            base_year=2025,
            base_year_emissions_grams=4000.0,
        )
        assert export.base_year == 2025
        assert export.base_year_change_percent == pytest.approx(25.0)

    def test_iso14064_export(self, sample_summary):
        export = export_iso14064(sample_summary, "2026")
        assert export.reporting_period == "2026"
        assert export.uncertainty_lower_tco2eq < export.emissions_tco2eq
        assert export.uncertainty_upper_tco2eq > export.emissions_tco2eq


# ── CarbonTracked Event Tests ─────────────────────────────────────


class TestCarbonTrackedEvent:
    """CarbonTracked event + metric wiring."""

    def test_event_fields(self):
        evt = CarbonTracked(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            total_co2_grams=0.3,
            water_ml=4.5,
            confidence=0.3,
            timestamp="2026-02-13T00:00:00+00:00",
        )
        assert evt.provider == "anthropic"
        assert evt.total_co2_grams == 0.3
        assert evt.water_ml == 4.5

    def test_carbon_metric_handler_registers(self):
        """Verify register_metric_handlers subscribes to CarbonTracked."""
        from qortex.observe.linker import QortexEventLinker
        from qortex.observe.metrics_handlers import register_metric_handlers

        mock_instruments = {
            "qortex_carbon_co2_grams": MagicMock(),
            "qortex_carbon_water_ml": MagicMock(),
            "qortex_carbon_tokens": MagicMock(),
            "qortex_carbon_confidence": MagicMock(),
            # Provide stubs for all other instruments so registration doesn't fail
        }
        # Use a defaultdict-like mock for instruments not under test
        from collections import defaultdict

        full_instruments = defaultdict(MagicMock, mock_instruments)
        register_metric_handlers(full_instruments)

        # CarbonTracked should now have at least one subscriber
        subs = QortexEventLinker.get_subscribers_from_events(CarbonTracked)
        assert len(subs) >= 1

    def test_carbon_metrics_in_schema(self):
        """Verify all 4 carbon metrics are defined in METRICS schema."""
        from qortex.observe.metrics_schema import METRICS

        metric_names = {m.name for m in METRICS}
        assert "qortex_carbon_co2_grams" in metric_names
        assert "qortex_carbon_water_ml" in metric_names
        assert "qortex_carbon_tokens" in metric_names
        assert "qortex_carbon_confidence" in metric_names


# ── Enrichment Integration Tests ──────────────────────────────────


class TestEnrichmentCarbonIntegration:
    """Verify AnthropicEnrichmentBackend emits CarbonTracked."""

    def test_enrich_batch_emits_carbon(self):
        """_enrich_one_batch calls _emit_carbon after API response."""
        from qortex.core.models import Rule
        from qortex.enrichment.anthropic import AnthropicEnrichmentBackend

        # Mock the Anthropic client response
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='[{"context":"c","antipattern":"a","rationale":"r","tags":["t"]}]')
        ]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200
        mock_response.usage.cache_read_input_tokens = 0

        # Create backend with mocked internals
        backend = object.__new__(AnthropicEnrichmentBackend)
        backend._client = MagicMock()
        backend._model = "claude-sonnet-4-20250514"
        backend._batch_size = 10
        backend._client.messages.create.return_value = mock_response

        captured_events = []
        with patch("qortex.enrichment.anthropic.emit", side_effect=captured_events.append):
            rule = Rule(
                id="test-1",
                text="Test rule",
                domain="testing",
                derivation="explicit",
                source_concepts=[],
                confidence=0.9,
            )
            backend.enrich_batch([rule], domain="testing")

        carbon_events = [e for e in captured_events if isinstance(e, CarbonTracked)]
        assert len(carbon_events) == 1
        evt = carbon_events[0]
        assert evt.provider == "anthropic"
        assert evt.input_tokens == 500
        assert evt.output_tokens == 200
        assert evt.total_co2_grams > 0


# ── Config/Factor Table Tests ─────────────────────────────────────


class TestFactorTable:
    """DEFAULT_CARBON_FACTORS integrity."""

    def test_all_factors_have_positive_values(self):
        for f in DEFAULT_CARBON_FACTORS:
            assert f.input_co2_per_1m_tokens > 0
            assert f.output_co2_per_1m_tokens > 0
            assert f.water_ml_per_1m_tokens > 0

    def test_output_higher_than_input(self):
        """Output tokens should cost more energy than input (3:1 ratio)."""
        for f in DEFAULT_CARBON_FACTORS:
            assert f.output_co2_per_1m_tokens > f.input_co2_per_1m_tokens

    def test_fallback_exists(self):
        assert FALLBACK_CARBON_FACTOR is not None
        assert FALLBACK_CARBON_FACTOR.provider == "unknown"

    def test_anthropic_models_present(self):
        providers = {f.provider for f in DEFAULT_CARBON_FACTORS}
        assert "anthropic" in providers

    def test_no_duplicate_entries(self):
        keys = [(f.provider, f.model) for f in DEFAULT_CARBON_FACTORS]
        assert len(keys) == len(set(keys))

    def test_confidence_in_valid_range(self):
        for f in DEFAULT_CARBON_FACTORS:
            assert 0 < f.confidence <= 1.0


# ── Config Env Var Validation Tests ──────────────────────────────


class TestConfigEnvVarValidation:
    """Verify _int_env and _float_env helpers."""

    def test_int_env_valid(self):
        from qortex.observe.config import _int_env

        with patch.dict("os.environ", {"TEST_PORT": "8080"}):
            assert _int_env("TEST_PORT", "9090") == 8080

    def test_int_env_default(self):
        from qortex.observe.config import _int_env

        assert _int_env("NONEXISTENT_VAR", "42") == 42

    def test_int_env_invalid_raises(self):
        from qortex.observe.config import _int_env

        with (
            patch.dict("os.environ", {"BAD_PORT": "abc"}),
            pytest.raises(ValueError, match="Invalid integer"),
        ):
            _int_env("BAD_PORT", "9090")

    def test_int_env_below_min_raises(self):
        from qortex.observe.config import _int_env

        with (
            patch.dict("os.environ", {"LOW_PORT": "0"}),
            pytest.raises(ValueError, match="below minimum"),
        ):
            _int_env("LOW_PORT", "9090", min_val=1)

    def test_int_env_above_max_raises(self):
        from qortex.observe.config import _int_env

        with (
            patch.dict("os.environ", {"HIGH_PORT": "99999"}),
            pytest.raises(ValueError, match="exceeds maximum"),
        ):
            _int_env("HIGH_PORT", "9090", max_val=65535)
