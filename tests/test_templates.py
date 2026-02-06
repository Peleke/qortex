"""Exhaustive tests for EdgeRuleTemplate registry."""

import pytest

from qortex.core.models import RelationType
from qortex.core.templates import (
    EDGE_RULE_TEMPLATE_REGISTRY,
    get_default_template,
    get_templates,
    select_template,
)


class TestRegistry:
    def test_registry_has_30_templates(self):
        assert len(EDGE_RULE_TEMPLATE_REGISTRY) == 30

    def test_3_variants_per_relation_type(self):
        for rt in RelationType:
            templates = get_templates(rt)
            assert len(templates) == 3, f"{rt} has {len(templates)} templates, expected 3"

    def test_unique_ids(self):
        ids = [t.id for t in EDGE_RULE_TEMPLATE_REGISTRY]
        assert len(ids) == len(set(ids)), "Template IDs must be unique"

    def test_all_templates_have_source_placeholder(self):
        for t in EDGE_RULE_TEMPLATE_REGISTRY:
            assert "{source}" in t.template, f"{t.id} missing {{source}} placeholder"

    def test_all_templates_have_target_placeholder(self):
        for t in EDGE_RULE_TEMPLATE_REGISTRY:
            assert "{target}" in t.template, f"{t.id} missing {{target}} placeholder"

    def test_severity_values(self):
        valid = {"info", "warning", "error"}
        for t in EDGE_RULE_TEMPLATE_REGISTRY:
            assert t.severity in valid, f"{t.id} has invalid severity: {t.severity}"

    def test_id_format(self):
        """IDs should be relation_type:variant."""
        for t in EDGE_RULE_TEMPLATE_REGISTRY:
            expected_prefix = t.relation_type.value
            assert t.id.startswith(expected_prefix + ":"), (
                f"{t.id} doesn't start with {expected_prefix}:"
            )

    def test_template_is_frozen(self):
        t = EDGE_RULE_TEMPLATE_REGISTRY[0]
        with pytest.raises(AttributeError):
            t.id = "mutated"


class TestLookup:
    def test_get_templates_by_type(self):
        templates = get_templates(RelationType.CONTRADICTS)
        assert len(templates) == 3
        assert all(t.relation_type == RelationType.CONTRADICTS for t in templates)

    def test_get_default_template(self):
        default = get_default_template(RelationType.REQUIRES)
        assert default.relation_type == RelationType.REQUIRES
        # Default is first in list
        assert default == get_templates(RelationType.REQUIRES)[0]

    def test_select_template_with_category_hint(self):
        t = select_template(RelationType.CONTRADICTS, category_hint="architectural")
        assert t.category == "architectural"

    def test_select_template_default_no_hint(self):
        t = select_template(RelationType.CONTRADICTS)
        assert t == get_default_template(RelationType.CONTRADICTS)

    def test_select_template_hint_not_found_returns_default(self):
        t = select_template(RelationType.CONTRADICTS, category_hint="nonexistent")
        assert t == get_default_template(RelationType.CONTRADICTS)


class TestTemplateRendering:
    def test_render_contradicts(self):
        t = get_default_template(RelationType.CONTRADICTS)
        text = t.template.format(source="Retry", target="Fail Fast")
        assert "Retry" in text
        assert "Fail Fast" in text

    def test_render_requires(self):
        t = get_default_template(RelationType.REQUIRES)
        text = t.template.format(source="Timeout Config", target="Circuit Breaker")
        assert "Timeout Config" in text
        assert "Circuit Breaker" in text

    def test_render_all_templates(self):
        """Every template renders without error."""
        for t in EDGE_RULE_TEMPLATE_REGISTRY:
            text = t.template.format(source="A", target="B")
            assert "A" in text
            assert "B" in text
