"""Tests for the SKILL.md round-trip pipeline.

5-layer fidelity testing:
  L1: Parser — parse_skill_md produces correct SkillMdDocument fields
  L2: Renderer — same-format render produces YAML-safe, reparseable output
  L3: Cross-format — canonical → OpenClaw and vice versa preserves content
  L4: Ingestor — SkillMdIngestor produces correct manifests deterministically
  L5: Full pipeline — parse → ingest → project → reparse == original

Generates publishable fidelity-report.json as a test artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from qortex.projectors.skillmd import (
    SkillMdDocument,
    _extract_sections,
    _yaml_quote,
    parse_skill_md,
    render_claude_code_skill_md,
    render_openclaw_skill_md,
)
from qortex.projectors.sources.skill_md import SkillMdIngestor
from qortex.projectors.targets.claude_code_skill import ClaudeCodeSkillTarget
from qortex.projectors.targets.openclaw_skill import OpenClawSkillTarget

# =========================================================================
# Fixtures
# =========================================================================

CANONICAL_SKILL = dedent("""\
    ---
    name: test-skill
    description: A test skill for unit testing round-trip fidelity.
    ---

    # Test Skill

    You are a test agent. Follow these rules:

    ## Input

    The user will provide test data.

    - Parse the input carefully
    - Validate all fields

    ## Output

    Return JSON with the results.

    ```json
    {"status": "ok", "data": []}
    ```
    """)

CANONICAL_WITH_LICENSE = dedent("""\
    ---
    name: licensed-skill
    description: A skill with license and compatibility fields.
    license: MIT
    compatibility: claude-code
    ---

    # Licensed Skill

    Follow these instructions carefully.
    """)

OPENCLAW_SKILL = dedent("""\
    ---
    name: oc-skill
    description: An OpenClaw format skill.
    homepage: https://example.com/oc-skill
    license: MIT
    metadata: {"openclaw":{"emoji":"🧪","tags":["testing","demo"]}}
    ---

    # OpenClaw Skill

    You are a specialized agent.

    ## Rules

    - Always validate input
    - Return structured output
    """)

CLAWHUB_SKILL = dedent("""\
    ---
    name: hub-skill
    description: A ClawHub style skill.
    model: claude-3
    category: productivity
    version: 1.0.0
    keywords: test, demo
    ---

    # Hub Skill

    Do the thing.
    """)

CLAWDBOT_SKILL = dedent("""\
    ---
    name: bot-skill
    description: Uses clawdbot namespace.
    metadata: {"clawdbot":{"emoji":"🤖","version":"2.0"}}
    ---

    # Bot Skill

    Instructions here.
    """)

COLON_DESCRIPTION = dedent("""\
    ---
    name: colon-skill
    description: "Create docs for: (1) new projects, (2) existing code."
    ---

    # Colon Skill

    Handle colons in descriptions.
    """)

MULTILINE_BODY = dedent("""\
    ---
    name: multi-section
    description: A skill with many sections.
    ---

    # Multi-Section Skill

    Preamble content before any ## heading.

    ## Phase 1

    Do the first thing.

    - Step A
    - Step B

    ## Phase 2

    Do the second thing.

    ```python
    print("hello")
    ```

    ## Phase 3

    Final phase. Use careful judgment.
    """)

EMPTY_BODY = dedent("""\
    ---
    name: empty-body
    description: Skill with no body content.
    ---
    """)

MINIMAL = dedent("""\
    ---
    name: minimal
    description: Bare minimum.
    ---

    Content.
    """)


# =========================================================================
# L1: Parser Tests
# =========================================================================


class TestParser:
    """L1: parse_skill_md produces correct SkillMdDocument fields."""

    def test_canonical_basic(self):
        doc = parse_skill_md(CANONICAL_SKILL)
        assert doc.name == "test-skill"
        assert doc.description == "A test skill for unit testing round-trip fidelity."
        assert doc.source_format == "canonical"
        assert "# Test Skill" in doc.body
        assert "## Input" in doc.body
        assert "## Output" in doc.body

    def test_canonical_license(self):
        doc = parse_skill_md(CANONICAL_WITH_LICENSE)
        assert doc.name == "licensed-skill"
        assert doc.license == "MIT"
        assert doc.compatibility == "claude-code"
        assert doc.source_format == "canonical"

    def test_openclaw_format(self):
        doc = parse_skill_md(OPENCLAW_SKILL)
        assert doc.name == "oc-skill"
        assert doc.source_format == "openclaw"
        assert doc.homepage == "https://example.com/oc-skill"
        assert doc.openclaw_metadata == {"emoji": "🧪", "tags": ["testing", "demo"]}

    def test_clawhub_format(self):
        doc = parse_skill_md(CLAWHUB_SKILL)
        assert doc.name == "hub-skill"
        assert doc.source_format == "clawhub"
        assert doc.raw_frontmatter.get("model") == "claude-3"
        assert doc.raw_frontmatter.get("category") == "productivity"

    def test_clawdbot_namespace(self):
        doc = parse_skill_md(CLAWDBOT_SKILL)
        assert doc.name == "bot-skill"
        assert doc.source_format == "openclaw"
        assert doc.openclaw_metadata == {"emoji": "🤖", "version": "2.0"}

    def test_content_hash_deterministic(self):
        doc1 = parse_skill_md(CANONICAL_SKILL)
        doc2 = parse_skill_md(CANONICAL_SKILL)
        assert doc1.content_hash == doc2.content_hash

    def test_content_hash_changes_with_content(self):
        doc1 = parse_skill_md(CANONICAL_SKILL)
        modified = CANONICAL_SKILL.replace("test-skill", "changed-skill")
        doc2 = parse_skill_md(modified)
        assert doc1.content_hash != doc2.content_hash

    def test_skill_id_format(self):
        doc = parse_skill_md(CANONICAL_SKILL)
        assert doc.skill_id.startswith("skill_md:test-skill:")
        assert len(doc.skill_id.split(":")) == 3

    def test_missing_name_raises(self):
        bad = "---\ndescription: no name\n---\nBody"
        with pytest.raises(ValueError, match="missing required 'name'"):
            parse_skill_md(bad)

    def test_missing_description_raises(self):
        bad = "---\nname: no-desc\n---\nBody"
        with pytest.raises(ValueError, match="missing required 'description'"):
            parse_skill_md(bad)

    def test_no_frontmatter_raises(self):
        bad = "Just some text without frontmatter."
        with pytest.raises(ValueError, match="YAML frontmatter"):
            parse_skill_md(bad)

    def test_source_path_stored(self):
        path = Path("/fake/SKILL.md")
        doc = parse_skill_md(CANONICAL_SKILL, source_path=path)
        assert doc.source_path == path

    def test_colon_in_description(self):
        doc = parse_skill_md(COLON_DESCRIPTION)
        assert doc.description == "Create docs for: (1) new projects, (2) existing code."

    def test_empty_body_parsed(self):
        doc = parse_skill_md(EMPTY_BODY)
        assert doc.name == "empty-body"
        assert doc.body == ""


class TestExtractSections:
    def test_no_headings(self):
        sections = _extract_sections("Just a paragraph.")
        assert len(sections) == 1
        assert sections[0]["heading"] == ""
        assert sections[0]["content"] == "Just a paragraph."

    def test_single_heading(self):
        sections = _extract_sections("## Title\n\nContent.")
        assert len(sections) == 1
        assert sections[0]["heading"] == "Title"
        assert sections[0]["content"] == "Content."

    def test_preamble_and_headings(self):
        body = "Preamble.\n\n## H1\n\nContent1.\n\n## H2\n\nContent2."
        sections = _extract_sections(body)
        assert len(sections) == 3
        assert sections[0]["heading"] == ""
        assert sections[1]["heading"] == "H1"
        assert sections[2]["heading"] == "H2"

    def test_multi_section_skill(self):
        doc = parse_skill_md(MULTILINE_BODY)
        sections = _extract_sections(doc.body)
        headings = [s["heading"] for s in sections]
        assert "" in headings  # preamble
        assert "Phase 1" in headings
        assert "Phase 2" in headings
        assert "Phase 3" in headings


class TestYamlQuote:
    def test_plain_string(self):
        assert _yaml_quote("hello world") == "hello world"

    def test_colon_space_gets_quoted(self):
        result = _yaml_quote("Create docs for: (1) new")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_newline_gets_quoted(self):
        result = _yaml_quote("line1\nline2")
        assert result.startswith('"')

    def test_starts_with_indicator(self):
        for char in ["{", "[", "#", "&", "*", "!", ">", "%"]:
            result = _yaml_quote(f"{char}value")
            assert result.startswith('"'), f"Expected quoting for leading '{char}'"

    def test_no_unnecessary_quoting(self):
        # Regular descriptions should NOT be quoted
        assert _yaml_quote("A simple description") == "A simple description"
        assert _yaml_quote("Score things on a scale of 1-5") == "Score things on a scale of 1-5"


# =========================================================================
# L2: Renderer Tests (same-format round-trip)
# =========================================================================


class TestRenderers:
    """L2: rendered output is valid YAML and reparses to identical fields."""

    def test_canonical_round_trip(self):
        doc = parse_skill_md(CANONICAL_SKILL)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
            license=doc.license, compatibility=doc.compatibility,
            allowed_tools=doc.allowed_tools,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.description == doc2.description
        assert doc.body == doc2.body

    def test_canonical_with_license_round_trip(self):
        doc = parse_skill_md(CANONICAL_WITH_LICENSE)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
            license=doc.license, compatibility=doc.compatibility,
            allowed_tools=doc.allowed_tools,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.license == doc2.license
        assert doc.compatibility == doc2.compatibility

    def test_openclaw_round_trip(self):
        doc = parse_skill_md(OPENCLAW_SKILL)
        rendered = render_openclaw_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
            homepage=doc.homepage,
            openclaw_metadata=doc.openclaw_metadata or None,
            user_invocable=doc.user_invocable,
            disable_model_invocation=doc.disable_model_invocation,
            license=doc.license,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.description == doc2.description
        assert doc.body == doc2.body
        assert doc.homepage == doc2.homepage

    def test_colon_description_survives_round_trip(self):
        doc = parse_skill_md(COLON_DESCRIPTION)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.description == doc2.description

    def test_rendered_output_ends_with_newline(self):
        rendered = render_claude_code_skill_md(
            name="test", description="desc", body="body",
        )
        assert rendered.endswith("\n")

    def test_rendered_has_blank_line_after_frontmatter(self):
        rendered = render_claude_code_skill_md(
            name="test", description="desc", body="body",
        )
        lines = rendered.split("\n")
        # Find the closing ---
        close_idx = None
        for i, line in enumerate(lines):
            if i > 0 and line == "---":
                close_idx = i
                break
        assert close_idx is not None
        assert lines[close_idx + 1] == ""  # blank line after ---

    def test_openclaw_metadata_as_single_line_json(self):
        rendered = render_openclaw_skill_md(
            name="test",
            description="desc",
            body="body",
            openclaw_metadata={"emoji": "🧪"},
        )
        # Metadata should be on a single line
        for line in rendered.split("\n"):
            if line.startswith("metadata:"):
                assert "{" in line  # JSON on same line
                assert "\n" not in line[len("metadata:"):].strip()
                break
        else:
            pytest.fail("No metadata line found")

    def test_empty_body_renders(self):
        rendered = render_claude_code_skill_md(
            name="empty", description="no body", body="",
        )
        doc = parse_skill_md(rendered)
        assert doc.name == "empty"
        assert doc.body == ""

    def test_minimal_round_trip(self):
        doc = parse_skill_md(MINIMAL)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.body == doc2.body


# =========================================================================
# L3: Cross-Format Tests
# =========================================================================


class TestCrossFormat:
    """L3: converting between formats preserves semantic content."""

    def test_canonical_to_openclaw(self):
        """Canonical skill re-emitted as OpenClaw preserves name/desc/body."""
        doc = parse_skill_md(CANONICAL_SKILL)
        rendered = render_openclaw_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
            openclaw_metadata={"emoji": "🧪"},
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.description == doc2.description
        assert doc.body == doc2.body
        assert doc2.source_format == "openclaw"

    def test_openclaw_to_canonical(self):
        """OpenClaw skill re-emitted as canonical preserves name/desc/body."""
        doc = parse_skill_md(OPENCLAW_SKILL)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.description == doc2.description
        assert doc.body == doc2.body
        assert doc2.source_format == "canonical"

    def test_clawhub_to_canonical(self):
        """ClawHub skill re-emitted as canonical preserves core fields."""
        doc = parse_skill_md(CLAWHUB_SKILL)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
        )
        doc2 = parse_skill_md(rendered)
        assert doc.name == doc2.name
        assert doc.description == doc2.description
        assert doc.body == doc2.body


# =========================================================================
# L4: Ingestor Tests
# =========================================================================


class TestIngestor:
    """L4: SkillMdIngestor produces correct, deterministic manifests."""

    def test_ingest_canonical(self, tmp_path: Path):
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(CANONICAL_SKILL)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)

        assert manifest.domain == "skill:test-skill"
        assert manifest.source.source_type == "skill_md"
        assert manifest.source.name == "test-skill"
        assert len(manifest.concepts) > 0
        assert len(manifest.edges) > 0
        assert len(manifest.rules) > 0

    def test_ingest_deterministic(self, tmp_path: Path):
        """Same input always produces same manifest."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(CANONICAL_SKILL)

        ingestor = SkillMdIngestor()
        m1 = ingestor.ingest(skill_path)
        m2 = ingestor.ingest(skill_path)

        assert m1.source.content_hash == m2.source.content_hash
        assert len(m1.concepts) == len(m2.concepts)
        assert len(m1.edges) == len(m2.edges)
        assert len(m1.rules) == len(m2.rules)
        for c1, c2 in zip(m1.concepts, m2.concepts):
            assert c1.id == c2.id
            assert c1.name == c2.name

    def test_root_concept_has_body(self, tmp_path: Path):
        """Root concept stores full body for re-emission."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(CANONICAL_SKILL)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)
        root = manifest.concepts[0]
        assert root.properties.get("body") is not None
        assert "# Test Skill" in root.properties["body"]

    def test_section_concepts_created(self, tmp_path: Path):
        """Each ## heading produces a section concept."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(MULTILINE_BODY)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)

        section_names = [c.name for c in manifest.concepts[1:]]
        assert "Phase 1" in section_names
        assert "Phase 2" in section_names
        assert "Phase 3" in section_names

    def test_part_of_edges(self, tmp_path: Path):
        """Each section concept has a PART_OF edge to root."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(MULTILINE_BODY)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)

        root_id = manifest.concepts[0].id
        part_of_targets = [e.target_id for e in manifest.edges]
        assert all(t == root_id for t in part_of_targets)
        # Number of edges == number of sections
        assert len(manifest.edges) == len(manifest.concepts) - 1

    def test_domain_override(self, tmp_path: Path):
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(CANONICAL_SKILL)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path, domain="custom:domain")
        assert manifest.domain == "custom:domain"

    def test_instructional_sections_become_rules(self, tmp_path: Path):
        """Sections with code blocks, lists, or imperative verbs become rules."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(MULTILINE_BODY)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)

        # Should have primary rule + at least some section rules
        assert len(manifest.rules) >= 2
        # Primary rule references the skill description
        assert "multi-section" in manifest.rules[0].text

    def test_batch_ingest_directory(self, tmp_path: Path):
        """ingest_directory finds and ingests all SKILL.md files."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "SKILL.md").write_text(CANONICAL_SKILL)
        (tmp_path / "b" / "SKILL.md").write_text(MINIMAL)

        ingestor = SkillMdIngestor()
        manifests = ingestor.ingest_directory(tmp_path)

        assert len(manifests) == 2
        names = {m.source.name for m in manifests}
        assert "test-skill" in names
        assert "minimal" in names

    def test_nonrecursive_ingest(self, tmp_path: Path):
        """Non-recursive mode only finds SKILL.md in immediate directory."""
        (tmp_path / "SKILL.md").write_text(CANONICAL_SKILL)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "SKILL.md").write_text(MINIMAL)

        ingestor = SkillMdIngestor()
        manifests = ingestor.ingest_directory(tmp_path, recursive=False)

        assert len(manifests) == 1
        assert manifests[0].source.name == "test-skill"


# =========================================================================
# L5: Full Pipeline Round-Trip Tests
# =========================================================================


class TestFullPipeline:
    """L5: parse → ingest → project → reparse preserves content."""

    def test_round_trip_claude_code(self, tmp_path: Path):
        """Full pipeline: SKILL.md → ingest → ClaudeCodeSkillTarget → reparse."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(CANONICAL_SKILL)

        # Parse original
        original = parse_skill_md(CANONICAL_SKILL)

        # Ingest
        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)

        # The root concept stores the body
        root = manifest.concepts[0]
        body = root.properties["body"]

        # Re-render as canonical
        rendered = render_claude_code_skill_md(
            name=original.name,
            description=original.description,
            body=body,
        )

        # Reparse and compare
        reparsed = parse_skill_md(rendered)
        assert reparsed.name == original.name
        assert reparsed.description == original.description
        assert reparsed.body == original.body

    def test_round_trip_openclaw(self, tmp_path: Path):
        """Full pipeline: OpenClaw SKILL.md → ingest → OpenClawSkillTarget → reparse."""
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text(OPENCLAW_SKILL)

        original = parse_skill_md(OPENCLAW_SKILL)

        ingestor = SkillMdIngestor()
        manifest = ingestor.ingest(skill_path)

        root = manifest.concepts[0]
        body = root.properties["body"]

        rendered = render_openclaw_skill_md(
            name=original.name,
            description=original.description,
            body=body,
            homepage=root.properties.get("homepage"),
            openclaw_metadata=root.properties.get("openclaw"),
        )

        reparsed = parse_skill_md(rendered)
        assert reparsed.name == original.name
        assert reparsed.description == original.description
        assert reparsed.body == original.body
        assert reparsed.source_format == "openclaw"

    def test_target_serialize_produces_valid_skill(self):
        """ClaudeCodeSkillTarget.serialize() output is a valid SKILL.md."""
        from qortex.core.models import Rule

        rules = [
            Rule(
                id="r1",
                text="Always validate input before processing.",
                domain="skill:test",
                derivation="explicit",
                source_concepts=["c1"],
                confidence=0.9,
            ),
            Rule(
                id="r2",
                text="Return structured JSON output.",
                domain="skill:test",
                derivation="explicit",
                source_concepts=["c1"],
                confidence=0.9,
            ),
        ]

        target = ClaudeCodeSkillTarget(skill_name="test-out")
        result = target.serialize(rules)
        assert len(result) == 1
        assert result[0]["path"] == "test-out/SKILL.md"

        # The content should be parseable
        doc = parse_skill_md(result[0]["content"])
        assert doc.name == "test-out"
        assert "validate input" in doc.description

    def test_target_openclaw_produces_valid_skill(self):
        """OpenClawSkillTarget.serialize() output is a valid SKILL.md."""
        from qortex.core.models import Rule

        rules = [
            Rule(
                id="r1",
                text="Do the thing.",
                domain="skill:oc-out",
                derivation="explicit",
                source_concepts=["c1"],
                confidence=0.9,
            ),
        ]

        target = OpenClawSkillTarget(skill_name="oc-out")
        result = target.serialize(rules)
        assert len(result) == 1

        doc = parse_skill_md(result[0]["content"])
        assert doc.name == "oc-out"
        assert doc.source_format == "openclaw"

    def test_target_groups_by_domain(self):
        """Without skill_name, target groups rules by domain."""
        from qortex.core.models import Rule

        rules = [
            Rule(id="r1", text="Rule A", domain="skill:alpha",
                 derivation="explicit", source_concepts=[], confidence=0.9),
            Rule(id="r2", text="Rule B", domain="skill:beta",
                 derivation="explicit", source_concepts=[], confidence=0.9),
        ]

        target = ClaudeCodeSkillTarget()
        result = target.serialize(rules)
        assert len(result) == 2
        paths = {r["path"] for r in result}
        assert "alpha/SKILL.md" in paths
        assert "beta/SKILL.md" in paths

    def test_empty_rules_returns_empty(self):
        target = ClaudeCodeSkillTarget()
        assert target.serialize([]) == []


# =========================================================================
# Normalization Tests
# =========================================================================


class TestNormalization:
    """Verify the 3 normalization transforms are applied correctly."""

    def test_colon_space_gets_quoted(self):
        rendered = render_claude_code_skill_md(
            name="test",
            description="Create docs for: (1) new, (2) old.",
            body="Body",
        )
        # Should reparse without error
        doc = parse_skill_md(rendered)
        assert doc.description == "Create docs for: (1) new, (2) old."

    def test_trailing_newline_added(self):
        rendered = render_claude_code_skill_md(
            name="test", description="desc", body="body",
        )
        assert rendered.endswith("\n")

    def test_single_blank_line_after_fence(self):
        rendered = render_claude_code_skill_md(
            name="test", description="desc", body="# Heading",
        )
        lines = rendered.split("\n")
        fence_idx = None
        for i, line in enumerate(lines):
            if i > 0 and line == "---":
                fence_idx = i
                break
        assert fence_idx is not None
        assert lines[fence_idx + 1] == ""
        assert lines[fence_idx + 2] == "# Heading"

    def test_no_quoting_without_colon_space(self):
        rendered = render_claude_code_skill_md(
            name="test",
            description="A simple description without special chars",
            body="Body",
        )
        for line in rendered.split("\n"):
            if line.startswith("description:"):
                # Should NOT be quoted
                assert not line.endswith('"')
                break


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_unicode_in_body(self):
        skill = dedent("""\
            ---
            name: unicode-skill
            description: Handles unicode.
            ---

            # Unicode Skill

            Emoji: 🧪 🤖 🎯
            CJK: 你好世界
            Arabic: مرحبا
            """)
        doc = parse_skill_md(skill)
        rendered = render_claude_code_skill_md(
            name=doc.name, description=doc.description, body=doc.body,
        )
        doc2 = parse_skill_md(rendered)
        assert "🧪" in doc2.body
        assert "你好世界" in doc2.body

    def test_description_with_quotes(self):
        skill = dedent("""\
            ---
            name: quote-skill
            description: Says "hello" and 'goodbye'.
            ---

            Body.
            """)
        doc = parse_skill_md(skill)
        assert '"hello"' in doc.description or "hello" in doc.description

    def test_very_long_description(self):
        long_desc = "A " * 500 + "skill."
        rendered = render_claude_code_skill_md(
            name="long", description=long_desc, body="Body",
        )
        doc = parse_skill_md(rendered)
        assert doc.description == long_desc

    def test_body_with_yaml_like_content(self):
        """Body containing --- should not confuse the parser."""
        skill = dedent("""\
            ---
            name: yaml-body
            description: Body has YAML-like content.
            ---

            # YAML Body

            Here is some YAML:

            ```yaml
            key: value
            nested:
              - item1
              - item2
            ```

            End of body.
            """)
        doc = parse_skill_md(skill)
        assert "key: value" in doc.body

    def test_is_instructional_detection(self):
        """Verify _is_instructional correctly classifies content."""
        from qortex.projectors.sources.skill_md import SkillMdIngestor

        ingestor = SkillMdIngestor()

        # Code blocks
        assert ingestor._is_instructional("```python\nprint('hi')\n```")
        # Lists
        assert ingestor._is_instructional("- Step one\n- Step two")
        assert ingestor._is_instructional("1. First\n2. Second")
        # Imperative verbs
        assert ingestor._is_instructional("Use this method for testing.")
        assert ingestor._is_instructional("Always validate input.")
        # Non-instructional
        assert not ingestor._is_instructional("")
        assert not ingestor._is_instructional("   ")
        assert not ingestor._is_instructional("This is a concept about things.")


# =========================================================================
# Fidelity Report Generation (artifact for publishing)
# =========================================================================


class TestFidelityReport:
    """Generate publishable fidelity-report.json as test artifact."""

    SKILLS_DIR = Path("/Users/peleke/Documents/Projects/skills/skills")

    @pytest.mark.skipif(
        not Path("/Users/peleke/Documents/Projects/skills/skills").exists(),
        reason="Skills corpus not available",
    )
    def test_corpus_fidelity_report(self, tmp_path: Path):
        """Batch test all available SKILL.md files and emit a report."""
        report = {
            "total": 0,
            "semantic_match": 0,
            "byte_identical": 0,
            "parse_errors": 0,
            "reparse_errors": 0,
            "skills": [],
        }

        for skill_md in sorted(self.SKILLS_DIR.rglob("SKILL.md")):
            original = skill_md.read_text()
            entry: dict = {
                "name": skill_md.parent.name,
                "path": str(skill_md),
                "original_bytes": len(original),
            }
            report["total"] += 1

            try:
                doc = parse_skill_md(original, skill_md)
            except ValueError as e:
                entry["status"] = "parse_error"
                entry["error"] = str(e)[:200]
                report["parse_errors"] += 1
                report["skills"].append(entry)
                continue

            entry["format"] = doc.source_format

            if doc.source_format in ("canonical", "clawhub"):
                rendered = render_claude_code_skill_md(
                    name=doc.name, description=doc.description, body=doc.body,
                    license=doc.license, compatibility=doc.compatibility,
                    allowed_tools=doc.allowed_tools,
                )
            else:
                rendered = render_openclaw_skill_md(
                    name=doc.name, description=doc.description, body=doc.body,
                    homepage=doc.homepage,
                    openclaw_metadata=doc.openclaw_metadata or None,
                    user_invocable=doc.user_invocable,
                    disable_model_invocation=doc.disable_model_invocation,
                    license=doc.license,
                )

            entry["rendered_bytes"] = len(rendered)
            entry["byte_diff"] = len(rendered) - len(original)
            entry["byte_identical"] = original == rendered

            if original == rendered:
                report["byte_identical"] += 1

            try:
                doc2 = parse_skill_md(rendered)
            except ValueError as e:
                entry["status"] = "reparse_error"
                entry["error"] = str(e)[:200]
                report["reparse_errors"] += 1
                report["skills"].append(entry)
                continue

            semantic_ok = (
                doc.name == doc2.name
                and doc.description == doc2.description
                and doc.body == doc2.body
            )
            entry["semantic_match"] = semantic_ok
            entry["status"] = "pass" if semantic_ok else "fail"

            if semantic_ok:
                report["semantic_match"] += 1
            else:
                diffs = []
                if doc.name != doc2.name:
                    diffs.append("name")
                if doc.description != doc2.description:
                    diffs.append("description")
                if doc.body != doc2.body:
                    diffs.append("body")
                entry["diff_fields"] = diffs

            report["skills"].append(entry)

        # Write the report artifact
        report_path = tmp_path / "fidelity-report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Also write to a stable location for publishing
        stable_path = Path("/Users/peleke/Documents/Projects/qortex/tests/artifacts")
        stable_path.mkdir(parents=True, exist_ok=True)
        (stable_path / "fidelity-report.json").write_text(
            json.dumps(report, indent=2)
        )

        # Assertions
        assert report["parse_errors"] == 0, f"{report['parse_errors']} skills failed to parse"
        assert report["reparse_errors"] == 0, f"{report['reparse_errors']} skills failed to reparse"
        assert report["semantic_match"] == report["total"], (
            f"Semantic match: {report['semantic_match']}/{report['total']}"
        )

        # Print summary for CI output
        print(f"\n--- Fidelity Report ---")
        print(f"Total: {report['total']}")
        print(f"Semantic match: {report['semantic_match']}/{report['total']} (100%)")
        print(f"Byte identical: {report['byte_identical']}/{report['total']}")
        print(f"Report written to: {stable_path / 'fidelity-report.json'}")
