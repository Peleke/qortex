"""Exhaustive tests for InMemoryBackend."""

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)


@pytest.fixture
def backend():
    b = InMemoryBackend()
    b.connect()
    return b


@pytest.fixture
def sample_nodes():
    return [
        ConceptNode(
            id="err:circuit_breaker",
            name="Circuit Breaker",
            description="Pattern that prevents cascading failures",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="err:timeout",
            name="Timeout Configuration",
            description="Configuring timeouts for external calls",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="err:retry",
            name="Retry",
            description="Retrying failed operations",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="err:fail_fast",
            name="Fail Fast",
            description="Failing immediately on unrecoverable errors",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="fp:pure_function",
            name="Pure Function",
            description="Function with no side effects",
            domain="fp",
            source_id="book:ch1",
        ),
    ]


@pytest.fixture
def sample_edges():
    return [
        ConceptEdge(
            source_id="err:circuit_breaker",
            target_id="err:timeout",
            relation_type=RelationType.REQUIRES,
        ),
        ConceptEdge(
            source_id="err:retry",
            target_id="err:fail_fast",
            relation_type=RelationType.CONTRADICTS,
            confidence=0.9,
        ),
    ]


# -------------------------------------------------------------------------
# Connection
# -------------------------------------------------------------------------


class TestConnection:
    def test_connect_disconnect(self, backend):
        assert backend.is_connected()
        backend.disconnect()
        assert not backend.is_connected()

    def test_starts_disconnected(self):
        b = InMemoryBackend()
        assert not b.is_connected()


# -------------------------------------------------------------------------
# Domains
# -------------------------------------------------------------------------


class TestDomains:
    def test_create_domain(self, backend):
        domain = backend.create_domain("error_handling", "Error handling patterns")
        assert domain.name == "error_handling"
        assert domain.description == "Error handling patterns"

    def test_create_domain_idempotent(self, backend):
        d1 = backend.create_domain("test")
        d2 = backend.create_domain("test")
        assert d1.name == d2.name

    def test_get_domain(self, backend):
        backend.create_domain("test", "desc")
        d = backend.get_domain("test")
        assert d is not None
        assert d.description == "desc"

    def test_get_domain_not_found(self, backend):
        assert backend.get_domain("nonexistent") is None

    def test_list_domains(self, backend):
        backend.create_domain("a")
        backend.create_domain("b")
        domains = backend.list_domains()
        assert len(domains) == 2
        names = {d.name for d in domains}
        assert names == {"a", "b"}

    def test_delete_domain(self, backend):
        backend.create_domain("doomed")
        assert backend.delete_domain("doomed")
        assert backend.get_domain("doomed") is None

    def test_delete_domain_not_found(self, backend):
        assert not backend.delete_domain("ghost")

    def test_delete_domain_cascades_nodes_and_edges(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes:
            if node.domain == "error_handling":
                backend.add_node(node)
        for edge in sample_edges:
            backend.add_edge(edge)

        backend.delete_domain("error_handling")

        # Nodes gone
        assert backend.get_node("err:circuit_breaker") is None
        assert backend.get_node("err:timeout") is None

        # Edges gone
        edges = list(backend.get_edges("err:circuit_breaker"))
        assert len(edges) == 0

    def test_delete_domain_cascades_rules(self, backend):
        backend.create_domain("test")
        rule = ExplicitRule(
            id="rule:1", text="Test rule", domain="test", source_id="src"
        )
        backend.add_rule(rule)
        backend.delete_domain("test")
        assert backend.get_rules("test") == []


# -------------------------------------------------------------------------
# Nodes
# -------------------------------------------------------------------------


class TestNodes:
    def test_add_and_get_node(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        node = sample_nodes[0]
        backend.add_node(node)
        retrieved = backend.get_node("err:circuit_breaker")
        assert retrieved is not None
        assert retrieved.name == "Circuit Breaker"

    def test_get_node_not_found(self, backend):
        assert backend.get_node("ghost") is None

    def test_get_node_wrong_domain(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        backend.add_node(sample_nodes[0])
        # Node exists but wrong domain filter
        assert backend.get_node("err:circuit_breaker", domain="fp") is None

    def test_get_node_correct_domain(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        backend.add_node(sample_nodes[0])
        result = backend.get_node("err:circuit_breaker", domain="error_handling")
        assert result is not None

    def test_find_nodes_by_domain(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        backend.create_domain("fp")
        for node in sample_nodes:
            backend.add_node(node)

        err_nodes = list(backend.find_nodes(domain="error_handling"))
        assert len(err_nodes) == 4

        fp_nodes = list(backend.find_nodes(domain="fp"))
        assert len(fp_nodes) == 1

    def test_find_nodes_by_pattern(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)

        results = list(backend.find_nodes(name_pattern="*circuit*"))
        assert len(results) == 1
        assert results[0].name == "Circuit Breaker"

    def test_find_nodes_limit(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)

        results = list(backend.find_nodes(limit=2))
        assert len(results) == 2

    def test_find_nodes_all(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        backend.create_domain("fp")
        for node in sample_nodes:
            backend.add_node(node)

        all_nodes = list(backend.find_nodes())
        assert len(all_nodes) == 5

    def test_add_node_updates_domain_stats(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        backend.add_node(sample_nodes[0])
        domain = backend.get_domain("error_handling")
        assert domain.concept_count == 1


# -------------------------------------------------------------------------
# Edges
# -------------------------------------------------------------------------


class TestEdges:
    def test_add_and_get_edges_out(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)
        backend.add_edge(sample_edges[0])

        edges_out = list(backend.get_edges("err:circuit_breaker", direction="out"))
        assert len(edges_out) == 1
        assert edges_out[0].target_id == "err:timeout"

    def test_get_edges_in(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)
        backend.add_edge(sample_edges[0])

        edges_in = list(backend.get_edges("err:timeout", direction="in"))
        assert len(edges_in) == 1
        assert edges_in[0].source_id == "err:circuit_breaker"

    def test_get_edges_both(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)
        for edge in sample_edges:
            backend.add_edge(edge)

        # circuit_breaker: 1 outgoing to timeout
        edges = list(backend.get_edges("err:circuit_breaker", direction="both"))
        assert len(edges) == 1

    def test_get_edges_by_type(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)
        for edge in sample_edges:
            backend.add_edge(edge)

        requires = list(
            backend.get_edges("err:circuit_breaker", relation_type="requires")
        )
        assert len(requires) == 1

        contradicts = list(
            backend.get_edges("err:circuit_breaker", relation_type="contradicts")
        )
        assert len(contradicts) == 0

    def test_get_edges_empty(self, backend):
        assert list(backend.get_edges("nobody")) == []

    def test_add_edge_updates_domain_stats(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)
        backend.add_edge(sample_edges[0])
        domain = backend.get_domain("error_handling")
        assert domain.edge_count == 1


# -------------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------------


class TestRules:
    def test_add_and_get_rules(self, backend):
        backend.create_domain("test")
        rule = ExplicitRule(
            id="rule:1", text="Always handle errors", domain="test", source_id="src"
        )
        backend.add_rule(rule)
        rules = backend.get_rules("test")
        assert len(rules) == 1
        assert rules[0].text == "Always handle errors"

    def test_get_rules_all_domains(self, backend):
        backend.create_domain("a")
        backend.create_domain("b")
        backend.add_rule(
            ExplicitRule(id="r1", text="Rule A", domain="a", source_id="s")
        )
        backend.add_rule(
            ExplicitRule(id="r2", text="Rule B", domain="b", source_id="s")
        )
        all_rules = backend.get_rules()
        assert len(all_rules) == 2

    def test_add_rule_updates_domain_stats(self, backend):
        backend.create_domain("test")
        backend.add_rule(
            ExplicitRule(id="r1", text="Rule", domain="test", source_id="s")
        )
        domain = backend.get_domain("test")
        assert domain.rule_count == 1


# -------------------------------------------------------------------------
# Manifest Ingestion
# -------------------------------------------------------------------------


class TestManifestIngestion:
    def test_ingest_manifest(self, backend, sample_nodes, sample_edges):
        source = SourceMetadata(
            id="book:ch3",
            name="Chapter 3",
            source_type="markdown",
            path_or_url="/book/ch3.md",
        )
        rule = ExplicitRule(
            id="rule:err:1",
            text="Always configure timeouts",
            domain="error_handling",
            source_id="book:ch3",
        )
        manifest = IngestionManifest(
            source=source,
            domain="error_handling",
            concepts=sample_nodes[:4],
            edges=sample_edges,
            rules=[rule],
        )

        backend.ingest_manifest(manifest)

        # Domain auto-created
        domain = backend.get_domain("error_handling")
        assert domain is not None
        assert domain.concept_count == 4
        assert domain.edge_count == 2
        assert domain.rule_count == 1
        assert "book:ch3" in domain.source_ids

        # Nodes stored
        assert backend.get_node("err:circuit_breaker") is not None
        assert backend.get_node("err:timeout") is not None

        # Edges stored
        edges = list(backend.get_edges("err:circuit_breaker", direction="out"))
        assert len(edges) == 1

        # Rules stored
        rules = backend.get_rules("error_handling")
        assert len(rules) == 1

    def test_ingest_manifest_creates_domain(self, backend):
        source = SourceMetadata(
            id="s1", name="S1", source_type="text", path_or_url="/s1.txt"
        )
        manifest = IngestionManifest(
            source=source,
            domain="new_domain",
            concepts=[],
            edges=[],
            rules=[],
        )
        backend.ingest_manifest(manifest)
        assert backend.get_domain("new_domain") is not None

    def test_ingest_manifest_preserves_existing_domain(self, backend):
        backend.create_domain("existing", "Original description")
        source = SourceMetadata(
            id="s1", name="S1", source_type="text", path_or_url="/s1.txt"
        )
        manifest = IngestionManifest(
            source=source,
            domain="existing",
            concepts=[],
            edges=[],
            rules=[],
        )
        backend.ingest_manifest(manifest)
        domain = backend.get_domain("existing")
        assert domain.description == "Original description"


# -------------------------------------------------------------------------
# PPR (BFS Fallback)
# -------------------------------------------------------------------------


class TestPersonalizedPageRank:
    def test_ppr_seed_scores(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        for node in sample_nodes[:4]:
            backend.add_node(node)
        for edge in sample_edges:
            backend.add_edge(edge)

        scores = backend.personalized_pagerank(["err:circuit_breaker"])
        assert scores["err:circuit_breaker"] == 1.0
        # Timeout is 1-hop neighbor
        assert "err:timeout" in scores
        assert scores["err:timeout"] > 0

    def test_ppr_domain_filter(self, backend, sample_nodes, sample_edges):
        backend.create_domain("error_handling")
        backend.create_domain("fp")
        for node in sample_nodes:
            backend.add_node(node)
        for edge in sample_edges:
            backend.add_edge(edge)

        scores = backend.personalized_pagerank(
            ["err:circuit_breaker"], domain="error_handling"
        )
        # fp:pure_function should NOT be in results
        assert "fp:pure_function" not in scores

    def test_ppr_empty_seeds(self, backend):
        scores = backend.personalized_pagerank([])
        assert scores == {}

    def test_supports_mage(self, backend):
        assert not backend.supports_mage()


# -------------------------------------------------------------------------
# Checkpointing
# -------------------------------------------------------------------------


class TestCheckpointing:
    def test_checkpoint_and_restore(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        for node in sample_nodes[:2]:
            backend.add_node(node)

        cp_id = backend.checkpoint("before_change")

        # Modify state
        backend.add_node(sample_nodes[2])
        assert len(list(backend.find_nodes(domain="error_handling"))) == 3

        # Restore
        backend.restore(cp_id)
        assert len(list(backend.find_nodes(domain="error_handling"))) == 2

    def test_list_checkpoints(self, backend):
        backend.checkpoint("cp1")
        backend.checkpoint("cp2")
        cps = backend.list_checkpoints()
        assert len(cps) == 2
        names = {cp["name"] for cp in cps}
        assert names == {"cp1", "cp2"}

    def test_restore_nonexistent_raises(self, backend):
        with pytest.raises(ValueError):
            backend.restore("nonexistent-id")

    def test_checkpoint_partial_domains(self, backend, sample_nodes):
        backend.create_domain("error_handling")
        backend.create_domain("fp")
        for node in sample_nodes:
            backend.add_node(node)

        cp_id = backend.checkpoint("err_only", domains=["error_handling"])

        # Restore only restores error_handling nodes; fp is gone
        backend.restore(cp_id)
        err_nodes = list(backend.find_nodes(domain="error_handling"))
        assert len(err_nodes) == 4
        fp_nodes = list(backend.find_nodes(domain="fp"))
        assert len(fp_nodes) == 0


# -------------------------------------------------------------------------
# Query (unsupported)
# -------------------------------------------------------------------------


class TestQuery:
    def test_query_cypher_raises(self, backend):
        with pytest.raises(NotImplementedError):
            list(backend.query_cypher("MATCH (n) RETURN n"))

    def test_query_raises(self, backend):
        with pytest.raises(NotImplementedError):
            from qortex.core.backend import GraphPattern
            list(backend.query(GraphPattern()))
