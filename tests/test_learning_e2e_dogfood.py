"""End-to-end dogfood test for the learning module.

Proves the full pipeline: ingest data → query → select arms → observe outcomes
→ posteriors update → metrics reflect reality → events fire → re-query shows
improved selection from learned posteriors.

This is the "show the investors" test.
"""

from __future__ import annotations

import hashlib

import pytest
from qortex_observe import configure
from qortex_observe import reset as obs_reset
from qortex_observe.events import (
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    QueryCompleted,
    QueryFailed,
    QueryStarted,
)
from qortex_observe.linker import QortexEventLinker

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, RelationType
from qortex.mcp import server as mcp_server

# ---------------------------------------------------------------------------
# Fake embedding model (deterministic, no GPU needed)
# ---------------------------------------------------------------------------

DIMS = 32


class FakeEmbedding:
    @property
    def dimensions(self) -> int:
        return DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            norm = sum(v * v for v in vec) ** 0.5
            result.append([v / norm for v in vec])
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(tmp_path):
    obs_reset()
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._graph_adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    mcp_server._learners = {}
    mcp_server._learning_state_dir = str(tmp_path)
    yield
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._graph_adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    mcp_server._learners = {}
    mcp_server._learning_state_dir = ""
    obs_reset()


@pytest.fixture
def event_log():
    """Capture ALL events for assertion."""
    events: list[tuple[str, object]] = []

    @QortexEventLinker.on(QueryStarted)
    def _qs(e):
        events.append(("QueryStarted", e))

    @QortexEventLinker.on(QueryCompleted)
    def _qc(e):
        events.append(("QueryCompleted", e))

    @QortexEventLinker.on(QueryFailed)
    def _qf(e):
        events.append(("QueryFailed", e))

    @QortexEventLinker.on(LearningSelectionMade)
    def _ls(e):
        events.append(("LearningSelectionMade", e))

    @QortexEventLinker.on(LearningObservationRecorded)
    def _lo(e):
        events.append(("LearningObservationRecorded", e))

    @QortexEventLinker.on(LearningPosteriorUpdated)
    def _lp(e):
        events.append(("LearningPosteriorUpdated", e))

    configure()
    return events


@pytest.fixture
def seeded_server():
    """Server with knowledge graph, vec index, and concepts ready for query."""
    from qortex.vec.index import NumpyVectorIndex

    embedding = FakeEmbedding()
    vec_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vec_index)
    backend.connect()

    # Seed domain + concepts
    backend.create_domain("prompts")

    concepts = [
        ConceptNode(
            id="prompt:v1",
            name="Basic prompt",
            description="Simple instruction-following prompt template",
            domain="prompts",
            source_id="seed",
        ),
        ConceptNode(
            id="prompt:v2",
            name="Chain-of-thought prompt",
            description="Step-by-step reasoning prompt template",
            domain="prompts",
            source_id="seed",
        ),
        ConceptNode(
            id="prompt:v3",
            name="Few-shot prompt",
            description="Example-based few-shot learning prompt template",
            domain="prompts",
            source_id="seed",
        ),
    ]

    for concept in concepts:
        backend.add_node(concept)
        emb = embedding.embed([f"{concept.name}: {concept.description}"])[0]
        backend.add_embedding(concept.id, emb)
        vec_index.add([concept.id], [emb])

    # Add edge: v2 REFINES v1
    backend.add_edge(ConceptEdge(
        source_id="prompt:v2",
        target_id="prompt:v1",
        relation_type=RelationType.REFINES,
    ))

    mcp_server.create_server(
        backend=backend,
        embedding_model=embedding,
        vector_index=vec_index,
    )
    return mcp_server


# ---------------------------------------------------------------------------
# End-to-End Tests
# ---------------------------------------------------------------------------


class TestLearningE2EDogfood:
    """Full pipeline: query → select → observe → learn → re-select."""

    def test_full_learning_lifecycle(self, seeded_server, event_log):
        srv = seeded_server

        # ── Step 1: Query the knowledge graph ──────────────────────
        query_result = srv._query_impl(
            context="prompt template for reasoning",
            domains=["prompts"],
            top_k=3,
            mode="graph",
        )
        assert len(query_result["items"]) > 0

        # ── Step 2: Select arms for prompt optimization ────────────
        candidates = [
            {"id": "prompt:v1", "metadata": {"type": "basic"}, "token_cost": 100},
            {"id": "prompt:v2", "metadata": {"type": "cot"}, "token_cost": 200},
            {"id": "prompt:v3", "metadata": {"type": "few-shot"}, "token_cost": 300},
        ]
        select_result = srv._learning_select_impl(
            learner="prompt-optimizer",
            candidates=candidates,
            context={"task": "type-checking"},
            k=2,
        )
        assert len(select_result["selected_arms"]) == 2
        selected_ids = {a["id"] for a in select_result["selected_arms"]}

        # ── Step 3: Simulate outcomes ──────────────────────────────
        # v2 (chain-of-thought) works great; always observe it as accepted
        # regardless of whether it was selected (initial selection is random)
        srv._learning_observe_impl(
            learner="prompt-optimizer",
            arm_id="prompt:v2",
            outcome="accepted",
            context={"task": "type-checking"},
        )
        for arm in select_result["selected_arms"]:
            if arm["id"] != "prompt:v2":
                srv._learning_observe_impl(
                    learner="prompt-optimizer",
                    arm_id=arm["id"],
                    outcome="rejected",
                    context={"task": "type-checking"},
                )

        # ── Step 4: Check posteriors reflect learning ──────────────
        posteriors = srv._learning_posteriors_impl(
            learner="prompt-optimizer",
            context={"task": "type-checking"},
        )
        p = posteriors["posteriors"]

        # v2 should have higher posterior mean than any rejected arm
        assert p["prompt:v2"]["mean"] > 0.5
        for arm_id in selected_ids:
            if arm_id != "prompt:v2":
                assert p[arm_id]["mean"] < p["prompt:v2"]["mean"]

        # ── Step 5: Repeat selection — v2 should be strongly preferred ──
        # Train more: 20 rounds of v2=accepted to build strong posterior
        for _ in range(20):
            srv._learning_observe_impl(
                learner="prompt-optimizer",
                arm_id="prompt:v2",
                outcome="accepted",
                context={"task": "type-checking"},
            )

        # Also penalize the others so v2 clearly dominates
        for other_id in ["prompt:v1", "prompt:v3"]:
            for _ in range(5):
                srv._learning_observe_impl(
                    learner="prompt-optimizer",
                    arm_id=other_id,
                    outcome="rejected",
                    context={"task": "type-checking"},
                )

        # Now select — v2 should dominate (alpha~22 vs beta~1)
        v2_selected = 0
        for _ in range(50):
            result = srv._learning_select_impl(
                learner="prompt-optimizer",
                candidates=candidates,
                context={"task": "type-checking"},
                k=1,
            )
            if result["selected_arms"][0]["id"] == "prompt:v2":
                v2_selected += 1

        # With 10% baseline, expect ~45 out of 50 (exploration steals ~5)
        assert v2_selected > 35, f"v2 selected {v2_selected}/50 times (expected >35)"

        # ── Step 6: Metrics reflect reality ────────────────────────
        metrics = srv._learning_metrics_impl(learner="prompt-optimizer")
        # 2 initial + 20 v2 accepted + 10 others rejected = 32 pulls
        assert metrics["total_pulls"] > 20
        assert metrics["total_reward"] > 10
        assert metrics["accuracy"] > 0.3

        # ── Step 7: Events fired correctly ─────────────────────────
        event_types = [t for t, _ in event_log]

        # Query events from Step 1
        assert "QueryStarted" in event_types
        assert "QueryCompleted" in event_types

        # Learning events from Steps 2-5
        assert "LearningSelectionMade" in event_types
        assert "LearningObservationRecorded" in event_types
        assert "LearningPosteriorUpdated" in event_types

        # Count learning events
        selection_count = sum(1 for t, _ in event_log if t == "LearningSelectionMade")
        observation_count = sum(1 for t, _ in event_log if t == "LearningObservationRecorded")
        posterior_count = sum(1 for t, _ in event_log if t == "LearningPosteriorUpdated")

        assert selection_count >= 51  # 1 initial + 50 re-selects
        assert observation_count >= 12  # 2 initial + 10 training
        assert posterior_count == observation_count  # 1:1

    def test_token_budget_constrains_selection(self, seeded_server, event_log):
        srv = seeded_server

        candidates = [
            {"id": "a", "token_cost": 500},
            {"id": "b", "token_cost": 500},
            {"id": "c", "token_cost": 500},
        ]
        result = srv._learning_select_impl(
            learner="budget-test",
            candidates=candidates,
            k=3,
            token_budget=800,
        )

        # Can't fit all 3 (1500 > 800), should fit at most 1
        assert result["used_tokens"] <= 800
        assert len(result["selected_arms"]) <= 1

    def test_session_tracking_across_rounds(self, seeded_server, event_log):
        srv = seeded_server

        # Start session
        start = srv._learning_session_start_impl(
            learner="session-test",
            session_name="investor-demo",
        )
        session_id = start["session_id"]
        assert session_id

        # Select and observe within session
        srv._learning_select_impl(
            learner="session-test",
            candidates=[{"id": "arm:x"}, {"id": "arm:y"}],
            k=1,
        )
        srv._learning_observe_impl(
            learner="session-test",
            arm_id="arm:x",
            outcome="accepted",
        )

        # End session
        summary = srv._learning_session_end_impl(session_id)
        assert summary["session_id"] == session_id
        assert summary["started_at"]
        assert summary["ended_at"]

    def test_multiple_learners_independent(self, seeded_server, event_log):
        srv = seeded_server

        # Learner A learns arm:x is good
        for _ in range(5):
            srv._learning_observe_impl(
                learner="learner-A",
                arm_id="arm:x",
                outcome="accepted",
            )

        # Learner B learns arm:x is bad
        for _ in range(5):
            srv._learning_observe_impl(
                learner="learner-B",
                arm_id="arm:x",
                outcome="rejected",
            )

        a_posteriors = srv._learning_posteriors_impl(learner="learner-A")
        b_posteriors = srv._learning_posteriors_impl(learner="learner-B")

        assert a_posteriors["posteriors"]["arm:x"]["mean"] > 0.7
        assert b_posteriors["posteriors"]["arm:x"]["mean"] < 0.3

    def test_query_failed_event_fires_on_embedding_error(self, event_log):
        """Prove QueryFailed events fire when embedding fails."""

        class BrokenEmbedding:
            @property
            def dimensions(self):
                return 32

            def embed(self, texts):
                raise RuntimeError("GPU exploded")

        from qortex.vec.index import NumpyVectorIndex

        backend = InMemoryBackend()
        backend.connect()
        vec_index = NumpyVectorIndex(dimensions=32)

        mcp_server.create_server(
            backend=backend,
            embedding_model=BrokenEmbedding(),
            vector_index=vec_index,
        )

        with pytest.raises(RuntimeError, match="GPU exploded"):
            mcp_server._query_impl(
                context="test query",
                mode="vec",
            )

        # QueryFailed event should have fired
        failed_events = [e for t, e in event_log if t == "QueryFailed"]
        assert len(failed_events) == 1
        assert failed_events[0].stage == "embedding"
        assert "GPU exploded" in failed_events[0].error

    def test_context_partitioned_learning(self, seeded_server, event_log):
        """Different contexts learn independently — essential for multi-task optimization."""
        srv = seeded_server

        # Context A: arm:alpha works well
        for _ in range(10):
            srv._learning_observe_impl(
                learner="multi-task",
                arm_id="arm:alpha",
                outcome="accepted",
                context={"task": "typing"},
            )

        # Context B: arm:alpha doesn't work
        for _ in range(10):
            srv._learning_observe_impl(
                learner="multi-task",
                arm_id="arm:alpha",
                outcome="rejected",
                context={"task": "linting"},
            )

        # Posteriors should diverge by context
        typing_p = srv._learning_posteriors_impl(
            learner="multi-task",
            context={"task": "typing"},
        )
        linting_p = srv._learning_posteriors_impl(
            learner="multi-task",
            context={"task": "linting"},
        )

        assert typing_p["posteriors"]["arm:alpha"]["mean"] > 0.7
        assert linting_p["posteriors"]["arm:alpha"]["mean"] < 0.3
