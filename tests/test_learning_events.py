"""Tests for observability event emission from the learning module."""

from __future__ import annotations

import asyncio

import pytest
from qortex.learning.learner import Learner
from qortex.learning.types import Arm, ArmOutcome, LearnerConfig
from qortex.observe import reset as obs_reset
from qortex.observe.emitter import configure
from qortex.observe.events import (
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
)
from qortex.observe.linker import QortexEventLinker


@pytest.fixture(autouse=True)
def _reset():
    obs_reset()
    yield
    obs_reset()


@pytest.fixture
def captured_events():
    """Capture emitted events for assertion."""
    events: list = []

    @QortexEventLinker.on(LearningSelectionMade)
    def _on_selection(event):
        events.append(("selection", event))

    @QortexEventLinker.on(LearningObservationRecorded)
    def _on_observation(event):
        events.append(("observation", event))

    @QortexEventLinker.on(LearningPosteriorUpdated)
    def _on_posterior(event):
        events.append(("posterior", event))

    configure()
    return events


@pytest.fixture
def learner(tmp_path):
    return Learner(
        LearnerConfig(
            name="event-test",
            baseline_rate=0.0,
            state_dir=str(tmp_path),
        )
    )


class TestSelectionEvents:
    async def test_selection_emits_event(self, learner, captured_events):
        candidates = [Arm(id="a"), Arm(id="b"), Arm(id="c")]
        await learner.select(candidates, k=2)

        # pyventus may schedule callbacks as tasks in async context — yield to let them run
        await asyncio.sleep(0.05)

        selection_events = [e for t, e in captured_events if t == "selection"]
        assert len(selection_events) == 1

        ev = selection_events[0]
        assert ev.learner == "event-test"
        assert ev.selected_count == 2
        assert ev.excluded_count == 1
        assert ev.is_baseline is False

    async def test_selection_with_token_budget(self, learner, captured_events):
        candidates = [Arm(id="a", token_cost=100), Arm(id="b", token_cost=200)]
        await learner.select(candidates, k=2, token_budget=500)

        await asyncio.sleep(0.05)

        ev = [e for t, e in captured_events if t == "selection"][0]
        assert ev.token_budget == 500


class TestObservationEvents:
    async def test_observe_emits_events(self, learner, captured_events):
        await learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))

        await asyncio.sleep(0.05)

        obs_events = [e for t, e in captured_events if t == "observation"]
        assert len(obs_events) == 1

        ev = obs_events[0]
        assert ev.learner == "event-test"
        assert ev.arm_id == "arm:a"
        assert ev.reward == 1.0
        assert ev.outcome == "accepted"

    async def test_observe_emits_posterior_update(self, learner, captured_events):
        await learner.observe(ArmOutcome(arm_id="arm:b", reward=0.0, outcome="rejected"))

        await asyncio.sleep(0.05)

        posterior_events = [e for t, e in captured_events if t == "posterior"]
        assert len(posterior_events) == 1

        ev = posterior_events[0]
        assert ev.learner == "event-test"
        assert ev.arm_id == "arm:b"
        assert ev.alpha == 1.0
        assert ev.beta == 2.0
        assert ev.pulls == 1


class TestEventFrozenness:
    def test_events_are_frozen(self):
        ev = LearningSelectionMade(
            learner="test",
            selected_count=1,
            excluded_count=2,
            is_baseline=False,
            token_budget=0,
            used_tokens=0,
        )
        with pytest.raises(AttributeError):
            ev.learner = "modified"

    def test_observation_event_frozen(self):
        ev = LearningObservationRecorded(
            learner="test",
            arm_id="a",
            reward=1.0,
            outcome="accepted",
            context_hash="abc",
        )
        with pytest.raises(AttributeError):
            ev.reward = 0.0

    def test_posterior_event_frozen(self):
        ev = LearningPosteriorUpdated(
            learner="test",
            arm_id="a",
            alpha=2.0,
            beta=1.0,
            pulls=1,
            mean=0.667,
        )
        with pytest.raises(AttributeError):
            ev.alpha = 99.0
