"""Learner: the main class for bandit-based learning.

Composes strategy + reward model + state store. Exposes select(),
observe(), batch_observe(), top_arms(), decay_arm(), metrics(),
and posteriors(). Emits observability events.

All I/O methods are async. Use ``await Learner.create(config)`` to
construct (seed boost requires store I/O).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from qortex.learning.reward import RewardModel, TernaryReward
from qortex.learning.store import LearningStore, SqliteLearningStore
from qortex.learning.strategy import LearningStrategy, ThompsonSampling
from qortex.learning.types import (
    Arm,
    ArmOutcome,
    ArmState,
    LearnerConfig,
    RunTrace,
    SelectionResult,
    context_hash,
)
from qortex.observe import emit
from qortex.observe.events import (
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
)
from qortex.observe.tracing import traced


class Learner:
    """A bandit learner that selects arms and updates posteriors.

    Usage::

        learner = await Learner.create(LearnerConfig(name="prompts"))
        result = await learner.select(candidates, context={"task": "type-errors"}, k=3)
        # ... use selected arms ...
        await learner.observe(ArmOutcome(arm_id="prompt:v2", reward=1.0, outcome="accepted"))
    """

    def __init__(
        self,
        config: LearnerConfig,
        strategy: LearningStrategy | None = None,
        reward_model: RewardModel | None = None,
        store: LearningStore | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy or ThompsonSampling()
        self.reward_model = reward_model or TernaryReward()
        self.store = store or SqliteLearningStore(config.name, config.state_dir)
        self._seeded = False
        self._sessions: dict[str, RunTrace] = {}

    @classmethod
    async def create(
        cls,
        config: LearnerConfig,
        strategy: LearningStrategy | None = None,
        reward_model: RewardModel | None = None,
        store: LearningStore | None = None,
    ) -> Learner:
        """Create a Learner and apply seed boosts (requires async store I/O)."""
        learner = cls(config, strategy, reward_model, store)
        await learner._apply_seed_boosts()
        return learner

    async def _apply_seed_boosts(self) -> None:
        """Apply seed arm boosts. Idempotent — only boosts arms with zero pulls."""
        if self._seeded:
            return
        self._seeded = True
        for arm_id in self.config.seed_arms:
            state = await self.store.get(arm_id)
            if state.pulls == 0:
                state = ArmState(
                    alpha=self.config.seed_boost,
                    beta=1.0,
                    pulls=0,
                    total_reward=0.0,
                    last_updated=datetime.now(UTC).isoformat(),
                )
                await self.store.put(arm_id, state)
        if self.config.seed_arms:
            await self.store.save()

    @traced("learning.select")
    async def select(
        self,
        candidates: list[Arm],
        context: dict | None = None,
        k: int = 1,
        token_budget: int = 0,
    ) -> SelectionResult:
        """Select k arms from candidates."""
        ctx = context or {}
        states = {arm.id: await self.store.get(arm.id, ctx) for arm in candidates}

        result = self.strategy.select(
            candidates=candidates,
            states=states,
            k=k,
            config=self.config,
            token_budget=token_budget,
        )

        emit(
            LearningSelectionMade(
                learner=self.config.name,
                selected_count=len(result.selected),
                excluded_count=len(result.excluded),
                is_baseline=result.is_baseline,
                token_budget=result.token_budget,
                used_tokens=result.used_tokens,
            )
        )

        return result

    @traced("learning.observe")
    async def observe(
        self,
        outcome: ArmOutcome,
        context: dict | None = None,
    ) -> ArmState:
        """Record an observation and update posterior."""
        ctx = context or outcome.context
        reward = outcome.reward
        if outcome.outcome and not outcome.reward:
            reward = self.reward_model.compute(outcome.outcome)

        state = await self.store.get(outcome.arm_id, ctx)
        now = datetime.now(UTC).isoformat()

        new_state = self.strategy.update(outcome.arm_id, reward, state)
        new_state = ArmState(
            alpha=new_state.alpha,
            beta=new_state.beta,
            pulls=new_state.pulls,
            total_reward=new_state.total_reward,
            last_updated=now,
        )

        await self.store.put(outcome.arm_id, new_state, ctx)
        await self.store.save()

        ctx_hash = context_hash(ctx)

        emit(
            LearningObservationRecorded(
                learner=self.config.name,
                arm_id=outcome.arm_id,
                reward=reward,
                outcome=outcome.outcome,
                context_hash=ctx_hash,
            )
        )

        emit(
            LearningPosteriorUpdated(
                learner=self.config.name,
                arm_id=outcome.arm_id,
                alpha=new_state.alpha,
                beta=new_state.beta,
                pulls=new_state.pulls,
                mean=new_state.mean,
            )
        )

        return new_state

    @traced("learning.apply_credit_deltas")
    async def apply_credit_deltas(
        self,
        deltas: dict[str, dict[str, float]],
        context: dict | None = None,
    ) -> dict[str, ArmState]:
        """Apply causal credit deltas directly to arm posteriors.

        Unlike observe() which computes alpha/beta from a binary reward,
        this applies arbitrary deltas from CreditAssigner output.
        """
        ctx = context or {}
        now = datetime.now(UTC).isoformat()
        results: dict[str, ArmState] = {}

        for arm_id, delta in deltas.items():
            state = await self.store.get(arm_id, ctx)
            new_state = ArmState(
                alpha=max(state.alpha + delta.get("alpha_delta", 0.0), 0.01),
                beta=max(state.beta + delta.get("beta_delta", 0.0), 0.01),
                pulls=state.pulls + 1,
                total_reward=state.total_reward + delta.get("alpha_delta", 0.0),
                last_updated=now,
            )
            await self.store.put(arm_id, new_state, ctx)
            results[arm_id] = new_state

            emit(
                LearningPosteriorUpdated(
                    learner=self.config.name,
                    arm_id=arm_id,
                    alpha=new_state.alpha,
                    beta=new_state.beta,
                    pulls=new_state.pulls,
                    mean=new_state.mean,
                )
            )

        await self.store.save()
        return results

    async def reset(
        self,
        arm_ids: list[str] | None = None,
        context: dict | None = None,
    ) -> int:
        """Delete arm states and return count of entries removed.

        Useful for resetting poisoned posteriors or clearing stale data.
        """
        count = await self.store.delete(arm_ids=arm_ids, context=context)
        await self.store.save()
        return count

    async def batch_observe(
        self,
        outcomes: list[ArmOutcome],
        context: dict | None = None,
    ) -> dict[str, ArmState]:
        """Record multiple observations in a single call.

        Delegates to observe() for each outcome so events are emitted
        per-arm. Saves once at the end (observe already saves per call,
        but this keeps the contract simple for callers).
        """
        results: dict[str, ArmState] = {}
        for outcome in outcomes:
            results[outcome.arm_id] = await self.observe(outcome, context)
        return results

    async def top_arms(
        self,
        context: dict | None = None,
        k: int = 10,
    ) -> list[tuple[str, ArmState]]:
        """Return the top-k arms by posterior mean, descending.

        Derives from posteriors(). Returns (arm_id, ArmState) tuples
        so callers get both the ID and the full state.
        """
        all_states = await self.store.get_all(context)
        sorted_arms = sorted(
            all_states.items(),
            key=lambda pair: pair[1].mean,
            reverse=True,
        )
        return sorted_arms[:k]

    async def decay_arm(
        self,
        arm_id: str,
        decay_factor: float = 0.9,
        context: dict | None = None,
    ) -> ArmState:
        """Shrink an arm's learned signal toward the prior.

        Multiplies alpha and beta by decay_factor, weakening confidence
        while preserving the mean ratio. Useful when seed data changes
        and old signal should fade.

        Floors alpha/beta at 0.01 to avoid degenerate posteriors.
        """
        state = await self.store.get(arm_id, context)
        now = datetime.now(UTC).isoformat()
        new_state = ArmState(
            alpha=max(state.alpha * decay_factor, 0.01),
            beta=max(state.beta * decay_factor, 0.01),
            pulls=state.pulls,
            total_reward=state.total_reward * decay_factor,
            last_updated=now,
        )
        await self.store.put(arm_id, new_state, context)
        await self.store.save()
        return new_state

    async def posteriors(
        self,
        context: dict | None = None,
        arm_ids: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get current posteriors for arms."""
        all_states = await self.store.get_all(context)

        if arm_ids is not None:
            all_states = {k: v for k, v in all_states.items() if k in set(arm_ids)}

        return {
            arm_id: {
                **state.to_dict(),
                "mean": state.mean,
            }
            for arm_id, state in all_states.items()
        }

    async def metrics(self, window: int | None = None) -> dict[str, Any]:
        """Compute learning metrics across all contexts."""
        total_pulls = 0
        total_reward = 0.0
        arm_count = 0

        all_states = await self.store.get_all_states()
        for states in all_states.values():
            for state in states.values():
                total_pulls += state.pulls
                total_reward += state.total_reward
                arm_count += 1

        accuracy = total_reward / max(total_pulls, 1)
        explore_ratio = self.config.baseline_rate

        return {
            "learner": self.config.name,
            "total_pulls": total_pulls,
            "total_reward": total_reward,
            "accuracy": round(accuracy, 4),
            "arm_count": arm_count,
            "explore_ratio": explore_ratio,
        }

    def session_start(self, session_name: str) -> str:
        """Start a named learning session for tracking."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = RunTrace(
            session_id=session_id,
            learner=self.config.name,
            selected_arms=[],
            started_at=datetime.now(UTC).isoformat(),
        )
        return session_id

    def session_end(self, session_id: str) -> dict[str, Any]:
        """End a session and return summary."""
        trace = self._sessions.pop(session_id, None)
        if trace is None:
            return {"error": f"Session {session_id} not found"}

        trace.ended_at = datetime.now(UTC).isoformat()
        return {
            "session_id": session_id,
            "learner": trace.learner,
            "selected_arms": trace.selected_arms,
            "outcomes": trace.outcomes,
            "started_at": trace.started_at,
            "ended_at": trace.ended_at,
        }
