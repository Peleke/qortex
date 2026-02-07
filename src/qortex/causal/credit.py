"""CreditAssigner — causal credit assignment via DAG ancestry.

Assigns credit to concepts when a rule receives a reward signal.
Direct rule concepts get full credit; ancestors get decayed credit
proportional to path strength and hop count. Output feeds directly
into buildlog Thompson Sampling posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dag import CausalDAG
from .types import CreditAssignment


@dataclass
class CreditAssigner:
    """Assigns causal credit along DAG paths.

    Credit flows backward from rewarded concepts to their causal
    ancestors, decaying by ``decay_factor`` per hop and weighted
    by edge strength.
    """

    dag: CausalDAG
    decay_factor: float = 0.5
    min_credit: float = 0.01
    max_depth: int = 50

    def assign_credit(
        self,
        rule_concept_ids: list[str],
        reward: float,
        magnitude: float = 1.0,
    ) -> list[CreditAssignment]:
        """Assign credit to concepts given a reward signal.

        Args:
            rule_concept_ids: Concept IDs directly linked to the rewarded rule.
            reward: Reward value (positive = success, negative = failure).
            magnitude: Scaling factor for credit.

        Returns:
            List of CreditAssignment for all concepts receiving meaningful credit.
        """
        assignments: list[CreditAssignment] = []
        seen: set[str] = set()

        base_credit = reward * magnitude

        # Direct credit for rule concepts
        for cid in rule_concept_ids:
            if cid not in self.dag.node_ids or cid in seen:
                continue
            seen.add(cid)
            assignments.append(
                CreditAssignment(
                    concept_id=cid,
                    credit=base_credit,
                    path=[cid],
                    method="direct",
                )
            )

        # Ancestor credit via DAG paths with decay
        for cid in rule_concept_ids:
            if cid not in self.dag.node_ids:
                continue
            self._propagate_ancestors(cid, base_credit, [cid], seen, assignments)

        return assignments

    def _propagate_ancestors(
        self,
        node_id: str,
        current_credit: float,
        path: list[str],
        seen: set[str],
        assignments: list[CreditAssignment],
        depth: int = 0,
    ) -> None:
        """Recursively propagate credit to ancestors."""
        if depth >= self.max_depth:
            return

        for parent_id in self.dag.parents(node_id):
            if parent_id in seen:
                continue

            edge_weight = self.dag.edge_strength(parent_id, node_id)
            ancestor_credit = current_credit * self.decay_factor * edge_weight

            if abs(ancestor_credit) < self.min_credit:
                continue

            seen.add(parent_id)
            ancestor_path = [parent_id, *path]
            assignments.append(
                CreditAssignment(
                    concept_id=parent_id,
                    credit=ancestor_credit,
                    path=ancestor_path,
                    method="ancestor",
                )
            )

            # Continue propagation
            self._propagate_ancestors(
                parent_id, ancestor_credit, ancestor_path, seen, assignments, depth + 1
            )

    @staticmethod
    def to_posterior_updates(
        assignments: list[CreditAssignment],
    ) -> dict[str, dict[str, float]]:
        """Convert credit assignments to Thompson Sampling posterior updates.

        Positive credit → alpha_delta (success count).
        Negative credit → beta_delta (failure count).

        Returns:
            dict mapping concept_id → {"alpha_delta": float, "beta_delta": float}
        """
        updates: dict[str, dict[str, float]] = {}

        for a in assignments:
            if a.concept_id not in updates:
                updates[a.concept_id] = {"alpha_delta": 0.0, "beta_delta": 0.0}

            if a.credit >= 0:
                updates[a.concept_id]["alpha_delta"] += a.credit
            else:
                updates[a.concept_id]["beta_delta"] += abs(a.credit)

        return updates
