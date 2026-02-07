"""DAGRefuter — statistical refutation of d-separation claims.

Uses scipy chi2_contingency when available.
Degrades gracefully: if scipy is not installed, ``DAGRefuter`` can still
be imported but ``test_independence`` raises ImportError.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .types import IndependenceAssertion

logger = logging.getLogger(__name__)

try:
    from scipy.stats import chi2_contingency

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class RefutationResult:
    """Outcome of a statistical independence test."""

    assertion: IndependenceAssertion
    test_statistic: float
    p_value: float
    dof: int
    consistent: bool  # Does the data agree with the structural claim?
    sample_size: int


@dataclass
class DAGRefuter:
    """Refute d-separation assertions against observed data.

    Requires ``scipy`` (``pip install scipy``).
    """

    significance_level: float = 0.05

    def test_independence(
        self,
        assertion: IndependenceAssertion,
        data: dict[str, list],
    ) -> RefutationResult:
        """Test a single independence assertion using chi-squared CI test.

        Args:
            assertion: The d-separation claim to test.
            data: Column-oriented data — keys are node ids, values are
                  lists of observed categorical values (same length).

        Returns:
            RefutationResult with test statistics and consistency check.

        Raises:
            ImportError: If scipy is not installed.
            ValueError: If data is insufficient for the test.
        """
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for refutation: pip install scipy")

        # Validate data availability
        all_vars = assertion.x | assertion.y | assertion.z
        missing = all_vars - set(data.keys())
        if missing:
            raise ValueError(f"Missing data for variables: {missing}")

        n = len(next(iter(data.values())))
        if n < 5:
            raise ValueError(f"Insufficient data: {n} samples (need >= 5)")

        # Build contingency table
        x_vals = self._combine_columns(assertion.x, data)
        y_vals = self._combine_columns(assertion.y, data)

        if assertion.z:
            # Conditional independence: stratify by Z, pool chi-squared
            z_vals = self._combine_columns(assertion.z, data)
            return self._test_conditional(assertion, x_vals, y_vals, z_vals, n)

        # Marginal independence
        return self._test_marginal(assertion, x_vals, y_vals, n)

    def refute_all(
        self,
        assertions: list[IndependenceAssertion],
        data: dict[str, list],
    ) -> list[RefutationResult]:
        """Refute all assertions, skipping those with insufficient data."""
        results: list[RefutationResult] = []
        for assertion in assertions:
            try:
                result = self.test_independence(assertion, data)
                results.append(result)
            except ValueError as e:
                logger.debug("Skipping assertion %s: %s", assertion, e)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_columns(var_set: frozenset[str], data: dict[str, list]) -> list[str]:
        """Combine multiple variable columns into a single joint column."""
        cols = sorted(var_set)
        if len(cols) == 1:
            return [str(v) for v in data[cols[0]]]
        return ["|".join(str(data[c][i]) for c in cols) for i in range(len(data[cols[0]]))]

    def _test_marginal(
        self,
        assertion: IndependenceAssertion,
        x_vals: list[str],
        y_vals: list[str],
        n: int,
    ) -> RefutationResult:
        """Chi-squared test for marginal independence."""
        table = self._contingency_table(x_vals, y_vals)
        stat, p, dof, _ = chi2_contingency(table)

        # Independent if p > significance (fail to reject null of independence)
        data_says_independent = p > self.significance_level
        consistent = data_says_independent == assertion.is_independent

        return RefutationResult(
            assertion=assertion,
            test_statistic=float(stat),
            p_value=float(p),
            dof=int(dof),
            consistent=consistent,
            sample_size=n,
        )

    def _test_conditional(
        self,
        assertion: IndependenceAssertion,
        x_vals: list[str],
        y_vals: list[str],
        z_vals: list[str],
        n: int,
    ) -> RefutationResult:
        """Stratified chi-squared test for conditional independence.

        Pools test statistics across strata of Z (Cochran-Mantel-Haenszel style).
        """
        # Group by Z strata
        strata: dict[str, list[int]] = {}
        for i, z in enumerate(z_vals):
            strata.setdefault(z, []).append(i)

        total_stat = 0.0
        total_dof = 0

        for indices in strata.values():
            if len(indices) < 5:
                continue  # Skip small strata
            x_stratum = [x_vals[i] for i in indices]
            y_stratum = [y_vals[i] for i in indices]

            table = self._contingency_table(x_stratum, y_stratum)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue  # Need at least 2x2

            stat, _p, dof, _ = chi2_contingency(table)
            total_stat += stat
            total_dof += dof

        if total_dof == 0:
            # Not enough data for any stratum
            raise ValueError("Insufficient data in all strata for conditional test")

        from scipy.stats import chi2

        pooled_p = float(1.0 - chi2.cdf(total_stat, total_dof))
        data_says_independent = pooled_p > self.significance_level
        consistent = data_says_independent == assertion.is_independent

        return RefutationResult(
            assertion=assertion,
            test_statistic=total_stat,
            p_value=pooled_p,
            dof=total_dof,
            consistent=consistent,
            sample_size=n,
        )

    @staticmethod
    def _contingency_table(x_vals: list[str], y_vals: list[str]) -> np.ndarray:
        """Build a contingency table from two categorical columns."""
        x_labels = sorted(set(x_vals))
        y_labels = sorted(set(y_vals))
        x_map = {v: i for i, v in enumerate(x_labels)}
        y_map = {v: i for i, v in enumerate(y_labels)}

        table = np.zeros((len(x_labels), len(y_labels)), dtype=int)
        for xv, yv in zip(x_vals, y_vals):
            table[x_map[xv], y_map[yv]] += 1

        return table
