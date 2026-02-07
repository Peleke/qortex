# Build Journal: Mnemosyne Phase 1 — Pyro-Aware Causal Reasoning Layer

**Date:** 2026-02-07
**Duration:** ~1 hour
**Status:** Complete

---

## The Goal

Implement the causal reasoning module for qortex (Issue #3 / #26). The key design constraint: build for Pyro/ChiRho from the start but ship Phase 1 with networkx d-separation only, degrading gracefully. The Thompson Sampling Beta posteriors in buildlog are already points on a statistical manifold — the type system carries enough information for those points to become Pyro sample sites in Phase 2 without rewriting anything.

---

## What We Built

### Architecture

```
                   CausalDispatcher (dispatch.py)
                   ┌─────────────────────────────┐
                   │ auto_detect(dag)             │
                   │   chirho → pyro → dowhy → nx │
                   └──────────┬──────────────────┘
                              │
                   ┌──────────v──────────┐
                   │ CausalBackend       │  ← Protocol
                   │   capabilities()    │
                   │   query()           │
                   │   is_d_separated()  │
                   └──────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    NetworkXCausal    (Phase 2: Pyro)  (Phase 3: ChiRho)
        Backend
              │
    ┌─────────┴─────────┐
    │  CausalDAG        │  ← networkx.DiGraph wrapper
    │  DSeparationEngine│  ← nx.is_d_separator()
    └───────────────────┘
              │
    ┌─────────┴─────────┐     ┌──────────────────┐
    │ CausalRuleProjector│────>│ Projection        │
    │ (ProjectionSource) │     │   pipeline        │
    └────────────────────┘     └──────────────────┘
              │
    ┌─────────┴─────────┐
    │ CreditAssigner    │────> Thompson Sampling
    │   (DAG ancestry)  │     posterior updates
    └───────────────────┘
```

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| types.py | Working | Enums, dataclasses, Pyro-ready fields |
| dag.py | Working | CausalDAG with cycle-breaking |
| dsep.py | Working | nx.is_d_separator (not nx.d_separated — API changed in nx 3.3+) |
| backend.py | Working | Protocol + NetworkXCausalBackend |
| refutation.py | Working | scipy optional, graceful degradation |
| projector.py | Working | Implements ProjectionSource protocol |
| credit.py | Working | DAG ancestry credit with decay |
| dispatch.py | Working | Degradation chain, stubs for Phase 2/3 |

---

## The Journey

### Phase 1: Type System + DAG + D-Separation

**What we tried:**
Implemented types.py with Pyro-aware fields (distribution_family, parameter_priors, functional_form, learned_params) defaulting to None. Built CausalDAG with from_edges() and from_backend() constructors. Used nx.d_separated() for d-separation.

**What happened:**
```
AttributeError: module 'networkx' has no attribute 'd_separated'
```

**The fix:**
networkx >= 3.3 replaced `nx.d_separated()` with `nx.is_d_separator()`. Updated dsep.py and all tests.

**Lesson:**
Always check the installed version's actual API — documentation for `nx.d_separated` was for older nx 3.0-3.2.

### Phase 2: Projector + Credit + Wiring

**What we tried:**
CausalRuleProjector implementing ProjectionSource, CreditAssigner with ancestor propagation, wiring into existing core/models.py and projectors/models.py.

**What happened:**
Clean pass on first try after the nx API fix. 78 tests passed, 7 skipped (scipy), 0 regressions across 485 total tests.

---

## Test Results

### Causal Test Suite

**Command:**
```bash
uv run pytest tests/causal/ -v
```

**Result:** 78 passed, 7 skipped (scipy not installed — correct behavior)

### Full Regression Suite

**Command:**
```bash
uv run pytest tests/ -v
```

**Result:** 485 passed, 41 skipped, 0 failures

### Type Check + Lint

```bash
uv run mypy src/qortex/causal/     # Success: no issues found in 9 source files
uv run ruff check src/qortex/causal/ tests/causal/  # All checks passed!
```

---

## What's Left

- [ ] Phase 1.5: DoWhy backend (identification + estimation)
- [ ] Phase 2: Pyro backend (learned structural equations, populate distribution_family / functional_form)
- [ ] Phase 3: ChiRho backend (counterfactual reasoning via world-splitting)
- [ ] Integration with buildlog #131 (rule-level attribution consuming CreditAssigner output)
- [ ] Interoception: affect engine consuming manifold geometry

---

## Improvements

### Architectural

- Pyro-ready type system with None defaults was the right call — zero Phase 2 friction
- Protocol-first design (CausalBackend) mirrors GraphBackend pattern, keeps dispatch clean

### Domain Knowledge

- networkx >= 3.3 uses `nx.is_d_separator(G, x, y, z)` not `nx.d_separated(G, x, y, z)` — the old API was removed
- `nx.find_minimal_d_separator()` also exists in nx 3.3+ but we rolled our own for pedagogical/control reasons

---

## Files Changed

```
src/qortex/causal/
├── __init__.py      # Re-exports public API
├── types.py         # Enums, dataclasses, RELATION_CAUSAL_DIRECTION
├── dag.py           # CausalDAG wrapping nx.DiGraph
├── dsep.py          # DSeparationEngine
├── backend.py       # CausalBackend protocol + NetworkXCausalBackend
├── refutation.py    # DAGRefuter (scipy optional)
├── projector.py     # CausalRuleProjector → ProjectionSource
├── credit.py        # CreditAssigner → Thompson Sampling
├── dispatch.py      # CausalDispatcher degradation chain

src/qortex/core/models.py        # +derivation="causal"
src/qortex/projectors/models.py  # +derivation="causal"
src/qortex/projectors/sources/flat.py  # +causal in filter checks
pyproject.toml                    # +causal dep groups

tests/causal/
├── conftest.py      # chain, fork, collider, sprinkler, smoking fixtures
├── test_types.py    # 15 tests
├── test_dag.py      # 16 tests (incl. hypothesis property-based)
├── test_dsep.py     # 16 tests
├── test_refutation.py # 7 tests (scipy-gated)
├── test_projector.py  # 12 tests
├── test_credit.py   # 10 tests
├── test_dispatch.py # 6 tests
```

---

*Next entry: Phase 1.5 — DoWhy integration for causal effect estimation*
