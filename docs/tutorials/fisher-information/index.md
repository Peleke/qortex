# The Geometry of Learning

You built a Thompson Sampling bandit. It picks rules by sampling from Beta distributions. When a rule catches a real bug, you bump alpha. When it doesn't, you bump beta. Simple.

Then you plot the trajectories and notice something wrong. Two updates that move alpha by exactly 1 feel completely different. One changes everything. The other changes nothing. Your coordinate system is lying to you.

This series introduces the Fisher information metric: the geometry that tells the truth about how much the system is learning, where, and how fast.

## The series

| Tutorial | What it covers |
|----------|----------------|
| [Your Coordinates Are Lying to You](part1-fisher-metric.md) | The Fisher information matrix; why Euclidean distance is meaningless for beliefs |
| [Paths Through Belief Space](part2-trajectories.md) | Real bandit trajectories with Fisher-speed coloring; entropy convergence |
| [The Shortest Path Is Curved](part3-geodesics.md) | Christoffel symbols, the geodesic equation, and solving it with scipy |
| [Where the Manifold Bends](part4-curvature.md) | Gaussian curvature; the GR correspondence; critical learning moments |
| [The Update Rule as a Dynamical System](part5-dynamical-systems.md) | Fixed points, stability, phase portraits, Lyapunov exponents |
| [What the System Should Feel](part6-interoception.md) | Noether's theorem, conserved quantities, affect signals, the interoception circuit |

## What you need

- Python with scipy, numpy, matplotlib
- `bandit_state.jsonl` from a buildlog Thompson Sampling run (real data)
- Comfort with plotting and basic calculus (derivatives, integrals)
- No differential geometry background required (we earn every term)

## Depth note

Chapters 1-4 are deep: full implementations, real data, runnable code. Chapters 5-6 are sketches: enough to build intuition, write pseudocode, and see the architecture. The full dynamical systems and Noether treatment requires research that's still in progress (see the [roadmap](../the-road-ahead.md)).

## Where this leads

The Fisher layer answers "is the system learning?" The companion series, [Causal Reasoning](../causal-dag/index.md), answers "what leads to what?" Together, they form two halves of a system that knows what it knows, how fast it's learning, and which relationships actually matter.

After both series, a [roadmap page](../the-road-ahead.md) maps the horizon: Hamiltonian mechanics on belief space, conservation laws as a sensory apparatus, and the affect signals that close the loop from geometry to action.
