# Causal Reasoning in Knowledge Graphs

Your knowledge graph has hundreds of edges. REQUIRES, SUPPORTS, IMPLEMENTS, CONTRADICTS. They all look the same: source, target, confidence score. But some of those edges are load-bearing walls and others are decorative trim.

This series teaches you to tell the difference. You'll build a causal model from a real knowledge graph, learn which edges carry information and which are redundant, and propagate reward through the structure when rules succeed or fail.

## The series

| Tutorial | What it covers |
|----------|----------------|
| [Not All Edges Are Created Equal](part1-edge-directions.md) | Causal vs correlational edges; the direction map hiding in your relation types |
| [The Shape of What You Know](part2-building-the-dag.md) | Building a DAG from a messy graph; cycles, breaking them, and what you lose |
| [What Can You Learn From Here?](part3-d-separation.md) | D-separation: which information paths are open, which are blocked, and why conditioning can create dependencies |
| [Propagating What Works](part4-credit-assignment.md) | Credit assignment through DAG ancestry; connecting reward to the Thompson Sampling bandit |
| [The Degradation Chain](part5-degradation-chain.md) | The dispatcher's fallback sequence; Pyro-aware fields waiting for Phase 2; where causal meets geometry |

## What you need

- A qortex knowledge graph (from the ingestion tutorials, or the Chapter 5 fixture data)
- `pip install qortex` (the `causal` module ships with the core package)
- networkx >= 3.3 (for `nx.is_d_separator`)

## Where this leads

The causal layer answers "what leads to what?" The next series, [The Geometry of Learning](../fisher-information/index.md), answers "is the system actually learning, and how fast?" The two bind together exactly where you'd expect: credit assignment outputs tangent vectors on the Fisher manifold. The causal DAG chooses the direction; the manifold determines the distance.

After both series, a [roadmap page](../the-road-ahead.md) sketches where this all goes: dynamical systems on curved belief space, Noether's theorem for conservation laws, and the interoception layer that makes the system monitor its own learning.
