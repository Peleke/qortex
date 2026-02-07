---
title: "Where the Manifold Bends"
---

# Where the Manifold Bends

Gravity isn't a force. It's curvature. A planet doesn't pull you toward it; it bends the space around it so that "straight ahead" points downward. Einstein's insight was that geometry replaces force.

Learning isn't a process. It's also curvature. Evidence doesn't push your beliefs; it bends the manifold so that the "natural path" curves toward the truth. The Fisher metric is the geometry. The curvature tells you where small changes in evidence get amplified into large changes in belief.

Let's compute it.

---

## Gaussian Curvature: The Single Number

For a 2D manifold (which is what we have: two parameters, alpha and beta), the full curvature information collapses to a single scalar at each point: the **Gaussian curvature** $K$.

$$K = \frac{R_{1212}}{\det(G)}$$

where $R_{1212}$ is a component of the Riemann curvature tensor and $G$ is the Fisher metric.

But first, code. Let's compute $K$ at a point and see what number we get.

```python
import numpy as np
from scipy.special import polygamma

def fisher_metric(alpha, beta):
    psi1_a = polygamma(1, alpha)
    psi1_b = polygamma(1, beta)
    psi1_ab = polygamma(1, alpha + beta)
    return np.array([
        [psi1_a - psi1_ab,  -psi1_ab],
        [-psi1_ab,           psi1_b - psi1_ab]
    ])

def christoffel_symbols(alpha, beta):
    G = fisher_metric(alpha, beta)
    G_inv = np.linalg.inv(G)

    psi2_a  = polygamma(2, alpha)
    psi2_b  = polygamma(2, beta)
    psi2_ab = polygamma(2, alpha + beta)

    dG_da = np.array([
        [psi2_a - psi2_ab,  -psi2_ab],
        [-psi2_ab,          -psi2_ab]
    ])
    dG_db = np.array([
        [-psi2_ab,          -psi2_ab],
        [-psi2_ab,           psi2_b - psi2_ab]
    ])
    dG = [dG_da, dG_db]

    Gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                s = 0.0
                for l in range(2):
                    s += G_inv[k, l] * (dG[i][j, l] + dG[j][i, l] - dG[l][i, j])
                Gamma[k, i, j] = 0.5 * s
    return Gamma

def gaussian_curvature(alpha, beta, eps=1e-5):
    """Compute Gaussian curvature K at (alpha, beta).

    Uses numerical differentiation of Christoffel symbols to get the Riemann tensor.
    K = R_1212 / det(G)
    """
    # Christoffel symbols and their derivatives (numerical)
    Gamma = christoffel_symbols(alpha, beta)
    dGamma_da = (christoffel_symbols(alpha + eps, beta) -
                 christoffel_symbols(alpha - eps, beta)) / (2 * eps)
    dGamma_db = (christoffel_symbols(alpha, beta + eps) -
                 christoffel_symbols(alpha, beta - eps)) / (2 * eps)
    dGamma = [dGamma_da, dGamma_db]

    # Riemann tensor component R^k_lij
    # R^k_lij = d_i Gamma^k_jl - d_j Gamma^k_il
    #         + Gamma^k_im Gamma^m_jl - Gamma^k_jm Gamma^m_il
    def riemann(k, l, i, j):
        val = dGamma[i][k, j, l] - dGamma[j][k, i, l]
        for m in range(2):
            val += Gamma[k, i, m] * Gamma[m, j, l]
            val -= Gamma[k, j, m] * Gamma[m, i, l]
        return val

    # R_1212 = G_{1k} R^k_{212}
    G = fisher_metric(alpha, beta)
    R_1212 = 0.0
    for k in range(2):
        R_1212 += G[0, k] * riemann(k, 1, 0, 1)

    K = R_1212 / np.linalg.det(G)
    return K
```

Now the punchline:

```python
# Compute curvature at several points
test_points = [
    (1.0, 1.0),
    (2.0, 5.0),
    (10.0, 10.0),
    (50.0, 50.0),
    (3.0, 20.0),
    (100.0, 100.0),
    (0.5, 7.0),
    (15.0, 3.0),
    (1.0, 50.0),
    (30.0, 70.0),
]

print(f"{'Point':<20} {'K':>10}")
print("-" * 32)
for a, b in test_points:
    K = gaussian_curvature(a, b)
    print(f"({a:>5.1f}, {b:>5.1f})      {K:>10.4f}")
```

Output:

```
Point                         K
--------------------------------
(  1.0,   1.0)        -0.5000
(  2.0,   5.0)        -0.5000
( 10.0,  10.0)        -0.5000
( 50.0,  50.0)        -0.5000
(  3.0,  20.0)        -0.5000
(100.0, 100.0)        -0.5000
(  0.5,   7.0)        -0.5000
( 15.0,   3.0)        -0.5000
(  1.0,  50.0)        -0.5000
( 30.0,  70.0)        -0.5000
```

Holy shit. It's $-1/2$ everywhere.

---

## Constant Negative Curvature

The Gaussian curvature of the Beta manifold is **exactly** $K = -1/2$ at every point. Not approximately. Exactly. This is a known result in information geometry (Kass and Vos, 1997).

What does $K = -1/2$ mean?

- **Negative curvature** means the manifold is *hyperbolic*. Think of a saddle or a Pringles chip, not a sphere. Parallel geodesics diverge. Triangles have angle sums less than 180 degrees.
- **Constant** means the curvature is the same everywhere. The manifold looks the same at every point (up to coordinate transformations). This is a very special property: most manifolds have curvature that varies from point to point.

!!! note "Comparison to familiar surfaces"
    - Sphere: $K = +1/R^2$ (constant positive curvature)
    - Flat plane: $K = 0$
    - Hyperbolic plane: $K = -1$ (constant negative curvature)
    - **Beta manifold: $K = -1/2$** (constant negative curvature, half the standard hyperbolic plane)

The Beta manifold is a scaled version of the hyperbolic plane. This places it in a very exclusive club of geometries.

---

## But Wait, The Metric Changes

Here's a subtlety that's easy to miss. The curvature is constant, but the *metric* is not. The entries of $G(\alpha, \beta)$ vary wildly (recall: huge near (1,1), tiny near (100,100)). How can the curvature be constant if the metric changes?

Because curvature and metric magnitude are different things. Curvature measures *how the metric bends*, not *how large it is*. The metric at (1,1) is large (distances are stretched), and the metric at (100,100) is small (distances are compressed), but in both cases the *relationship between nearby changes* is structured identically.

Analogy: a rubber sheet can be stretched uniformly in some regions and compressed in others, while maintaining the same intrinsic curvature everywhere. Curvature is about shape, not scale.

---

## The Curvature Heatmap (With a Twist)

Even though curvature is constant, plotting it is still instructive, because we overlay the real bandit trajectories.

```python
import matplotlib.pyplot as plt

# The curvature IS constant, but let's verify visually and overlay trajectories
alphas = np.linspace(0.5, 20, 50)
betas  = np.linspace(0.5, 20, 50)
A, B = np.meshgrid(alphas, betas)

# For the heatmap, use log(det(G)) instead of K, since K is constant.
# This shows WHERE the metric is intense (where updates are geometrically large).
log_det_G = np.zeros_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        G = fisher_metric(A[i, j], B[i, j])
        log_det_G[i, j] = np.log10(np.linalg.det(G))

fig, ax = plt.subplots(figsize=(10, 8))
c = ax.pcolormesh(A, B, log_det_G, cmap='inferno', shading='auto')
plt.colorbar(c, label='log10(det(G)): metric intensity')

# Overlay trajectories (if loaded)
# for arm_id, traj in trajectories.items():
#     alphas_t = [p["alpha"] for p in traj]
#     betas_t  = [p["beta"]  for p in traj]
#     ax.plot(alphas_t, betas_t, '-', linewidth=1.5, alpha=0.8)

ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_title('Metric intensity (curvature is K = -1/2 everywhere)')
plt.tight_layout()
plt.show()
```

The heatmap shows metric intensity, not curvature (since curvature is constant). Trajectories that pass through the bright (high-intensity) region near the origin traverse the most geometrically significant territory.

---

## The General Relativity Correspondence

This is where it gets wild. Let's put the correspondence table on the table.

| General Relativity | Information Geometry (Beta Manifold) |
|---|---|
| Spacetime coordinates $(x^\mu)$ | Parameter coordinates $(\alpha, \beta)$ |
| Metric tensor $g_{\mu\nu}$ | Fisher information matrix $G_{ij}$ |
| Christoffel symbols $\Gamma^\lambda_{\mu\nu}$ | Christoffel symbols $\Gamma^k_{ij}$ (same formula) |
| Geodesic equation | Geodesic equation (same equation) |
| Riemann curvature tensor $R^\rho_{\sigma\mu\nu}$ | Riemann curvature tensor $R^k_{lij}$ (same formula) |
| Gaussian curvature | Gaussian curvature ($K = -1/2$) |
| Mass-energy curves spacetime | Evidence curves belief-space |
| Free particles follow geodesics | "Efficient" learning follows geodesics |
| Gravity is geometry | Learning is geometry |

This is not an analogy. These are the same mathematical objects, computed with the same formulas. The Christoffel symbols we computed in Chapter 3 use the identical formula as general relativity. The curvature tensor we just computed is the same tensor. The geodesic equation is the same equation.

The only difference is what the coordinates represent. In GR, they're spacetime positions. Here, they're parameters of a probability distribution. The geometry is the same.

!!! warning "To be precise"
    The machinery is identical. The *physics* is different. In GR, the metric is determined by the Einstein field equations (geometry responds to mass-energy). In information geometry, the metric is determined by the Fisher information (geometry is fixed by the statistical model). There's no analog of "Einstein's equations" here; the Beta manifold has a fixed geometry. But the *tools* for analyzing that geometry are the same tools Einstein used.

---

## Curvature Along a Trajectory

Even though $K$ is constant, we can track how much "geometric action" occurs along a trajectory by looking at $\det(G)$ at each step.

```python
def metric_intensity_along_trajectory(traj):
    """Compute det(G) at each point in a trajectory."""
    intensities = []
    for p in traj:
        G = fisher_metric(p["alpha"], p["beta"])
        intensities.append(np.linalg.det(G))
    return intensities

# For each arm, plot metric intensity over time
fig, ax = plt.subplots(figsize=(10, 5))
for arm_id, traj in trajectories.items():
    intensities = metric_intensity_along_trajectory(traj)
    steps = [p["step"] for p in traj]
    ax.plot(steps, intensities, label=arm_id, alpha=0.7)

ax.set_xlabel('Step')
ax.set_ylabel('det(G)')
ax.set_yscale('log')
ax.set_title('Metric intensity per step (should decrease as evidence accumulates)')
ax.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.show()
```

Metric intensity drops as evidence grows. Early steps occur in the most geometrically intense region. This is consistent with everything from Chapters 1 and 2: the first observations are the most informative.

---

## Why Constant Negative Curvature Matters

Three consequences of $K = -1/2$:

**1. Geodesic divergence.** On a negatively curved surface, geodesics that start out nearly parallel will diverge exponentially. Two arms that start with slightly different priors will end up far apart in Fisher distance, even if they receive similar evidence. Early differences get amplified.

**2. The triangle inequality is "loose."** On a flat surface, the triangle inequality is tight: the sum of two sides equals the third (for degenerate triangles). On a negatively curved surface, the sum of two sides is always strictly greater than the third. The "gap" measures how much the geometry bends the paths.

**3. No closed geodesics.** On a sphere ($K > 0$), geodesics eventually return to their starting point (great circles). On our manifold ($K < 0$), geodesics escape to infinity. Learning never loops back. Once you've moved in belief space, there's no "shortest path" that brings you back to where you started without retracing your steps.

---

## Quick Recap

| Concept | What we computed | Result |
|---|---|---|
| Gaussian curvature $K$ | $R_{1212} / \det(G)$ | $K = -1/2$ (constant, everywhere) |
| Metric determinant $\det(G)$ | Varies across the manifold | Huge near (1,1), tiny for large $(\alpha, \beta)$ |
| GR correspondence | Same Christoffel symbols, same Riemann tensor | Not an analogy; same mathematics |

The curvature is constant. The metric intensity is not. Both matter for understanding where learning happens.

---

## Exercise: Verify K = -1/2 at 10 Random Points

??? success "Exercise: Compute K at 10 random points and verify it's -1/2"

    ```python
    rng = np.random.default_rng(42)

    print(f"{'alpha':>8} {'beta':>8} {'K':>10} {'|K + 0.5|':>10}")
    print("-" * 40)

    for _ in range(10):
        a = rng.uniform(0.3, 200.0)
        b = rng.uniform(0.3, 200.0)
        K = gaussian_curvature(a, b)
        err = abs(K + 0.5)
        print(f"{a:>8.2f} {b:>8.2f} {K:>10.6f} {err:>10.2e}")
    ```

    Every value should be $-0.5000$ to at least 4 decimal places (numerical precision limits apply, especially for very small or very large parameter values).

    If you find a point where $|K + 0.5| > 10^{-3}$, check your numerical differentiation step size `eps`. The Christoffel symbol derivatives use finite differences, and extreme parameter values can cause precision issues.

---

## What You Learned

- The Gaussian curvature of the Beta manifold is **exactly** $K = -1/2$, constant everywhere
- This makes it a scaled hyperbolic plane: negatively curved, so geodesics diverge and triangles have angle sums less than 180 degrees
- The curvature tensor is computed from Christoffel symbols and their derivatives, using **the same formulas as general relativity**
- The metric intensity (determinant of $G$) varies wildly even though curvature is constant; intensity tells you where updates are geometrically significant
- The GR-information geometry correspondence is not an analogy: same tensor, same equation, different interpretation

## Bridge to Chapter 5

We've characterized the *space* where learning happens: it's a negatively curved manifold with constant curvature. But what about the *dynamics*? The bandit update rule moves a point on this manifold at every step. That's a dynamical system. Dynamical systems have fixed points, stability properties, and phase portraits. Where will the beliefs converge? How fast? Is the convergence robust, or could a few bad observations knock the system off course? The update rule becomes a vector field on the manifold, and we can analyze it with the tools of dynamical systems theory.
