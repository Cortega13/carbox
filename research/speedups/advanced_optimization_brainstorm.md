# Advanced Optimization & Speedup Brainstorming

This document explores advanced strategies for accelerating `carbox` beyond the initial QSSA and Sparse Solver analysis. It focuses on architectural changes, low-level optimizations, and algorithmic shifts.

## 1. Analytical Jacobian Construction

Currently, `carbox` relies on JAX's Forward-Mode Automatic Differentiation (`jax.jacfwd`) to compute the Jacobian matrix required by implicit solvers. While convenient, this approach scales poorly with network size compared to a direct sparse construction.

### Mathematical Derivation of Speedup

Let $N$ be the number of species and $M$ be the number of reactions.
The state vector is $y \in \mathbb{R}^N$.
The ODE is defined as:
$$ \frac{dy}{dt} = f(y) = S \cdot R(y) $$
where:
*   $S \in \mathbb{R}^{N \times M}$ is the sparse stoichiometry matrix (Incidence matrix).
*   $R(y) \in \mathbb{R}^M$ is the vector of reaction rates.

#### Cost of Automatic Differentiation (Forward Mode)
Forward-mode AD computes the Jacobian $J = \frac{\partial f}{\partial y}$ by propagating "tangents". To compute the full Jacobian, JAX essentially evaluates the primal function vectorized over the $N$ basis vectors of the input space.

For a reaction $k$ with rate $R_k(y) = \alpha \prod_{i \in \text{reactants}} y_i$:
1.  **Primal Cost:** $O(1)$ FLOPS per reaction. Total $O(M)$.
2.  **Tangent Cost:** AD propagates a tangent matrix of size $M \times N$. Even if $S$ is sparse, the intermediate Jacobian of the rates $J_R = \frac{\partial R}{\partial y}$ is an $M \times N$ matrix (conceptually).
    *   The chain rule operation $J = S \cdot J_R$ involves multiplying $(N \times M) \times (M \times N)$.
    *   JAX's `jacfwd` typically materializes these dense intermediates or iterates over $N$ batches.
    *   **Complexity:** $O(N \cdot M)$ FLOPS.

#### Cost of Analytical Construction
We can derive the Jacobian entries directly. The element $J_{ij}$ is:
$$ J_{ij} = \sum_{k=1}^M S_{ik} \frac{\partial R_k}{\partial y_j} $$

Since $S_{ik}$ is non-zero only if species $i$ is involved in reaction $k$, and $\frac{\partial R_k}{\partial y_j}$ is non-zero only if species $j$ is a reactant in reaction $k$, we can iterate strictly over the $M$ reactions.

For each reaction $k$:
1.  Compute partial derivatives w.r.t. its reactants (typically 1 or 2).
2.  Add contributions to the specific entries in the sparse Jacobian matrix.
    *   Number of updates $\approx (\text{# reactants}) \times (\text{# species affected}) \approx 2 \times 4 = 8$ entries per reaction.
3.  **Complexity:** $O(M)$ FLOPS.

#### Theoretical Comparison

| Metric | Forward-Mode AD | Analytical Sparse | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Compute Complexity** | $O(N \cdot M)$ | $O(M)$ | $\propto N$ |
| **Memory Writes** | $N^2$ (Dense) | $NNZ \approx 10N$ (Sparse) | $\approx N/10$ |
| **Cache Efficiency** | Low (Stride $N$) | High (Linear scan of $R$) | High |

#### Case Study: Standard Network ($N=163$)
Assuming a typical reaction-to-species ratio of ~10 (so $M \approx 1600$):

1.  **Memory Footprint:**
    *   **Dense (AD):** $163^2 = 26,569$ entries. At double precision (8 bytes), this is **~212 KB** of contiguous memory writes per step.
    *   **Sparse (Analytical):** With ~10 entries per row, $NNZ \approx 1,630$. This is **~13 KB**.
    *   **Reduction:** **16x** less memory traffic.

2.  **Compute Operations:**
    *   **AD:** Effectively evaluates the rate function $N=163$ times (once per basis vector).
    *   **Analytical:** Evaluates the rate derivatives once.
    *   **Theoretical Speedup:** $\sim 163\times$ fewer FLOPs for the Jacobian term.

3.  **Real-World Impact:**
    *   While JAX overhead dampens the theoretical 163x, the **Jacobian construction itself is expected to be 10-20x faster**.
    *   Since Jacobian evaluation typically consumes ~60% of the total runtime for this size, the **overall solver speedup is estimated at ~2x**.

For a larger network ($N \approx 500, M \approx 5000$):
*   **Jacobian Construction Speedup:** Theoretically $\sim 100\times$ faster for the construction step.
*   **Total Runtime Impact:** Since Jacobian evaluation is often 50-80% of the solver time for stiff systems, the overall speedup is estimated at **1.5x - 3x**.

### Implementation Path
1.  Derive the partial derivative form $\frac{\partial R_k}{\partial y_j}$ for each reaction type (Arrhenius, Cosmic Ray, etc.).
2.  Construct a static mapping of `(reaction_index, species_index) -> non_zero_jacobian_entry`.
3.  Implement a `get_jacobian(y, args)` function that populates the values of the sparse matrix directly, bypassing JAX's AD tracing.

---

## 2. Symbolic Code Generation & Kernel Fusion

The current `JNetwork` uses vectorized JAX operations (matrix multiplication). While efficient, generic matrix multiplication reads/writes to global memory.

### The Concept
Use a symbolic math library (like SymPy) or a custom transpiler to generate a single, fused function that computes the entire ODE derivative $\frac{dy}{dt}$.

### Potential Speedup
*   **Kernel Fusion:** Instead of `Rates = K * Abundances` then `dy = Incidence @ Rates`, we generate code that loads species into registers, computes rates, and updates derivatives in a single pass.
*   **Constant Folding:** Reaction coefficients (alpha, beta, gamma) can be baked into the code as immediates (if T is constant) or loaded efficiently.
*   **Register Pressure:** For $N=163$, the entire state might not fit in registers, but significant chunks can.
*   **Estimated Gain:** 2x - 5x on ODE evaluation.

### Implementation Path
1.  Write a script to parse the network and generate raw JAX/XLA code (or even Triton/CUDA kernels) representing the specific system of equations.
2.  "Unroll" the matrix multiplication loops at compile time.

---

## 3. Mixed Precision Solvers (Float32/Float64 Hybrid)

Chemical kinetics are stiff and usually require `float64` for mass conservation and stability. However, not every part of the solver needs high precision.

### The Concept
Use `float64` for the state integration (accumulation), but use `float32` for:
1.  **Jacobian Construction:** The direction of the Newton step needs to be roughly correct, not perfect.
2.  **Linear Solve (Preconditioning):** Solving $Jx = b$ can often be done in lower precision, followed by iterative refinement.
3.  **Error Estimation:** The error controller can tolerate some noise.

### Potential Speedup
*   **Memory Bandwidth:** `float32` moves 2x data.
*   **Compute:** CPU SIMD units (AVX2/AVX-512) can process 2x as many float32 elements per cycle as float64.
*   **Estimated Gain:** 1.5x - 2x (CPU).

### Implementation Path
1.  Modify `diffrax` solver configuration to allow mixed precision.
2.  Cast to `float32` before Jacobian evaluation, cast back for the update.
3.  **Risk:** Validation is critical. Can lead to catastrophic divergence in stiff systems.

---

## 4. Rate Coefficient Lookup Tables

Reaction rates $k(T)$ usually involve expensive transcendental functions: `exp`, `pow`, `log`.

### The Concept
If the temperature range is bounded (e.g., 10K - 300K), we can precompute rate coefficients into a lookup table (L1/L2 cache on CPU) and interpolate.

### Potential Speedup
*   **Instruction Cost:** Memory fetch + Linear Interpolation is cheaper than `exp` + `pow`.
*   **Estimated Gain:** 1.2x - 1.5x (CPU, dependent on cache hits).

---

## 5. Operator Splitting (Multirate Methods)

Not all species evolve at the same speed.

### The Concept
Split the species into "Fast", "Medium", and "Slow" groups.
*   **Fast:** Integrate with implicit solver and small steps (or QSSA).
*   **Medium:** Integrate with explicit solver.
*   **Slow:** Update less frequently.

### Potential Speedup
*   **Reduced Linear Algebra:** Solving smaller implicit systems for the fast species is cheaper ($O(N_{fast}^3)$ vs $O(N_{total}^3)$).
*   **Larger Steps for Slow Species:** Slow species don't need to be updated as often.
*   **Estimated Gain:** 2x - 5x (highly dependent on the ratio of fast/slow species).

### Implementation Path
1.  Analyze timescale spectrum (already done).
2.  Implement a custom `diffrax` stepper that updates sub-components of the state vector at different frequencies.

---

## 6. Linear Conservation Laws (LCL) Reduction

Chemical networks strictly conserve elemental nuclei (H, He, C, O, etc.) and electric charge.

### The Concept
The system has inherent redundancy. If we have $N_{elem}$ conserved quantities, the rank of the Jacobian is at most $N_{species} - N_{elem}$.
We can remove $N_{elem}$ species from the ODE integration and compute them algebraically:
$$ y_{removed} = C_{total} - \sum_{i \in remaining} \nu_i y_i $$

### Potential Speedup
*   **Size Reduction:** Reduces $N$ by ~10-15 (number of elements).
*   **Conditioning:** Often removes the singular modes of the Jacobian, improving linear solver convergence.
*   **Estimated Gain:** 1.1x - 1.2x (but improves stability significantly).

---

## 7. Graph-Based Decomposition (Block-Diagonalization)

The chemical interaction graph is not random; it has structure (e.g., Carbon chemistry, Nitrogen chemistry).

### The Concept
Use graph partitioning algorithms (like METIS) to reorder the species indices such that the Jacobian becomes block-diagonal (or close to it).
$$ J \approx \begin{pmatrix} A & \epsilon \\ \epsilon & B \end{pmatrix} $$

### Potential Speedup
*   **Parallel Linear Algebra:** We can factorize blocks $A$ and $B$ independently and in parallel on different CPU cores.
*   **Preconditioning:** Block-Jacobi preconditioners become extremely effective.
*   **Estimated Gain:** 1.5x - 2x (for large networks with distinct chemical families).

---

## 8. Flux-Based Adaptive Networks

Not all reactions matter all the time.

### The Concept
Dynamically prune the network during integration. At each step:
1.  Compute all fluxes $F_k = k_j \prod y_i$.
2.  Identify "active" reactions that contribute > $\epsilon$ (e.g., $10^{-6}$) to the total rate of change of any species.
3.  Construct a temporary, smaller `JNetwork` containing only active reactions.

### Potential Speedup
*   **Reduced Computation:** Evaluates fewer rates and smaller matrix multiplications.
*   **Estimated Gain:** 1.5x - 3x (highly dependent on the physical regime).
*   **Overhead Risk:** Recompiling JAX graphs is expensive. This requires a "masking" approach (multiplying rates by 0 or 1) rather than true graph modification to be efficient in JAX.

---

## 9. Proposed Deep Analysis: Eigenmode Decomposition

To scientifically determine the best optimization strategy, we can perform a deeper mathematical analysis than just "stiffness ratio".

### The Analysis
Compute the **Eigenvectors** of the Jacobian at various snapshots.
*   **Identify Stiff Modes:** Which linear combination of species corresponds to the largest negative eigenvalues?
*   **Participation Factors:** For each stiff mode, which species contribute the most?

### The Insight
*   If a stiff mode is dominated by 1-2 species $\rightarrow$ **QSSA** those specific species.
*   If a stiff mode involves a conserved cycle (e.g., $A \leftrightarrow B$) $\rightarrow$ **LCL Reduction** or **Group QSSA**.
*   If stiff modes are isolated in a subgraph $\rightarrow$ **Operator Splitting**.

This analysis moves us from "guessing" speedups to "deriving" them from the physics.

---

## Summary of Recommendations

| Strategy | Difficulty | Risk | Potential Speedup (N=163, CPU) | Potential Speedup (N=500+, CPU) |
| :--- | :--- | :--- | :--- | :--- |
| **Analytical Jacobian** | Medium | Low | 1.5x | 3x+ |
| **Operator Splitting** | High | Medium | 2x | 5x |
| **Symbolic/Fused Kernels** | High | Low | 2x | 3x |
| **LCL Reduction** | Medium | Low | 1.1x | 1.2x |
| **Graph Decomposition** | High | Low | 1.0x | 2x |
| **Adaptive Networks** | Medium | Medium | 1.5x | 2x |
| **Mixed Precision** | Medium | High | 1.5x | 2x |
| **Lookup Tables** | Low | Low | 1.2x | 1.1x |

**Immediate Next Step:** The **Analytical Jacobian** is the most robust engineering step after QSSA. It provides safe, guaranteed speedups without numerical risks.
