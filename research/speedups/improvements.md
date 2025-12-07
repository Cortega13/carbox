# Chemical Network Optimization & Speedup Strategies

This document outlines potential strategies for accelerating the `carbox` chemical kinetics solver, informed by stiffness analysis of the `uclchem_gas_phase_only` network.

## 1. Stiffness Analysis & The Quasi-Steady State Approximation (QSSA)

The stiffness analysis (Run: `stiffness_gas_phase_only`) reveals extremely high stiffness ratios ranging from **$1.37 \times 10^{18}$** to **$2.09 \times 10^{20}$**. This confirms the absolute necessity of implicit solvers (currently `kvaerno5`).

However, the analysis identifies specific culprits for the fastest timescales:

*   **Early Evolution ($t < 10^4$ yr):** The stiffness is dominated by **`SIH+`** (Silicon Hydride ion).
    *   Timescale: $\sim 10^5$ seconds ($\sim 10^{-3}$ years).
    *   Abundance: Very low, ranging from $10^{-26}$ to $10^{-10}$.
*   **Late Evolution ($t > 10^4$ yr):** The stiffness is dominated by **`H2+`**.
    *   Timescale: $\sim 10^5$ seconds.
    *   Abundance: Extremely low, $< 10^{-15}$.

### Proposal: Implement QSSA for Limiting Species
Since these species react on timescales orders of magnitude faster than the simulation flow and maintain very low abundances, they are prime candidates for the **Quasi-Steady State Approximation (QSSA)**.

**Strategy:**
1.  **Identify QSSA Species:** Flag `SIH+`, `H2+`, and potentially other radical ions with lifetimes $\tau < \Delta t_{min}$.
2.  **Algebraic Substitution:** Instead of integrating $\frac{dy}{dt} = P - L \cdot y$, assume equilibrium $\frac{dy}{dt} \approx 0 \implies y_{eq} = \frac{P}{L}$.
3.  **Reduced ODE System:** Remove these species from the state vector $y$ passed to the ODE solver. Calculate their abundances algebraically at each step to compute reaction rates for other species.

**Expected Gain:**
*   Drastic reduction in the spectral radius of the Jacobian.
*   Allows the ODE solver to take significantly larger time steps.
*   Potential to switch to less expensive solvers (e.g., explicit or lower-order implicit) if the remaining stiffness is manageable.

## 2. Jacobian & Linear Algebra Optimizations

The current solver uses `jax.jacfwd` to compute the Jacobian and dense linear algebra for the Newton steps.

### Sparse Jacobian Support
Analysis of the `gas_phase_only` network confirms significant sparsity.
*   **Matrix Size:** $163 \times 163$ (26,569 elements)
*   **Non-zero Elements:** 5,565
*   **Sparsity:** **79.05%** (Zeros) / 20.95% (Fill-in)

The relatively high fill-in (~21%) is due to highly connected "hub" species that interact with many others:
*   **Electron (E-):** Affects 142 species / Affected by 125 species
*   **H / H+:** Affects >100 species / Affected by >110 species
*   **He+:** Affects 113 species

**Strategy:**
*   **Sparsity Pattern:** The 79% sparsity is sufficient to warrant sparse solvers, but the structure is not band-diagonal. It is a "hub-and-spoke" structure.
*   **Sparse Linear Solvers:** Use `jax.experimental.sparse` with `diffrax`. Given the size (163x163), dense solvers are still quite fast on GPU, but for CPU or larger networks (e.g., grain surface chemistry), sparse solvers will be critical.
*   **Analytical Jacobian:** While JAX AD is fast, a manually defined sparse Jacobian (constructed from the stoichiometry) might be faster to evaluate and naturally sparse.

### Preconditioning
If using iterative linear solvers (like GMRES), the convergence depends heavily on the condition number.
*   **Physics-based Preconditioner:** Group species by chemical family (e.g., C-bearing, O-bearing) or by timescale to build a block-diagonal preconditioner.

## 3. Network Reduction

The network contains 2227 reactions. Not all pathways carry significant flux throughout the simulation.

**Strategy:**
*   **Flux Analysis:** Analyze the contribution of each reaction to the total production/loss rates of species across the trajectory.
*   **Pruning:** Remove reactions that contribute $< \epsilon$ (e.g., 1%) to the flux of *any* species at *any* time.
*   **Reduced Networks:** Generate "Light", "Medium", and "Heavy" versions of the network. Use the lighter networks for rapid prototyping or parameter sweeps where high precision isn't required.

## 4. Implementation Optimizations (JAX)

### Precision Handling
The current setup enforces `jax_enable_x64`.
*   **Mixed Precision:** Investigate if `float32` is sufficient for the non-stiff parts or the preconditioning steps, keeping `float64` only for the critical accumulation steps. *Note: Chemical kinetics are notoriously sensitive to precision, so this requires careful validation.*

### Custom Kernels
*   **Reaction Rate Evaluation:** The `JNetwork` vectorizes reaction rate calculations. Ensure this compiles to efficient kernels (fused multiply-add) on the target hardware (GPU/AVX).
*   **Graph Compilation:** Verify that the overhead of `diffrax`'s internal graph compilation isn't dominating for small networks.

## 5. Summary of Priorities

1.  **High:** Implement QSSA for `SIH+` and `H2+`. This directly addresses the identified stiffness source.
2.  **Medium:** Analyze Jacobian sparsity and switch to sparse linear solvers if sparsity > 80%.
3.  **Low:** Network pruning (requires careful validation against the full network).
