# Mathematical Analysis of QSSA Speedup

This document provides a detailed mathematical analysis of the potential speedups from applying the Quasi-Steady State Approximation (QSSA) to the `gas_phase_only` network.

## 1. Problem Definition: Stiffness

The stiffness ratio $S$ is defined as the ratio of the absolute values of the largest and smallest real parts of the eigenvalues of the Jacobian $J$:

$$ S = \frac{\max_i |\text{Re}(\lambda_i)|}{\min_i |\text{Re}(\lambda_i)|} $$

From our stiffness analysis:
*   **Maximum Stiffness:** $S \approx 2.09 \times 10^{20}$
*   **Fastest Timescale:** $\tau_{fast} = \frac{1}{|\lambda_{max}|} \approx 10^5 \text{ s} \approx 3 \times 10^{-3} \text{ yr}$
*   **Limiting Species:** `SIH+` (early time), `H2+` (late time)
*   **Simulation Duration:** $T_{end} = 10^6 \text{ yr}$

### The Cost of Stiffness
To integrate this system, we currently use `diffrax.Kvaerno5`, an implicit Runge-Kutta method (SDIRK).
*   **Step Cost:** Requires solving a non-linear system $F(y) = 0$ at each stage.
*   **Newton Iteration:** Requires solving a linear system $(I - \gamma \Delta t J)x = b$.
*   **Complexity:** $O(N^3)$ for dense LU decomposition (or $O(N^2)$ per iteration for iterative solvers).

## 2. The QSSA Approach

For a fast species $y_f$ (e.g., `SIH+`), the ODE is:

$$ \frac{dy_f}{dt} = P(y) - L(y) \cdot y_f $$

where $P(y)$ is the production rate and $L(y)$ is the loss rate coefficient.
Because $\tau_f = 1/L(y)$ is very small compared to the evolution of other species, we assume instantaneous equilibrium:

$$ \frac{dy_f}{dt} \approx 0 \implies y_{f, QSSA} = \frac{P(y)}{L(y)} $$

### Mathematical Reduction
We partition the state vector $y$ into "slow" species $y_s$ and "fast" species $y_f$.
The original system of size $N$ becomes a system of size $N_s = N - N_{fast}$.

**New System:**
$$ \frac{dy_s}{dt} = f_s(y_s, y_{f, QSSA}(y_s)) $$

## 3. Theoretical Speedup Calculation

We evaluate two scenarios for speedup:

### Scenario A: Reduced Implicit Solve Cost
We stick with the implicit solver (`Kvaerno5`), but the system size is smaller.

*   **Original Complexity:** $C_{orig} \propto N^3$
*   **Reduced Complexity:** $C_{QSSA} \propto (N - N_{fast})^3 + C_{algebraic}$
*   **Algebraic Cost:** $C_{algebraic} \approx O(N)$ (to compute $P/L$).

If we remove only 2 species ($N_{fast}=2$), the speedup is negligible:
$$ \text{Speedup} \approx \left(\frac{163}{161}\right)^3 \approx 1.04\text{x} $$
**Conclusion:** Removing just a few species doesn't help the linear algebra cost significantly.

### Scenario B: Enabling Explicit Solvers (The "Holy Grail")
The real potential lies in reducing the stiffness enough to use an **explicit solver** (e.g., `Tsit5`).

*   **Explicit Step Cost:** $O(N^2)$ (Matrix-Vector) or $O(N)$ (if just function eval).
*   **Stability Constraint:** Explicit solvers are stable only if $\Delta t < C \cdot \tau_{fastest}$.

**Condition:**
If removing `SIH+` and `H2+` increases the fastest timescale from $\tau_{fast} \approx 10^5 \text{ s}$ to something much larger (e.g., $\tau_{next} \approx 100 \text{ yr}$), then:

1.  **Max Stable Step:** Increases by factor $F = \frac{\tau_{next}}{\tau_{orig}} \approx \frac{100 \text{ yr}}{3 \times 10^{-3} \text{ yr}} \approx 3 \times 10^4$.
2.  **Explicit vs Implicit Cost:**
    *   Implicit Step: $\sim 163^3 / 3 \approx 1.4 \times 10^6$ FLOPs (LU).
    *   Explicit Step: $\sim 163 \times 2000$ (Rate eval) $\approx 3 \times 10^5$ FLOPs.
    *   Explicit is $\sim 5\text{x}$ cheaper per step.

**Total Speedup Potential:**
If the stiffness is removed:
$$ \text{Speedup} \approx \frac{\text{Cost}_{implicit}}{\text{Cost}_{explicit}} \times \frac{\text{Steps}_{implicit}}{\text{Steps}_{explicit}} $$
*Note: Implicit solvers already take large steps, so $\text{Steps}_{implicit}$ is small. Explicit solvers would need $\text{Steps}_{explicit} \approx T_{end} / \tau_{next}$.*

**Critical Check:**
For explicit to be faster, we need:
$$ \Delta t_{explicit} > \frac{\text{Cost}_{explicit}}{\text{Cost}_{implicit}} \times \Delta t_{implicit} $$
Given $\Delta t_{implicit}$ can be huge ($> 10^4$ years), explicit solvers are likely **only** viable if $\tau_{next}$ is also very large ($> 1000$ years).

## 4. Benchmark Results (Initial Conditions)

Using `benchmarks/estimate_qssa_speedup.py`, we analyzed the eigenvalue spectrum at $t=0$ (using initial abundances).

*   **Fastest Timescale (All Species):** $\tau_0 \approx 9.2 \times 10^8 \text{ s} \approx 29 \text{ yr}$.
    *   *Note: This is slower than the evolved state ($10^5$ s), likely due to low initial densities of ions.*
*   **Target Step Size:** > 100 years.

### QSSA Candidates Required
To achieve an explicit time step stable for > 100 years, we must remove the top **22 species** (13% of the network).

| Removed ($k$) | New Fastest Timescale | Max Explicit Step |
| :--- | :--- | :--- |
| 1 | 31.3 yr | ~31 yr |
| 5 | 39.0 yr | ~39 yr |
| 10 | 58.4 yr | ~58 yr |
| 20 | 95.7 yr | ~96 yr |
| **22** | **> 100 yr** | **> 100 yr** |
| 50 | 297 yr | ~300 yr |

## 5. Implementation Strategy

Given that we need to remove ~22 species to gain a modest explicit step of 100 years (while implicit solvers easily take $10^4+$ year steps), **pure explicit integration is likely not the best path**.

### Recommended Hybrid Approach
Instead of aiming for a fully explicit solver, we should use QSSA to **reduce the stiffness for the Implicit Solver**.

1.  **Algebraic Reduction:** Apply QSSA to the top 2-5 fastest species (e.g., `SIH+`, `H2+`) identified in the dynamic stiffness analysis.
2.  **Reduced Jacobian:** This improves the condition number of the Jacobian matrix used in the Newton iterations of `Kvaerno5`.
3.  **Benefit:**
    *   Faster convergence of Newton iterations (fewer linear solves per step).
    *   Allows the implicit solver to take larger steps limited by *accuracy* rather than *stability* of the fastest transient modes.

### Action Plan
1.  **Dynamic QSSA:** Implement a solver that flags species with $L(y) > \text{threshold}$ at each step.
2.  **Partitioned Solver:**
    *   Solve algebraic equations for $y_{fast}$.
    *   Pass reduced $y_{slow}$ to `diffrax`.
