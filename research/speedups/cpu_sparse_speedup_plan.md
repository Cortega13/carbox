# CPU Sparse Speedup Plan & Analysis

## Mathematical Analysis
For the `gas_phase_only` network ($N=163$):

*   **Sparsity:** 79.05%
*   **Theoretical Op Reduction (MatVec):** 4.77x
*   **Memory Reduction:** 3.15x

## Benchmarks (CPU)
*   **Linear Solve Speedup:** 0.18x
*   **Matrix-Vector Speedup:** 0.53x

## Plan for Implementation

1.  **Interpretation of Results:**
    *   **Small Network Paradox:** For $N=163$, dense operations are significantly faster (Linear Solve: 5.5x faster, MatVec: 1.9x faster) despite 79% sparsity.
    *   **Reason:** The overhead of sparse indexing and non-contiguous memory access on the CPU outweighs the FLOP reduction for matrices that fit comfortably in the L2 cache (207 KB).
    *   **Threshold:** Mathematical extrapolation suggests sparse solvers will overtake dense solvers only when $N > 500-1000$ (e.g., when adding grain surface chemistry).

2.  **Recommendation for `gas_phase_only` ($N=163$):**
    *   **Stick to Dense Solvers:** Use `diffrax.Kvaerno5` with standard dense LU decomposition. It is currently optimal for this specific network size on CPU.
    *   **Focus on QSSA:** The primary speedup will come from reducing stiffness (QSSA for `SIH+`, `H2+`) rather than linear algebra optimization.

3.  **Future-Proofing for Larger Networks:**
    *   Implement `jax.experimental.sparse` support now, but keep it behind a configuration flag (`use_sparse_solver=False` default).
    *   When the network grows (e.g., adding 300+ surface species), the sparse solver will become essential as the dense $O(N^3)$ scaling hits.

3.  **JAX Implementation:**
    *   Use `jax.experimental.sparse.BCOO` for the Jacobian.
    *   Use `diffrax.ODETerm` with a custom Jacobian function that returns a sparse matrix.
    *   Configure `diffrax.Kvaerno5` to use an iterative linear solver.

