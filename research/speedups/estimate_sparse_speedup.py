#!/usr/bin/env python3
"""Estimate CPU speedups from sparse Jacobian operations.

Performs mathematical FLOP counting and actual benchmarks for
dense vs. sparse linear algebra operations on the CPU.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.parsers import UCLCHEMParser

# Force JAX to use CPU for this benchmark
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def estimate_speedups(network_name: str = "gas_phase_only"):
    print(f"Estimating CPU sparse speedups for: {network_name}")
    print(f"JAX Platform: {jax.devices()[0].platform}")

    # --- 1. Load Network & Compute Jacobian Structure ---
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    network_file = data_dir / "uclchem_gas_phase_only.csv"
    ic_file = base_dir / "benchmarks/initial_conditions/gas_phase_only_initial.yaml"

    parser = UCLCHEMParser(cloud_radius_pc=1.0, number_density=1e4)
    network = parser.parse_network(str(network_file))
    jnetwork = network.get_ode()
    n_species = len(network.species)

    # Compute Jacobian at random point
    key = jax.random.PRNGKey(42)
    y_random = jax.random.uniform(key, shape=(n_species,), minval=1e-10, maxval=1e-5)

    def ode_func(y):
        return jnetwork(0.0, y, 100.0, 1.3e-17, 1.0, 2.0)

    print("\nComputing Jacobian...")
    J = jax.jit(jax.jacfwd(ode_func))(y_random)
    J_np = np.array(J)

    # Analyze Structure
    nnz = np.count_nonzero(J_np)
    N = n_species
    sparsity = 1.0 - (nnz / (N * N))

    print(f"Matrix Size (N): {N}")
    print(f"Non-zeros (NNZ): {nnz}")
    print(f"Sparsity: {sparsity:.2%}")

    # --- 2. Mathematical FLOP Estimation ---
    print("\n--- Theoretical FLOP Estimation ---")

    # Dense LU Decomposition: (2/3) * N^3
    dense_lu_flops = (2 / 3) * (N**3)

    # Sparse LU Decomposition: Depends on fill-in, roughly proportional to NNZ * (Average Bandwidth or Fill factor)
    # A common approximation for sparse direct solvers is O(NNZ^1.5) or O(N^2) depending on structure
    # For random sparse matrices, it can be high. For chemical networks, we often have structure.
    # Let's use a conservative estimate based on NNZ and typical fill-in factor k=2 to 5
    # Sparse Ops ~ k * NNZ * sqrt(N) is one heuristic, but let's stick to simple matrix-vector for now
    # as factorization is complex to estimate purely mathematically without symbolic analysis.

    # Let's look at Matrix-Vector Multiplication (needed for iterative solvers like GMRES)
    # Dense MatVec: 2 * N^2
    dense_mv_flops = 2 * (N**2)

    # Sparse MatVec: 2 * NNZ
    sparse_mv_flops = 2 * nnz

    mv_speedup_theoretical = dense_mv_flops / sparse_mv_flops

    print(f"Dense MatVec FLOPs: {dense_mv_flops:.2e}")
    print(f"Sparse MatVec FLOPs: {sparse_mv_flops:.2e}")
    print(f"Theoretical MatVec Speedup: {mv_speedup_theoretical:.2f}x")

    # --- 3. Actual CPU Benchmarking ---
    print("\n--- Actual CPU Benchmarking (averaged over 1000 runs) ---")

    # Dense Solve (LU)
    # Using numpy/scipy directly to avoid JAX compilation overhead variability for this specific micro-benchmark
    # We want to measure the raw linear algebra speed.

    b = np.random.rand(N)

    # Benchmark Dense Solve
    start = time.perf_counter()
    for _ in range(1000):
        np.linalg.solve(J_np, b)
    dense_time = (time.perf_counter() - start) / 1000

    # Benchmark Sparse Solve (using scipy.sparse.linalg.spsolve)
    J_sp = sp.csc_matrix(J_np)
    # Pre-factorization (analyze) is often done once, but let's measure full solve first
    start = time.perf_counter()
    for _ in range(1000):
        spla.spsolve(J_sp, b)
    sparse_time = (time.perf_counter() - start) / 1000

    print(f"Dense Solve Time: {dense_time * 1e6:.2f} µs")
    print(f"Sparse Solve Time: {sparse_time * 1e6:.2f} µs")
    print(f"Actual Solve Speedup: {dense_time / sparse_time:.2f}x")

    # Benchmark Sparse Matrix-Vector (for iterative solvers)
    start = time.perf_counter()
    for _ in range(10000):
        J_np @ b
    dense_mv_time = (time.perf_counter() - start) / 10000

    start = time.perf_counter()
    for _ in range(10000):
        J_sp @ b
    sparse_mv_time = (time.perf_counter() - start) / 10000

    print(f"Dense MatVec Time: {dense_mv_time * 1e6:.2f} µs")
    print(f"Sparse MatVec Time: {sparse_mv_time * 1e6:.2f} µs")
    print(f"Actual MatVec Speedup: {dense_mv_time / sparse_mv_time:.2f}x")

    # --- 4. Memory Bandwidth Implications ---
    print("\n--- Memory Bandwidth Implications ---")
    # Dense Matrix: N*N * 8 bytes (float64)
    dense_mem = N * N * 8
    # Sparse Matrix (CSC): NNZ * 8 (values) + NNZ * 4 (row indices) + (N+1) * 4 (col ptrs)
    sparse_mem = nnz * 8 + nnz * 4 + (N + 1) * 4

    print(f"Dense Memory: {dense_mem / 1024:.2f} KB")
    print(f"Sparse Memory: {sparse_mem / 1024:.2f} KB")
    print(f"Memory Reduction: {dense_mem / sparse_mem:.2f}x")
    print("Lower memory footprint = better cache locality = faster CPU execution.")

    # --- 5. Generate Report ---
    report_path = Path("docs/cpu_sparse_speedup_plan.md")
    report_content = f"""# CPU Sparse Speedup Plan & Analysis

## Mathematical Analysis
For the `{network_name}` network ($N={N}$):

*   **Sparsity:** {sparsity:.2%}
*   **Theoretical Op Reduction (MatVec):** {mv_speedup_theoretical:.2f}x
*   **Memory Reduction:** {dense_mem / sparse_mem:.2f}x

## Benchmarks (CPU)
*   **Linear Solve Speedup:** {dense_time / sparse_time:.2f}x
*   **Matrix-Vector Speedup:** {dense_mv_time / sparse_mv_time:.2f}x

## Plan for Implementation

1.  **Iterative Solvers (GMRES):**
    The high speedup in Matrix-Vector products ({dense_mv_time / sparse_mv_time:.2f}x) suggests that using an iterative solver like GMRES with a sparse Jacobian operator will be highly effective, provided the condition number is managed.

2.  **Sparse Direct Solvers:**
    For this system size ($N={N}$), the overhead of sparse indexing in direct solvers might outweigh the FLOP reduction compared to highly optimized dense BLAS routines (as seen in the solve speedup of {dense_time / sparse_time:.2f}x).
    *   *Recommendation:* Use Sparse GMRES for large steps.
    *   *Recommendation:* Stick to Dense LU for small networks unless $N > 500$.

3.  **JAX Implementation:**
    *   Use `jax.experimental.sparse.BCOO` for the Jacobian.
    *   Use `diffrax.ODETerm` with a custom Jacobian function that returns a sparse matrix.
    *   Configure `diffrax.Kvaerno5` to use an iterative linear solver.

"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    estimate_speedups()
