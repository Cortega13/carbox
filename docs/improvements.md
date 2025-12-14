# Carbox Benchmark Speedup Strategies

This document outlines brainstorming strategies to optimize `benchmarks/cosmicai/batch_run_carbox_benchmark.py` for CPU-based execution, focusing on minimizing code changes while maximizing throughput.

## 1. Vectorization with `jax.vmap` (High Impact)

**Concept:**
Currently, `process_tracer` solves one ODE system at a time. JAX incurs overhead for every function call (dispatch, Python control flow). By processing tracers in batches (e.g., 64 or 128 at a time), we can amortize this overhead and leverage CPU SIMD instructions.

**Implementation Plan:**
Instead of passing a single tracer to the solver, we pass a batch of tracers. We can define a mapped version of the solver core locally in the worker.

```python
# Pseudo-code for batched solver
# We map over y0 (initial conditions) and the time-dependent physical parameters
batch_solve = jax.vmap(
    solve_network_core,
    in_axes=(
        None,  # jnetwork
        0,     # y0 (batch)
        None,  # t_eval (shared)
        None,  # time_grid (shared)
        0,     # number_density (batch, time)
        0,     # temperature (batch, time)
        0,     # cr_rate (batch, time)
        0,     # fuv_field (batch, time)
        0,     # visual_extinction (batch, time)
    )
)
```

**Required Changes:**
1.  Modify the main loop to chunk `tracers` into batches (e.g., lists of 100).
2.  Create a `process_tracer_batch` function that:
    *   Prepares stacked numpy arrays for the whole batch (e.g., `densities` becomes `(batch_size, time_steps)`).
    *   Calls the `vmap`'d solver once.
    *   Iterates over the results to save outputs individually.

## 2. Hybrid Parallelism (Joblib + vmap)

**Concept:**
Since this runs on a CPU, we want to utilize all cores. JAX `vmap` on CPU typically runs on a single thread (or a few BLAS threads). To maximize utilization, we should combine `joblib` (multiprocessing) with `vmap`.

**Strategy:**
*   Keep the top-level `joblib.Parallel` structure.
*   Each worker processes a *batch* of tracers (e.g., 50-100) instead of 1.
*   Inside the worker, use `vmap` to solve that batch efficiently.

**Why it helps:**
*   Reduces the number of `joblib` tasks, lowering inter-process communication overhead.
*   Maximizes core usage (Joblib) while maximizing instruction efficiency (JAX/SIMD).

## 3. Remove Pandas Overhead

**Concept:**
The current pipeline creates a Pandas DataFrame for every tracer in `build_tracer_frame`, only to extract numpy arrays from it immediately in `process_tracer`.

```python
# Current flow
data -> slice -> DataFrame -> slice -> numpy

# Optimized flow
data -> slice -> numpy
```

**Implementation:**
*   Modify `build_tracer_frame` (or create a new `get_tracer_data`) to return a dictionary of numpy arrays or a simple dataclass holding the arrays directly.
*   This avoids the costly DataFrame construction and destruction for every single tracer, which adds up significantly when processing thousands of tracers.

## 4. Lift Common Computations

**Concept:**
Certain variables are recalculated for every tracer but are actually constant across the benchmark (assuming fixed trajectory length).

**Identified Redundancies:**
*   `time_grid = build_time_axis(orig_times)`: If all tracers have the same length and timestep (which they do in the standard benchmark), this grid is identical. It should be computed once and passed to workers.
*   `t_eval`: Similarly, the evaluation time points are constant.

**Implementation:**
*   Compute `time_grid` and `t_eval` in `main()` or `get_cached_assets()`.
*   Pass them to the solver.

## 5. Algorithmic Tuning (Trade-off)

**Concept:**
The solver uses `atol=1e-14` and `rtol=1e-6`.
*   `atol=1e-14` is extremely tight for many astrophysical applications where abundances span 20 orders of magnitude.
*   Relaxing `atol` to `1e-10` or `1e-12` could reduce the number of solver steps significantly without meaningfully affecting the scientific results for dominant species.

**Recommendation:**
*   Expose `atol` and `rtol` as command-line arguments to allow tuning.

## Summary of Proposed Plan

1.  **Refactor Data Loading**: Bypass Pandas; extract numpy arrays directly.
2.  **Batch Execution**: Implement `process_tracer_batch` using `jax.vmap`.
3.  **Optimize Parallelism**: Feed batches to `joblib` workers.
4.  **Global Constants**: Compute time grids once.
